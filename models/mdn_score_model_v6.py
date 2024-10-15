import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean,scatter_add
import numpy as np
from e3nn.nn import BatchNorm
from torch_geometric.utils import  to_dense_batch
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims
from utils.mdn_utils import compute_euclidean_distances_matrix,compute_euclidean_distances_matrix_TopN
from utils.training_mdn import mdn_loss_fn,calculate_probablity
""""
Version 6: this version use surface node to replace rec node to cal mdn ,correspond to surface_score_model version3
"""
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] #+ sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim,f'{x.shape[1]}=={self.num_categorical_features} + {self.num_scalar_features} + {self.lm_embedding_dim}'
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:

            out = self.batch_norm(out)
        return out


class TensorProductScoreModelV6(torch.nn.Module):
    def __init__(self, args,t_to_sigma, device, timestep_emb_func, in_lig_edge_features=10, in_rec_edge_features = 5,sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, mdn_mode=True,
                 mdn_dropout=0, mdn_no_batchnorm=False,n_gaussians = 20):
        super(TensorProductScoreModelV6, self).__init__()
        self.args = args
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.mdn_mode = mdn_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(in_rec_edge_features + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        # self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim , ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.surface_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=[[],4], sigma_embed_dim=sigma_embed_dim)
        self.surface_edge_embedding = nn.Sequential(nn.Linear(3 + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        
        self.surface_rec_cross_edge_embedding = nn.Sequential(nn.Linear(cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        self.surface_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        lig_conv_layers= []
        # surface modules
        surface_conv_layers=[]
        residue_to_surface_conv_layers = []
        lig_conv_layers, rec_conv_layers = [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            if i ==0:
                residue_to_surface_conv_layers.append(TensorProductConvLayer(** {
                'in_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                'sh_irreps': self.sh_irreps,
                'out_irreps':  in_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }))
                rec_conv_layers.append(TensorProductConvLayer(** {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }))
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)


            if i != num_conv_layers - 1:
                surface_layer = TensorProductConvLayer(**parameters)
                surface_conv_layers.append(surface_layer)

        self.residue_to_surface_conv_layers = nn.ModuleList(residue_to_surface_conv_layers)
        self.surface_conv_layers = nn.ModuleList(surface_conv_layers)
        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)

        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)


        if self.mdn_mode:
            # only mdn model without atom type and bond type predict 
            if self.num_conv_layers >=2:
                mdn_hidden_dim = 4 * ns
            else:
                mdn_hidden_dim = 2 * ns
            self.MLP = nn.Sequential(nn.Linear(mdn_hidden_dim, mdn_hidden_dim), nn.BatchNorm1d(mdn_hidden_dim), nn.ELU(), nn.Dropout(p= mdn_dropout)) 
            self.z_pi = nn.Linear(mdn_hidden_dim, n_gaussians)
            self.z_sigma = nn.Linear(mdn_hidden_dim, n_gaussians)
            self.z_mu = nn.Linear(mdn_hidden_dim, n_gaussians)
            # ligand distance prediction
            if self.args.ligand_distance_prediction:
                self.ligand_MLP = nn.Sequential(nn.Linear(mdn_hidden_dim, mdn_hidden_dim), nn.BatchNorm1d(mdn_hidden_dim), nn.ELU(), nn.Dropout(p= mdn_dropout)) 
                self.ligand_z_pi = nn.Linear(mdn_hidden_dim, n_gaussians)
                self.ligand_z_sigma = nn.Linear(mdn_hidden_dim, n_gaussians)
                self.ligand_z_mu = nn.Linear(mdn_hidden_dim, n_gaussians)
            if self.args.atom_type_prediction:
                self.atom_types = nn.Sequential(nn.Linear(mdn_hidden_dim//2, 2*mdn_hidden_dim), nn.BatchNorm1d(2*mdn_hidden_dim), nn.ELU(), nn.Dropout(p= mdn_dropout),nn.Linear(2*mdn_hidden_dim, 119)) 
            if self.args.bond_type_prediction:
                self.bond_types = nn.Sequential(nn.Linear(mdn_hidden_dim, 2*mdn_hidden_dim), nn.BatchNorm1d(2*mdn_hidden_dim), nn.ELU(), nn.Dropout(p= mdn_dropout),nn.Linear(2*mdn_hidden_dim, 4))
            if self.args.residue_type_prediction:
                self.residue_types = nn.Sequential(nn.Linear(mdn_hidden_dim//2, 2*mdn_hidden_dim), nn.BatchNorm1d(2*mdn_hidden_dim), nn.ELU(), nn.Dropout(p= mdn_dropout),nn.Linear(2*mdn_hidden_dim, 38)) 

        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            if not no_torsion:
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )
    def forward(self, data):

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(data['receptor'].x)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        # build surface graph
        surface_node_attr,surface_edge_index, surface_edge_attr, surface_edge_sh = self.build_surface_conv_graph(data)
        surface_src, surface_dst = surface_edge_index
        surface_node_attr = self.surface_node_embedding(surface_node_attr)
        surface_edge_attr = self.surface_edge_embedding(surface_edge_attr)

        # update receptor embedding and then update embedding to surface
        rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
        rec_intra_update = self.rec_conv_layers[0](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
        rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
        rec_node_attr = rec_node_attr + rec_intra_update


        # surface ,residue cross graph builld this info will use one shot
        surface_rec_cross_edge_index, surface_rec_cross_edge_attr, surface_rec_cross_edge_sh = self.build_surface_rec_cross_conv_graph(data)
        surface_rec_cross_rec, surface_rec_cross_surface = surface_rec_cross_edge_index
        surface_rec_cross_edge_attr = self.surface_rec_cross_edge_embedding( surface_rec_cross_edge_attr)

        residue_to_surface_edge_attr_ = torch.cat([surface_rec_cross_edge_attr, rec_node_attr[ surface_rec_cross_rec, :self.ns], surface_node_attr[surface_rec_cross_surface, :self.ns]], -1)
        # just one layer for feature update ,maybe can add more layers?
        surface_inter_residue_update = self.residue_to_surface_conv_layers[0](rec_node_attr, torch.flip(surface_rec_cross_edge_index,dims = [0]), residue_to_surface_edge_attr_, surface_rec_cross_edge_sh,
                                                              out_nodes=surface_node_attr.shape[0])
        surface_node_attr = F.pad(surface_node_attr, (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
        surface_node_attr = surface_node_attr + surface_inter_residue_update


        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)
            if l != len(self.lig_conv_layers) - 1:
                surface_edge_attr_ = torch.cat([surface_edge_attr, surface_node_attr[surface_src, :self.ns], surface_node_attr[surface_dst, :self.ns]], -1)
                surface_intra_update = self.surface_conv_layers[l](surface_node_attr, surface_edge_index, surface_edge_attr_, surface_edge_sh)
            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update #+ lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                surface_node_attr = F.pad(surface_node_attr, (0, surface_intra_update.shape[-1] - surface_node_attr.shape[-1]))
                surface_node_attr = surface_node_attr + surface_intra_update
            # print('after pad',lig_node_attr.shape,rec_node_attr.shape)
        if self.mdn_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            scalar_rec_attr = torch.cat([surface_node_attr[:,:self.ns],surface_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else surface_node_attr[:,:self.ns]
            h_l_x, l_mask = to_dense_batch(scalar_lig_attr, data['ligand'].batch, fill_value=0)
            h_t_x, t_mask = to_dense_batch(scalar_rec_attr, data['surface'].batch, fill_value=0)
            h_l_pos, _ = to_dense_batch(data['ligand'].pos, data['ligand'].batch, fill_value=0)
            h_t_pos, _ = to_dense_batch(data['surface'].pos, data['surface'].batch, fill_value=0)
            # aeesrtEncountered unequal batch-sizes
            assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
            (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
            h_l_x = h_l_x.unsqueeze(-2)
            h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]
            h_t_x = h_t_x.unsqueeze(-3)
            h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
            ########
            C = torch.cat((h_l_x, h_t_x), -1)
            self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
            self.C = C = C[C_mask]
            C = self.MLP(C)
            # Get batch indexes for ligand-target combined features
            C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(self.device)
            C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]#.to(self.device)
            # Outputs interactions predictions
            pi = F.softmax(self.z_pi(C), -1)
            sigma = F.elu(self.z_sigma(C))+1.1
            mu = F.elu(self.z_mu(C))+1
            dist = compute_euclidean_distances_matrix(h_l_pos, h_t_pos)[C_mask]
            mdn_loss_interaction = mdn_loss_fn(pi, sigma, mu, dist.unsqueeze(1).detach(),dist_threhold=self.args.mdn_dist_threshold_train if self.args.mdn_dist_threshold_train is not None else 7.0)

            if self.args.ligand_distance_prediction:
                # output for ligand distance predictions
                h_l_x, l_mask = to_dense_batch(scalar_lig_attr, data['ligand'].batch, fill_value=0)
                h_l = h_l_x.unsqueeze(-2)
                h_l = h_l.repeat(1, 1, N_l, 1) # [B, N_l, N_t, C_out]
                h_t_x = h_l_x.unsqueeze(-3)
                h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
                C_ligand = torch.cat((h_l, h_t_x), -1)
                C_mask_ligand = l_mask.view(B, N_l, 1) & l_mask.view(B, 1, N_l)

                sample = 1-torch.eye(N_l).to(self.device)
                sample = sample.unsqueeze(0)
                C_mask_ligand = C_mask_ligand*sample.repeat(B, 1, 1).bool()

                self.C_ligand = C_ligand = C_ligand[C_mask_ligand]
                C_ligand = self.ligand_MLP(C_ligand)

                pi_ligand = F.softmax(self.ligand_z_pi(C_ligand), -1)
                sigma_ligand = F.elu(self.ligand_z_sigma(C_ligand))+1.1
                mu_ligand = F.elu(self.ligand_z_mu(C_ligand))+1
                dist_ligand = compute_euclidean_distances_matrix(h_l_pos, h_l_pos)[C_mask_ligand]
                mdn_loss_ligand = mdn_loss_fn(pi_ligand, sigma_ligand, mu_ligand, dist_ligand.unsqueeze(1).detach(),dist_threhold=self.args.mdn_dist_threshold_train if self.args.mdn_dist_threshold_train is not None else 7.0)
            else:
                mdn_loss_ligand = mdn_loss_interaction*0.0
            if self.args.atom_type_prediction:
            ####### node type predictions
            # ligand atom types 119
                atom_types_label = data['ligand'].x[:,0]
                atom_types_pred = self.atom_types(scalar_lig_attr)
                # classification loss
                atom_types_loss = F.cross_entropy(atom_types_pred, atom_types_label)
            else:
                atom_types_loss = mdn_loss_interaction*0.0
            if self.args.bond_type_prediction:
                    # bond types 4
                bond_types_label = torch.argmax(data['ligand', 'lig_bond', 'ligand'].edge_attr[:,:4], dim=-1, keepdim=False)
                bond_types_pred = self.bond_types(torch.cat([scalar_lig_attr[data['ligand', 'ligand'].edge_index[0]], scalar_lig_attr[data['ligand', 'ligand'].edge_index[1]]], axis=1))
                        # # classification loss
                bond_types_loss = F.cross_entropy(bond_types_pred, bond_types_label)
                # print('bond_types_loss',bond_types_loss)
            else:
                bond_types_loss = mdn_loss_interaction*0.0
            if self.args.residue_type_prediction:
            # residue types 38
                residue_types_label = data['receptor'].x[:,0].long()
                residue_types_pred = self.residue_types(scalar_rec_attr)
                residue_types_loss = F.cross_entropy(residue_types_pred, residue_types_label)
            else:
                residue_types_loss = mdn_loss_interaction*0.0
            if self.training:
                if torch.isnan(mdn_loss_interaction  + mdn_loss_ligand  + atom_types_loss + bond_types_loss + residue_types_loss):
                    print(mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss)
                return mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss

            else:
                # probx = 0.0
                # for dist in dists_list:
                prob = calculate_probablity(pi, sigma, mu, dist.unsqueeze(1).detach(),dist_threhold=self.args.mdn_dist_threshold_test if self.args.mdn_dist_threshold_test is not None else 5.0)
                probx = scatter_add(prob,C_batch, dim=0, dim_size=B)
                return mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss,probx
    def build_lig_conv_graph(self, data):
   
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)
        # compute initial features

        edge_attr = edge_attr#torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = data['ligand'].x#torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        # 
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh
    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings

        node_attr = data['receptor'].x#torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]

        edge_attr =torch.cat([data['receptor', 'rec_contact', 'receptor'].edge_attr, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh
    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=30)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)
        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return bonds, edge_index, edge_attr, edge_sh
    def build_surface_conv_graph(self, data):
        node_attr = torch.nan_to_num(data['surface'].x)
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['surface','surface_edge','surface'].edge_index
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['surface'].pos[src.long()]
        edge_length_emb = self.surface_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['surface'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([data['surface','surface_edge','surface'].edge_attr, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh
    
    def build_surface_rec_cross_conv_graph(self, data, cross_distance_cutoff = 15):
        edge_index = radius(data['surface'].pos, data['receptor'].pos, cross_distance_cutoff,
                        data['surface'].batch, data['receptor'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['receptor'].node_sigma_emb[src.long()]
        edge_attr = edge_length_emb#torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
