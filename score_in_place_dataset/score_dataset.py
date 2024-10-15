import os
import copy
import torch
from rdkit.Chem import RemoveHs
import MDAnalysis as mda
from plyfile import PlyData
from torch_geometric.data import Data
from utils.torsion import get_transformation_mask
from torch_geometric.transforms import FaceToEdge, Cartesian
from torch_geometric.data import Dataset, HeteroData
from joblib import Parallel,delayed
from tqdm import tqdm
import re,glob
import multiprocessing
from rdkit.Chem import  AddHs
from loguru import logger
from datasets.process_mols import read_molecule, get_rec_graph, \
    get_lig_graph_with_matching, extract_receptor_structure,  parse_pdb_from_path,get_lig_graph,initConformer
class ScreenDataset(Dataset):
    def __init__(self, pocket_path,ligands_path,ref_ligand, surface_path=None,pocket_center=None,transform=None, cache_path='data/cache', split_path='data/',
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=False, keep_original=True, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,keep_input_pose = True,save_dir = None,inference_mode='Screen',ligandsMaxAtoms=80):
        super(ScreenDataset, self).__init__()
        # use False for docking & True for rescoring
        self.keep_input_pose = keep_input_pose
        self.ref_ligand = ref_ligand
        self.pocket_path = pocket_path
        self.ligands_path = ligands_path
        self.surface_path = surface_path
        self.pocket_center = pocket_center
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings = esm_embeddings
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original =  matching,keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.save_dir = save_dir
        self.inference_mode = inference_mode
        self.ligandsMaxAtoms= ligandsMaxAtoms
        # looking af the reusult dir to find the saved conformers and skip the ligand
        if not self.keep_input_pose:
            finished_samples = glob.glob(f'{self.save_dir}/*')
            if len(finished_samples) == 0:
                self.finished_idx = 0
            else:
                max_inner_idx = [int(re.findall(r'file_inner_idx_(\d+)', sample)[0]) for sample in finished_samples]
                self.finished_idx = max(max_inner_idx)+1
        
        self.preprocessing()

    def len(self):
        return len(self.ligs)
    def get(self, idx):
        """
        Get protein-ligand complex graph
        
        """
        lig = copy.deepcopy(self.ligs[idx])
        lig_path_name = self.ligand_names[idx]
        complex_graph = copy.deepcopy(self.protein_graph)
        if not self.keep_input_pose:
            try:
                complex_graph['name'] += '_' + lig.GetProp('_Name') + '_' + lig_path_name + f'_file_inner_idx_{idx + self.finished_idx}'
            except:
                complex_graph['name'] += '_' + lig_path_name + f'_file_inner_idx_{idx}'
        else:
            complex_graph['name'] += '_' + lig_path_name
        complex_graph.rmsd_matching = 0
        # if want to use SurfDock to generate conformers, use the following code
        """this module will work when you want to use SurfDock to generate conformers"""
        lig_complex_graph = HeteroData()
        get_lig_graph(lig, lig_complex_graph)

        edge_mask, mask_rotate = get_transformation_mask(lig_complex_graph)
        complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
        complex_graph['ligand'].mask_rotate = mask_rotate
        complex_graph['ligand'].x = lig_complex_graph['ligand'].x
        complex_graph['ligand'].pos = lig_complex_graph['ligand'].pos
        complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = lig_complex_graph['ligand', 'lig_bond', 'ligand'].edge_index
        complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = lig_complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr

        if not self.keep_input_pose:
            complex_graph['ligand'].pos  = complex_graph['ligand'].pos - torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True) + complex_graph.original_center


        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= complex_graph.original_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= complex_graph.original_center
        
        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph.original_ligand_center = ligand_center + complex_graph.original_center
        # if self.pocket_center is not None:
        #     complex_graph['receptor'].pocket_center -= complex_graph.original_center.numpy()
        complex_graph['receptor'].center_pos -= complex_graph.original_center.numpy()
        complex_graph['receptor'].atoms_pos -= complex_graph.original_center.numpy()
        if not self.keep_input_pose:
            complex_graph['mol'] = lig
        return complex_graph
    def preprocessing(self):
        self.ref_ligand,_ = read_mol(self.ref_ligand,remove_hs=False)
        self.protein_name = self.pocket_path.split('/')[-1].split('_')[0]
        self.protein_graph, self.ligs =self.get_complex(self.protein_name, self.esm_embeddings)

    def get_complex(self, name, lm_embedding_chains):
        try:
            rec_model = parse_pdb_from_path(self.pocket_path)
            # rec_model = parse_receptor(name, self.pdbbind_dir)
            pure_pocket_path = os.path.splitext(self.pocket_path)[0] + '_pure.pdb'
            # pure_pocket_path = os.path.join(self.pdbbind_dir, name, f'{name}_pocket_pure.pdb')
        except Exception as e:
            logger.info(f'Skipping {name} because of the error:{e}')
            logger.info(e)
            return [],[]
        ligs=[]
        ligand_names = []
        if os.path.isdir(self.ligands_path):
            ligands_paths = [ path for path in os.listdir(self.ligands_path) if path.endswith('.sdf') ]
            num_cores = multiprocessing.cpu_count()
            ligs_paths = Parallel(n_jobs = min(len(ligands_paths),num_cores-10),backend = 'threading')(delayed(read_mol)(os.path.join(self.ligands_path,ligand_path), remove_hs=self.remove_hs) for ligand_path in tqdm(ligands_paths,total = len(ligands_paths)))
            for lig,ligand_path in ligs_paths:
                if lig is not None:
                    if self.inference_mode=='Screen':
                        if len(lig.GetAtoms()) <= self.ligandsMaxAtoms:
                            ligs.append(lig)
                            ligand_names.append(ligand_path)
                    else:
                        ligs.append(lig)
                        ligand_names.append(ligand_path)
            if not self.keep_input_pose:
                num_cores = multiprocessing.cpu_count()
                if len(ligs) > 0:
                    ligs = Parallel(n_jobs = min(max(1,len(ligs)),max(1,num_cores-10)),backend = 'threading')(delayed(initConformer)(lig_mol,self.inference_mode) for lig_mol in tqdm(ligs,total = len(ligs)))
                    ligs = [mol for mol in ligs if mol is not None and mol.GetNumConformers() > 0][self.finished_idx:]
                    ligand_names = [ligand_name for ligand_name,mol in zip(ligand_names,ligs) if mol is not None and mol.GetNumConformers() > 0][self.finished_idx:]

            self.ligand_names = ligand_names
            
        else:

            # filter large mols
            if self.inference_mode=='Screen':
                ligs = read_mols(self.ligands_path, remove_hs=self.remove_hs) 
                ligs = [mol for mol in ligs if len(mol.GetAtoms()) <= self.ligandsMaxAtoms]
            else:
                ligs,_ = read_mol(self.ligands_path, remove_hs=self.remove_hs)
                ligs = [ligs] if ligs is not None else []
                logger.info(f'ligs: {self.ligands_path}')
            if not self.keep_input_pose:
                num_cores = multiprocessing.cpu_count()
                if len(ligs) > 0:
                    ligs = Parallel(n_jobs = min(max(1,len(ligs)),max(1,num_cores-10)),backend = 'threading')(delayed(initConformer)(lig_mol,self.inference_mode) for lig_mol in tqdm(ligs,total = len(ligs)))
                    ligs = [mol for mol in ligs if mol is not None and mol.GetNumConformers() > 0][self.finished_idx:]
                
            self.ligand_names = [os.path.basename(self.ligands_path)]*len(ligs)
        complex_graph = HeteroData()
        complex_graph['name'] = name
        logger.info(f'Processing {name}')
        try:
            if self.ref_ligand is None:

                logger.warning('No reference ligand was provided. Using the first ligand in the list as reference.')
                self.ref_ligand = ligs[0]
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), self.ref_ligand,save_file=pure_pocket_path, lm_embedding_chains=lm_embedding_chains)
            if lm_embeddings is not None and c_alpha_coords is not None and len(c_alpha_coords) != len(lm_embeddings):
                logger.info(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
            mda_rec_model = mda.Universe(pure_pocket_path)
            get_rec_graph(mda_rec_model, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius=self.receptor_radius,
                            c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                            atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
        except Exception as e:
            logger.info(f'Skipping {name} because of the error:{e}')

            return [],[]

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        complex_graph.original_center = protein_center

        # add surface data
        if self.surface_path is not None:
                
            with open(self.surface_path, 'rb') as f:
                data = PlyData.read(f)
            features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if axis.name not in ['nx', 'ny', 'nz'] ])
            pos = torch.stack(features[:3], dim=-1)
            # pos - center_protein_pos
            pos -= complex_graph.original_center
            features = torch.stack(features[3:], dim=-1)
            face = None
            if 'face' in data:
                faces = data['face']['vertex_indices']
                faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
                face = torch.stack(faces, dim=-1)
            data = Data(x=features, pos=pos, face=face)
            data = FaceToEdge()(data)
            data = Cartesian(cat=False)(data)
            complex_graph['surface'].pos = data.pos
            complex_graph['surface'].x = data.x
            complex_graph['surface','surface_edge','surface'].edge_index = data.edge_index
            complex_graph['surface','surface_edge','surface'].edge_attr = data.edge_attr

            # add pocket center for later use
            if self.pocket_center is not None:
                complex_graph['receptor'].pocket_center = torch.tensor(self.pocket_center).float() - complex_graph.original_center
            else:
                complex_graph['receptor'].pocket_center = None

        ####################################################################################################

        return complex_graph, ligs

def read_mol(ref_path, remove_hs=False):
    lig = read_molecule(ref_path, remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        logger.info('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
        try:
            lig = read_molecule(os.path.splitext(ref_path)[0] + '.mol2', remove_hs=remove_hs, sanitize=True)
        except:
            lig= None

    return lig,ref_path
from rdkit.Chem import AllChem
import warnings
from rdkit import Chem
def read_mols(ligands_path, sanitize=True, calc_charges=False, remove_hs=False):
# def read_molecules(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    mols = Chem.SDMolSupplier(ligands_path, sanitize=False, removeHs=False)
    # mols = supplier
    filter_mols = []
    for mol in mols:
        try:
            if sanitize or calc_charges:
                Chem.SanitizeMol(mol)

            if calc_charges:
                # Compute Gasteiger charges on the molecule.
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except:
                    warnings.warn('Unable to compute charges for the molecule.')
            if remove_hs:
                mol = Chem.RemoveHs(mol, sanitize=sanitize)
            if mol is not None:
                filter_mols.append(mol)
        except Exception as e:
            logger.info(e)
            logger.info("RDKit was unable to read the molecule.")
            # return None
            continue

    return filter_mols