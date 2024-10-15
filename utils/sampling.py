import numpy as np
import torch,os
from torch_geometric.loader import DataLoader
import traceback
from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
import warnings
# from datasets.process_mols import write_mol_with_coords
from force_optimize.minimize_utils import UpdateGrpah,GetfixedPDB,GetFFGenerator
from openmm.app import Modeller
from joblib import Parallel,delayed
from tqdm import tqdm
from loguru import logger
def randomize_position(data_list, no_torsion, no_random, tr_sigma_max,ligand_to_pocket_center = False):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate, torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())
        # put the molecule in the center of the pocket by caoduanhua
        if ligand_to_pocket_center and complex_graph['receptor'].pocket_center is not None:
            # logger.info('Use predict pocket center to put ligand in the center of pocket! {},{},{}'.format(complex_graph['ligand'].pos.shape,complex_graph['receptor'].pocket_center.shape,torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True).shape))
            complex_graph['ligand'].pos  = complex_graph['ligand'].pos - torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True) + complex_graph['receptor'].pocket_center.to(complex_graph['ligand'].pos.device)
            logger.info('Use predict pocket center to put ligand in the center of pocket!')
        else:
            if not no_random:  # note for now the torsion angles are still randomised
                tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
                complex_graph['ligand'].pos += tr_update

def inferenceFFOptimize(data_list,args,receptor_path,N=40):
    # loaded ligand docking pose and add Hs
    fixer = GetfixedPDB(receptor_path)
    modeller = Modeller(fixer.topology, fixer.positions)
    protein_atoms = list(fixer.topology.atoms())
    system_generator = GetFFGenerator()
    with Parallel(n_jobs=max(args.num_process,N)) as parallel:
        logger.info('Use force field to do energy minimized!')
        new_data_list = parallel(delayed(UpdateGrpah)(graph,system_generator,modeller,protein_atoms) for graph in data_list) # succssed return graph object ,error return int(1)
        
    result = np.array([i if type(i) == int else 0 for i in new_data_list])

    new_data_list = list(filter(lambda x:type(x)!=int,new_data_list))
    
    if result.sum() == 0:
        return new_data_list,[]
    else:
        indices = np.where(result == 1)
        failed_graphs = [data_list[i] for i in indices[0]]
        logger.info(f'Minimized not Completed:{receptor_path}, {len(failed_graphs)} sdf not be minimized by default forcefield , try use gaff-2.11 forcefield!')
        with Parallel(n_jobs=max(args.num_process,len(failed_graphs))) as parallel:
            new_data_list_add =  parallel(delayed(UpdateGrpah)(graph,system_generator,modeller,protein_atoms) for graph in failed_graphs)
            
        result = np.array([i if type(i) == int else 0 for i in new_data_list_add ])
        new_data_list_add = list(filter(lambda x:type(x)!=int,new_data_list_add))
        new_data_list = new_data_list + new_data_list_add
        if result.sum() != 0:
            indices = np.where(result == 1)
            failed_graphs = [failed_graphs[i] for i in indices[0]]
            logger.info(f'Minimized not Completed:{receptor_path}, {len(failed_graphs)} sdf not be minimized by default forcefield , try use gaff-2.11 forcefield!')
        else:
            failed_graphs = []
        return new_data_list,failed_graphs
@logger.catch
def sampling(input_data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False,args = None):
    data_list = input_data_list
    N = len(data_list)
    pred_score = []
    for t_idx in range(inference_steps):
        # use prediction score as a ranking metric , implemented by caoduanhua
        # pred_score = []
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)
            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch) 
                
                #
                
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            else:
                tor_perturb = None

            # Apply noise
            tor_count_head = 0
            tor_count_tail = 0
            # node_head = 0
            # node_tail = 0
            for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list()):
                # node_tail += complex_graph['ligand'].pos.shape[0]
                # complex_graph['ligand']['final_ligand'] = lig_node_attr[node_head:node_tail]
                # node_head+=complex_graph['ligand'].pos.shape[0]
                # if i==0:
                if type(complex_graph['ligand'].mask_rotate) is list:
                    complex_graph['ligand'].mask_rotate = complex_graph['ligand'].mask_rotate[0]
                tor_count_tail += complex_graph['ligand'].mask_rotate.shape[0]
                try:
                    new_data_list.append(modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                            tor_perturb[tor_count_head :tor_count_tail] if not model_args.no_torsion else None))
                except:
                    new_data_list.append(complex_graph)
                tor_count_head += complex_graph['ligand'].mask_rotate.shape[0]
                
                
        data_list = new_data_list
        

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)
    #Before scoring the final conformers, we need to use force field to do energy minimized
    if args is not None and args.force_optimize:
        
        """
        step 1 : load mol to make energy minimize
        step 2 : load fixed protein or pocket
        """
        try:
            receptor_path = data_list[0]["protein_path"]
            logger.info('recptor path: {}',receptor_path)
            new_data_list,failed_graphs = inferenceFFOptimize(data_list,args,receptor_path,N)
            if len(failed_graphs) != 0:
                receptor_path = data_list[0]["pocket_path"]
                logger.info('Some minimized failed ! Use pocket file to do energy minimized!')
                new_data_list_pocket,failed_graphs = inferenceFFOptimize(failed_graphs,args,receptor_path,len(failed_graphs))
                new_data_list += new_data_list_pocket
            logger.info('Return sucessed examples: {}',len(new_data_list))
            logger.info('Return failed examples: {}',len(failed_graphs))
            data_list = new_data_list + failed_graphs
            
        except Exception as e:
            error_info = traceback.format_exc()
            logger.info(error_info)

            warnings.warn(f'Complex {data_list[0]["name"]} will scoring without energy minimized!')
            pass
        


    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            # try use forcefields to do energy minimized!
            if confidence_data_list is not None:
                confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    # confidence need all_atoms or not
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch)[-1])
                else:
                    b = complex_graph_batch.num_graphs
                    set_time(complex_graph_batch, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(complex_graph_batch)[-1])
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
    if confidence is not None:
        pred_score = confidence

    return data_list, pred_score
