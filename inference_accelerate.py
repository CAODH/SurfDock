"""
caoduanhua : we should to implemented a parapllel version of evaluate.py for a large dataset
"""

import copy
import os
import torch
import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import gc
import pandas as pd
import wandb
import glob
from rdkit import RDLogger
from rdkit.Chem import RemoveHs
from datasets.process_mols import write_mol_with_coords
from torch_geometric.loader import DataLoader
from datasets.pdbbind import PDBBind, read_mol,read_abs_file_mol
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling,inferenceFFOptimize
from utils.utils import get_symmetry_rmsd, remove_all_hs
from score_in_place_dataset.score_dataset import ScreenDataset
from utils.utils import get_model, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm
from collections import defaultdict
from packaging import version
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
RDLogger.DisableLog('rdApp.*')
import yaml
from loguru import logger
cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()

parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--data_csv', type=str, default='~/Screen_dataset/dataset/DEKOIS2.csv', help='Path to folder with dataset for score in place')
parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
# save docking result or not
parser.add_argument('--save_docking_result', action='store_true', default=False, help='Whether to save docking result')
# put ligand to pocket center
parser.add_argument('--ligand_to_pocket_center', action='store_true', default=False, help='Whether to put ligand on pocket center')
parser.add_argument('--keep_input_pose', action='store_false', default=False, help='Whether keep original input pose')
parser.add_argument('--use_noise_to_rank', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test_ns_48_nv_10_layer_62023-06-25_07-54-08_model', help='')
parser.add_argument('--project', type=str, default='ligbind_inf_test_mdn', help='')
parser.add_argument('--surface_path', type=str, default='~/PDBBind_processed_8A_surface/', help='test dataset surface path')
parser.add_argument('--esm_embeddings_path', type=str, default='~/PDBBIND/esm_embedding/esm_embedding_pocket_for_train/esm2_3billion_embeddings.pt', help='test dataset esmbedding path')
parser.add_argument('--out_dir', type=str, default='~/test_workdir/mdn_result_40', help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel we recommand set number = batch_size_molecule*samples_per_complex')
parser.add_argument('--batch_size_molecule', type=int, default=1, help='Number of molecul to sample in parallel')
parser.add_argument('--cache_path', type=str, default='~/PDBBIND/cache_PDBBIND_pocket_8A', help='Folder from where to load/restore cached dataset')
parser.add_argument('--data_dir', type=str, default='~/PDBBIND/PDBBind_pocket_8A/', help='Folder containing original structures')
parser.add_argument('--split_path', type=str, default='~/data/splits/timesplit_test', help='Path of file defining the split')
parser.add_argument('--no_overlap_names_path', type=str, default='~/data/splits/timesplit_test_no_rec_overlap', help='Path text file with the folder names in the test set that have no receptor overlap with the train set')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--wandb_dir', type=str, default='~/test_workdir', help='Folder in which to save wandb logs')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
parser.add_argument('--num_process', type=int, default=20, help='Number of parallel workers for minimized.')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Whether to save visualizations')
parser.add_argument('--samples_per_complex', type=int, default=40, help='Number of poses to sample for each complex')
parser.add_argument('--save_docking_result_number', type=int, default=1, help='Number of poses to save in disk for each complex')
parser.add_argument('--actual_steps', type=int, default=None, help='')
parser.add_argument('--inference_mode', default='Screen', help='inference mode',choices=['Screen','evaluate'])
parser.add_argument('--head_index', type=int, default=0, help='the head index to start inference,this optinal to inference use multi-GPU every GPU minimized a part of csv file ')
parser.add_argument('--tail_index', type=int, default=-1, help='the tail index to start inference,this optinal to inference use multi-GPU every GPU minimized a part of csv file')
parser.add_argument('--ligandsMaxAtoms', type=int, default=80, help='the max number of atoms in ligand')
parser.add_argument('--random_seed', type=int, default=42,  help='random seed')
# force_minimized param
parser.add_argument('--force_optimize', action='store_true', default=False, help='')
parser.add_argument('--mdn_dist_threshold_test', type=float, default=3.0, help='mdn_dist_threshold_test')
args = parser.parse_args()
nowtime = datetime.now().strftime('%Y-%m-%d')
log_file_flag = '-'.join(args.project.split('/'))
logger.add(f'./log-inference-{log_file_flag}-{nowtime}.log', rotation="500MB")
logger.info('Runing inference script in path: {}',os.getcwd())
logger.info('Runing inference with args: {}',args)

def main_function():
    if accelerator.is_local_main_process:
        if args.wandb:
            wandb.login(key = 'yourkey')
            run = wandb.init(
                entity='SurfDock',
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name,
                dir = args.wandb_dir,
                config=args
            )
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    if args.out_dir is None: args.out_dir = f'inference_out_dir_not_specified/{args.run_name}'
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
       

    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))
            # 
            confidence_args.transfer_weights = False
            confidence_args.use_original_model_cache = True
            confidence_args.original_model_dir = None
            confidence_args.mdn_dist_threshold_test = args.mdn_dist_threshold_test if args.mdn_dist_threshold_test is not None else 5.0
            if not hasattr(confidence_args,'mdn_dist_threshold_train'):
                confidence_args.mdn_dist_threshold_train =7.0

    if args.confidence_model_dir is not None:
        if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
            # if the confidence model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            logger.info('HAPPENING | confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.')
            confidence_test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                                    receptor_radius=confidence_args.receptor_radius,
                                cache_path=args.cache_path, split_path=args.split_path,
                                remove_hs=confidence_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                matching=not confidence_args.no_torsion, keep_original=True,
                                popsize=confidence_args.matching_popsize,
                                maxiter=confidence_args.matching_maxiter,
                                all_atoms=confidence_args.all_atoms,
                                atom_radius=confidence_args.atom_radius,
                                atom_max_neighbors=confidence_args.atom_max_neighbors,
                                esm_embeddings_path= args.esm_embeddings_path, require_ligand=True,
                                num_workers=args.num_workers,surface_path = args.surface_path)
            confidence_complex_dict = {d.name: d for d in confidence_test_dataset}

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    if not args.no_model:
        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,model_type = score_model_args.model_type)
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
        if args.ckpt == 'last_model.pt':
            model_state_dict = state_dict['model']
            ema_weights_state = state_dict['ema_weights']
            model.load_state_dict(model_state_dict, strict=True)
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
            ema_weights.load_state_dict(ema_weights_state, device=device)
            ema_weights.copy_to(model.parameters())
        else:
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            logger.info('loaded model weight for score model')
        if args.confidence_model_dir is not None:
            if confidence_args.transfer_weights:
                with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                    confidence_model_args = Namespace(**yaml.full_load(f))
            else:
                confidence_model_args = confidence_args

            confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                        model_type = confidence_model_args.model_type)
            state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None
            confidence_model_args = None


    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    logger.info('t schedule:{}',tr_schedule)
    logger.info('Loading data ...........')

    """
    Load data from csv file to get the path of pocket,ligand,ref_ligand,surface
    
    """
    df = pd.read_csv(args.data_csv)[args.head_index:args.tail_index]
    protein_paths = df['protein_path'].tolist()
    pocket_paths = df['pocket_path'].tolist()
    ligands_paths = df['ligand_path'].tolist()
    ref_ligands = df['ref_ligand'].tolist()
    surface_paths = df['protein_surface'].tolist()
    if 'pocket_center' in df.columns:
        pocket_centers = df['pocket_center'].tolist()
        new_pocket_centers = []
        for center in pocket_centers:
            x = center.split(',')[0]
            y = center.split(',')[1]
            z = center.split(',')[2]
            new_pocket_centers.append(np.array([(float(x),float(y),float(z))]))
        pocket_centers = new_pocket_centers
    else:
        pocket_centers = [None]*len(protein_paths)

    esm_embeddings_dict = torch.load(args.esm_embeddings_path)
    confidence_list = []
    confidence_names = []
    sdf_names = []
    pocket_path_list =[]

    failures = 0
    N = args.samples_per_complex
    all_molecules = 0
    pbar = tqdm(zip(pocket_paths,ligands_paths,ref_ligands,surface_paths,protein_paths,pocket_centers),total=len(pocket_paths))
    start_time = time.time()
    for pocket_path,ligands_path,ref_ligand,surface_path,protein_path,pocket_center in pbar:
        in_loop_start_time = time.time()
        
        try:
        
            dirname = os.path.splitext(pocket_path.split('/')[-1])[0] + '_'+ os.path.splitext(ligands_path.split('/')[-1])[0] 
            write_dir =  os.path.join(args.out_dir,'SurfDock_docking_result',dirname)#f'{args.out_dir}/SurfDock_docking_result/{dirname}'
            os.makedirs(write_dir, exist_ok=True)
            
            esm_embeddings = copy.deepcopy(esm_embeddings_dict[os.path.splitext(os.path.basename(pocket_path))[0]])

            test_dataset = ScreenDataset(pocket_path,ligands_path,ref_ligand,surface_path,pocket_center,transform=None,
                                receptor_radius=confidence_args.receptor_radius,
                                cache_path=None, split_path=None,
                                remove_hs=confidence_args.remove_hs, max_lig_size=None,
                                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                matching= False, keep_original=True,
                                popsize=confidence_args.matching_popsize,
                                maxiter=confidence_args.matching_maxiter,
                                all_atoms=confidence_args.all_atoms,
                                atom_radius=confidence_args.atom_radius,
                                atom_max_neighbors=confidence_args.atom_max_neighbors,
                                esm_embeddings=esm_embeddings,
                                require_ligand=False,
                                num_workers=args.num_workers,
                                keep_input_pose = args.keep_input_pose,
                                save_dir = write_dir,
                                inference_mode = args.inference_mode,
                                ligandsMaxAtoms=args.ligandsMaxAtoms)
            test_sample_num = len(test_dataset)
            all_molecules += test_sample_num
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_molecule, shuffle=False)
            if test_sample_num == 0:
                logger.error('No complexes need to be docking (skip before done or some errors) in {}', pocket_path)
                continue
            # test_loader= accelerator.prepare(test_loader)
            logger.info('Protein {} Size of test dataset: {}',os.path.splitext(os.path.basename(pocket_path))[0],  test_sample_num)
            ##### use torch.__version__orch complie to speed up the process ###
            # if version.parse(torch.__version__.split('+')[0])> version.parse("2.0"):
            #     model = torch.compile(model)
            #     confidence_model = torch.compile(confidence_model)
            #     logger.info('Your are using torch version={} , so SurfDock will use torch.compile to complie  model and confidence model',torch.__version__)
            #########################################################
            model = accelerator.prepare(model)
            test_loader= accelerator.prepare(test_loader)
            confidence_model = accelerator.prepare(confidence_model)
            """
            Start sampling conformers by SurfDock
            """
            for idx, orig_complex_graph in tqdm(enumerate(test_loader),total = len(test_loader),disable= not accelerator.is_local_main_process):
                
                try:
                    if 'ligand' not in orig_complex_graph.node_types:
                        logger.error('some error failed for conformer generate in rdkit: idx in batch graph: {}, ligand_path: {}',idx,ligands_path)
                        continue
                    orig_complex_graph_list = orig_complex_graph.to_data_list()
                    # add protein pocket information for minimized stage 
                    for temp_graph in orig_complex_graph_list:
                        temp_graph['protein_path'] = protein_path
                        temp_graph['pocket_path'] = pocket_path

                    success = 0
                    sample_count_failed = 0
                    data_list = []
                    # object 
                    data_list = [copy.deepcopy(temp_graph) for temp_graph in orig_complex_graph_list for _ in range(N)]
                    while not success: # keep trying in case of failure (sometimes stochastic)
                        
                        try:
                           
                            # data_list = [copy.deepcopy(temp_graph) for temp_graph in orig_complex_graph_list for _ in range(N)]
                            success = 1
                            randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max,ligand_to_pocket_center = args.ligand_to_pocket_center)
                            pdb = None
                            if args.save_visualisation:
                                visualization_list = []
                                for idx, graph in enumerate(data_list):
                                    # raw pose
                                    lig = read_mol(args.data_dir, graph['name'][0], remove_hs=score_model_args.remove_hs)
                                    pdb = PDBFile(lig)
                                    pdb.add(lig, 0, 0)
                                    # pose rdkit matching
                                    orig_complex_count = idx//N
                                         
                                    pdb.add((orig_complex_graph_list[orig_complex_count]['ligand'].pos + orig_complex_graph_list[orig_complex_count].original_center).detach().cpu(), 1, 0)
                                    # random rdkit matching
                                    pdb.add((graph['ligand'].pos + (graph.original_center).detach().cpu()), part=1, order=1)
                                    visualization_list.append(pdb)
                            else:
                                visualization_list = None

                            if not args.no_model:

                                confidence_data_list = None

                                data_list, confidence = sampling(input_data_list=data_list, model=model,
                                                                inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                                tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                                tor_schedule=tor_schedule,
                                                                device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                                no_random=args.no_random,
                                                                ode=args.ode, visualization_list=visualization_list,
                                                                confidence_model=confidence_model,
                                                                confidence_data_list=confidence_data_list,
                                                                confidence_model_args=confidence_model_args,
                                                                batch_size=args.batch_size,
                                                                no_final_step_noise=args.no_final_step_noise,args = args)
                                accelerator.wait_for_everyone()

                                confidence = confidence.cpu().detach().numpy()

                                # save confidence
                                confidence_list += confidence.tolist()
                                for _ in range(len(orig_complex_graph_list)):
                                    
                                    confidence_names.extend([orig_complex_graph_list[_]['name']]*N)
                                    pocket_path_list.extend([os.path.basename(pocket_path)]*N)
                                
                                sdf_names += [os.path.basename(ligands_path)]*len(confidence)

                                assert len(confidence_list)==len(confidence_names)==len(sdf_names)==len(pocket_path_list)
                            """ add a save command by caoduanhua to save the last state of ligand """
                            ########################################################################
                            if args.save_docking_result:
                                """"if you use multiple molecule parallel inference, you should re_order the confidence one by one"""
                                # add a parm to control the number of save ligand pose
                                head_threshold = 0
                                tail_threshold = N
                                confidence_tmp = confidence[head_threshold:tail_threshold]
                                re_order = np.argsort(confidence_tmp)[::-1]
                                if args.inference_mode=='evaluate':
                                    true_mol = remove_all_hs(read_abs_file_mol(ref_ligand))
                                for _ in range(len(orig_complex_graph_list)):
                                    for rank, batch_idx in enumerate(re_order[:args.save_docking_result_number]):
                                        true_idx = head_threshold + batch_idx
                                        mol_pred = copy.deepcopy(data_list[true_idx]['mol'])
                                        
                                        pos = data_list[true_idx]['ligand'].pos.cpu().numpy() + orig_complex_graph_list[_].original_center.cpu().numpy()
 
                                        if score_model_args.remove_hs: mol_pred = remove_all_hs(mol_pred)
                                        
                                        
                                        if args.inference_mode=='evaluate':
                                            try:
                                                rmsd = get_symmetry_rmsd(true_mol, true_mol.GetConformers()[0].GetPositions(), [pos])[0]
                                            except Exception as e:
                                                logger.warning("Using non corrected RMSD because of the error:{}", e)

                                                rmsd = np.sqrt(((true_mol.GetConformers()[0].GetPositions() - pos) ** 2).sum(axis=-1).mean(axis=0))
                                            result_filename = f'{data_list[true_idx]["name"]}_sample_idx_{batch_idx}_rank_{rank + 1}_rmsd_{rmsd}_confidence_{confidence_tmp[batch_idx]}.sdf'
                                        else:
                                            result_filename = f'{data_list[true_idx]["name"]}_sample_idx_{batch_idx}_rank_{rank + 1}_confidence_{confidence_tmp[batch_idx]}.sdf'
                                        
                                        write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, result_filename))
                                        
                                        if args.save_visualisation:
                                            write_dir_vis = f'{args.out_dir}/SurfDock_docking_result/{data_list[true_idx]["name"]}'
                                            os.makedirs(write_dir, exist_ok=True)
                                            if args.inference_mode=='evaluate':
                                                vis_filename =f'{data_list[true_idx]["name"]}_sample_idx_{batch_idx}_rank_{rank + 1}_rmsd_{rmsd}_confidence_{confidence_tmp[batch_idx]}.pdb'
                                            else:
                                                vis_filename = f'{data_list[true_idx]["name"]}_sample_idx_{batch_idx}_rank_{rank + 1}_confidence_{confidence_tmp[batch_idx]}.pdb'
                                            try:
                                                visualization_list[batch_idx].write(
                                                    f'{write_dir_vis}/{vis_filename}')
                                            except:
                                                continue
                                    head_threshold += N
                                    tail_threshold += N

                                    if _ < len(orig_complex_graph_list) - 1:
  
                                        confidence_tmp = confidence[head_threshold:tail_threshold]
                                        re_order = np.argsort(confidence_tmp)[::-1]
                        except Exception as e:
                            # if isinstance(e,RecursionError) or 'out of memory' in str(e):
                            data_list = None
                            referrers = gc.get_referrers(data_list)
                            for ref in referrers:
                                    ref=None
                            gc.collect()
                            torch.cuda.empty_cache()
                            data_list = [copy.deepcopy(temp_graph) for temp_graph in orig_complex_graph_list for _ in range(N)]
                            logger.error("Failed on :{}, error of :{}", orig_complex_graph["name"], e)
                            failures += 1
                            sample_count_failed +=1
                            if sample_count_failed > 5:
                                logger.error(" Skip by five times Failed on :{}, error of :{}", orig_complex_graph["name"], e)
                                success = 1
                            else:
                                success = 0

                except Exception as e:
                    if 'out of memory' in str(e):
                        logger.critical('| WARNING: ran out of memory, skipping batch')
                    orig_complex_graph_list,orig_complex_graph,data_list=None,None,None
                    referrers = gc.get_referrers(data_list)
                    for ref in referrers:
                        ref=None
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.error('Some error failed for sampling: idx in batch : {}, ligand_path: {},error of :{} ',idx,ligands_path,e)
                    
                    continue
            # if args.inference_mode=='evaluate':
            esm_embeddings,test_dataset,test_loader,orig_complex_graph_list,orig_complex_graph,data_list=None,None,None,None,None,None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error('Some error failed for graph data. ligand_path: {},error of :{}',ligands_path,e)
            esm_embeddings,test_dataset,test_loader,orig_complex_graph_list,orig_complex_graph,data_list=None,None,None,None,None,None
            referrers = gc.get_referrers(data_list)
            for ref in referrers:
                ref=None
            gc.collect()
            torch.cuda.empty_cache()
            continue
        logger.info('Protein {} used time: {}',os.path.splitext(os.path.basename(pocket_path))[0],time.time() - in_loop_start_time)
    accelerator.wait_for_everyone()
    docking_time = time.time() - start_time
    if accelerator.is_local_main_process:
        logger.info('Docking time used for one moleculer: {}',docking_time/ all_molecules)
        logger.info('Docking time used: {}', docking_time)
        logger.info('Sampling conformers number: {}',all_molecules*args.samples_per_complex)
        logger.info('Output conformers number: {}',  all_molecules*args.save_docking_result_number)
        logger.info('Docking output molecule number: {}',all_molecules)
        # logger.info('Docking time used for one moleculer: {}',docking_time/all_molecules)

    result = pd.DataFrame({'sdf_name':sdf_names,'confidence':confidence_list,'confidence_name':confidence_names,'pocket_path':pocket_path_list})
    csv_flag = os.path.basename(args.data_csv).split('.')[0]
    result.to_csv(f'{args.out_dir}/{csv_flag}_head_{str(args.head_index)}_tail_{str(args.tail_index)}_confidence_on_device_{device}.csv',index=False)

    if accelerator.is_local_main_process:
        if args.wandb:
            wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    from accelerate.utils import set_seed
    device = accelerator.device
    set_seed(args.random_seed)
    from functools import partial

    accelerator.print(f'device {str(accelerator.device)} is used!')
    main_function()
