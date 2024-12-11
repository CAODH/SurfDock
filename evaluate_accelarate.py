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
import wandb
from biopandas.pdb import PandasPdb
from rdkit import RDLogger
from rdkit.Chem import RemoveHs,AllChem
from datasets.process_mols import write_mol_with_coords, generate_conformer
from torch_geometric.loader import DataLoader
from datasets.pdbbind import PDBBind, read_mol
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, get_symmetry_rmsd, remove_all_hs, read_strings_from_txt, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm
from loguru import logger
torch.multiprocessing.set_sharing_strategy('file_system')
RDLogger.DisableLog('rdApp.*')
import yaml
cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--model_version', type=str, default='version3', help='version of mdn model')
# save docking result or not
parser.add_argument('--save_docking_result', action='store_true', default=False, help='Whether to save docking result')
# put ligand to pocket center
parser.add_argument('--ligand_to_pocket_center', action='store_true', default=False, help='Whether to put ligand on pocket center')
# use_noise_to_rank
parser.add_argument('--use_noise_to_rank', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test_ns_48_nv_10_layer_62023-06-25_07-54-08_model', help='')
parser.add_argument('--project', type=str, default='ligbind_inf_test_mdn', help='')
parser.add_argument('--surface_path', type=str, default='~/PDBBind_processed_8A_surface/', help='test dataset surface path')
parser.add_argument('--esm_embeddings_path', type=str, default='~/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/esm_embedding/esm_embedding_pocket_for_train/esm2_3billion_embeddings.pt', help='test dataset esmbedding path')
parser.add_argument('--out_dir', type=str, default='~/diffScreen/test_workdir/mdn_result_40', help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')
parser.add_argument('--cache_path', type=str, default='~/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/cache_PDBBIND_pocket_8A', help='Folder from where to load/restore cached dataset')
parser.add_argument('--data_dir', type=str, default='~/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/PDBBind_pocket_8A/', help='Folder containing original structures')
parser.add_argument('--split_path', type=str, default='~/DeepLearningForDock/DiffDockForScreen/diffScreen/data/splits/timesplit_test', help='Path of file defining the split')
parser.add_argument('--no_overlap_names_path', type=str, default='~/DeepLearningForDock/DiffDockForScreen/diffScreen/data/splits/timesplit_test_no_rec_overlap', help='Path text file with the folder names in the test set that have no receptor overlap with the train set')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--wandb', action='store_true', default=True, help='')
parser.add_argument('--wandb_dir', type=str, default='~/diffScreen/test_workdir', help='Folder in which to save wandb logs')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--multi_seed_conformer', action='store_true', default=False, help='Whether to use multi_seed_conformer in inference steps')
parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Whether to save visualizations')
parser.add_argument('--samples_per_complex', type=int, default=40, help='Number of poses to sample for each complex')
parser.add_argument('--actual_steps', type=int, default=None, help='')
parser.add_argument('--mdn_dist_threshold_test', type=float, default=None, help='mdn_dist_threshold_test')
# force_minimized param
parser.add_argument('--force_optimize', action='store_true', default=False, help='')
args = parser.parse_args()
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

    if args.force_optimize:
        logger.info('Using ForceField for energy minimized!')
    test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                        receptor_radius=score_model_args.receptor_radius,
                        cache_path=args.cache_path, split_path=args.split_path,
                        remove_hs=score_model_args.remove_hs, max_lig_size=None,
                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                        matching=not score_model_args.no_torsion, keep_original=True,
                        popsize=score_model_args.matching_popsize,
                        maxiter=score_model_args.matching_maxiter,
                        all_atoms=score_model_args.all_atoms,
                        atom_radius=score_model_args.atom_radius,
                        atom_max_neighbors=score_model_args.atom_max_neighbors,
                        esm_embeddings_path=args.esm_embeddings_path,
                        require_ligand=True,
                        num_workers=args.num_workers,surface_path = args.surface_path)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
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
    logger.info('t schedule', tr_schedule)

    rmsds_list, obrmsds, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list, names_list = [], [], [], 0, 0, [], [], [], []
    run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
    N = args.samples_per_complex
    names_no_rec_overlap = read_strings_from_txt(args.no_overlap_names_path)
    names_test_all = read_strings_from_txt(args.split_path)
    name_map_idx = {name: idx for idx, name in enumerate(names_test_all)}
    idx_map_name = {idx: name for idx, name in enumerate(names_test_all)}

    logger.info('Size of test dataset: ', len(test_dataset))

    model = accelerator.prepare(model)
    test_loader= accelerator.prepare(test_loader)
    confidence_model = accelerator.prepare(confidence_model)

    for idx, orig_complex_graph in tqdm(enumerate(test_loader),total = len(test_loader),disable= not accelerator.is_local_main_process):
        if confidence_model is not None and not (confidence_args.use_original_model_cache or
                                                confidence_args.transfer_weights) and orig_complex_graph.name[0] not in confidence_complex_dict.keys():
            skipped += 1
            logger.info(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
            continue
        success = 0
        sample_count_failed = 0
        while not success: # keep trying in case of failure (sometimes stochastic)
            try:
                success = 1
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                
                if args.multi_seed_conformer:
                    # random multi seed conformers
                    if test_dataset.require_ligand:
                        for data in data_list:
                            mol_rdkit = copy.deepcopy(data.mol[0])
                            mol_rdkit.RemoveAllConformers()
                            mol_rdkit = AllChem.AddHs(mol_rdkit)
                            generate_conformer(mol_rdkit)
                            mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
                            data.mol = [mol_rdkit]
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
                        pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        # logger.info(orig_complex_graph['ligand'].pos.shape,orig_complex_graph.original_center.shape)
                        # logger.info(graph['ligand'].pos.device,graph.original_center.device)
                        # random rdkit matching
                        pdb.add((graph['ligand'].pos + (graph.original_center).detach().cpu()), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                rec_path = os.path.join(args.data_dir, data_list[0]["name"][0], f'{data_list[0]["name"][0]}_pocket.pdb')
                if not os.path.exists(rec_path):
                    rec_path = os.path.join(args.data_dir, data_list[0]["name"][0], f'{data_list[0]["name"][0]}_protein_obabel_reduce.pdb')
                rec = PandasPdb().read_pdb(rec_path)
                rec_df = rec.df['ATOM']
                receptor_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(
                    np.float32) - orig_complex_graph.original_center.cpu().numpy()
                receptor_pos = np.tile(receptor_pos, (N, 1, 1))
                start_time = time.time()
                if not args.no_model:
                    if confidence_model is not None and not (
                            confidence_args.use_original_model_cache or confidence_args.transfer_weights):
                        confidence_data_list = [copy.deepcopy(confidence_complex_dict[orig_complex_graph.name[0]]) for _ in
                                            range(N)]
                    else:
                        confidence_data_list = None

                    data_list, confidence = sampling(data_list=data_list, model=model,
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

                    confidence = confidence.cpu().detach().numpy()

                run_times.append(time.time() - start_time)
                if score_model_args.no_torsion: orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

                filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

                ligand_pos = np.asarray(
                    [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])
                orig_ligand_pos = np.expand_dims(
                    orig_complex_graph['ligand'].orig_pos[filterHs],
                    axis=0) # - orig_complex_graph.original_center.cpu().numpy() ,since get_idx have done this function!

                try:
                    mol = remove_all_hs(orig_complex_graph.mol[0])
                    rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[0], [l for l in ligand_pos])
                except Exception as e:
                    logger.info("Using non corrected RMSD because of the error", e)
                    rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
                rmsds_list.append(rmsd)
                centroid_distance = np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos.mean(axis=1), axis=1)
                # if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                #     confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = np.array(confidence)#.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    # logger.info(confidence, re_order,rmsd)
                    logger.info(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1)[re_order], ' centroid distance',
                        np.around(centroid_distance, 1)[re_order], ' confidences ', np.around(confidence, 4)[re_order])
                    confidences_list.append(confidence)

                else:
                    logger.info(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1), ' centroid distance',
                        np.around(centroid_distance, 1))
                    
                """ add a save command by caoduanhua to save the last state of ligand"""
                ########################################################################
                if args.save_docking_result:
                    ligand_pos_add_center = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
                    lig = orig_complex_graph.mol[0]
                    # save predictions
                    
                    write_dir = f'{args.out_dir}/docking_result_{os.path.basename(args.split_path)}/{orig_complex_graph.name[0]}'
                    os.makedirs(write_dir, exist_ok=True)
                    for  pos,rmsd_i,score in zip(ligand_pos_add_center,rmsd,confidence):
                        mol_pred = copy.deepcopy(lig)
                        if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
                        # if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                        write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'{orig_complex_graph.name[0]}_rmsd_{rmsd_i}_confidence_{score}.sdf'))
                ########################################################################

                centroid_distances_list.append(centroid_distance)

                cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
                self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

                base_cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - orig_ligand_pos[:, None, :, :], axis=-1)
                base_min_cross_distances_list.append(np.min(base_cross_distances, axis=(1, 2)))

                if args.save_visualisation:
                    write_dir_vis = f'{args.out_dir}/docking_result_{os.path.basename(args.split_path)}/{orig_complex_graph.name[0]}'
                    os.makedirs(write_dir, exist_ok=True)
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            try:
                                visualization_list[batch_idx].write(
                                    f'{write_dir_vis}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}.pdb')
                            except:
                                continue
                    else:
                        for rank, batch_idx in enumerate(np.argsort(rmsd)):
                            try:
                                visualization_list[batch_idx].write(
                                    f'{write_dir_vis}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
                            except:
                                continue
                without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
                names_list.append(name_map_idx[orig_complex_graph.name[0]])
            except Exception as e:
                logger.info("Failed on", orig_complex_graph["name"], e)
                failures += 1
                sample_count_failed +=1
                if sample_count_failed > 5:
                    logger.info(" Skip by five times Failed on", orig_complex_graph["name"], e)
                    success = 1
                else:
                    success = 0
                
    accelerator.wait_for_everyone()
    rmsds_list, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list =\
         accelerator.gather(torch.tensor(rmsds_list).to(device)), accelerator.gather(torch.tensor(centroid_distances_list).to(device)),accelerator.gather(torch.tensor(failures).to(device)),accelerator.gather(torch.tensor(skipped).to(device)),\
         accelerator.gather(torch.tensor(min_cross_distances_list).to(device)), accelerator.gather(torch.tensor(base_min_cross_distances_list).to(device)), accelerator.gather(torch.tensor(confidences_list).to(device))
    run_times, min_self_distances_list, without_rec_overlap_list = accelerator.gather(torch.tensor(run_times).to(device)), accelerator.gather(torch.tensor(min_self_distances_list).to(device)),accelerator.gather(torch.tensor(without_rec_overlap_list).to(device))
    rmsds_list, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list = \
    rmsds_list.cpu().detach().numpy(), centroid_distances_list.cpu().detach().numpy(), failures.cpu().detach().numpy(), skipped.cpu().detach().numpy(), min_cross_distances_list.cpu().detach().numpy(), base_min_cross_distances_list.cpu().detach().numpy(), confidences_list.cpu().detach().numpy()
    run_times, min_self_distances_list, without_rec_overlap_list = \
        run_times.cpu().detach().numpy(), min_self_distances_list.cpu().detach().numpy(), without_rec_overlap_list.cpu().detach().numpy()
    names_list = accelerator.gather(torch.tensor(names_list).to(device))
    names_list = names_list.cpu().detach().numpy()
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:

        logger.info('Performance without hydrogens included in the loss')
        logger.info(failures, "failures due to exceptions")
        logger.info(skipped, ' skipped because complex was not in confidence dataset')
        performance_metrics = {}
        for overlap in ['', 'no_overlap_']:
            if 'no_overlap_' == overlap:
                without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
                if without_rec_overlap.sum() == 0: continue
                rmsds = np.array(rmsds_list)[without_rec_overlap]
                min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
                centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
                # if confidence_model is not None:
                confidences = np.array(confidences_list)[without_rec_overlap]
                # else:
                #     confidences = None
                min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
                base_min_cross_distances = np.array(base_min_cross_distances_list)[without_rec_overlap]
                names = np.array(names_list)[without_rec_overlap]
            else:
                rmsds = np.array(rmsds_list)
                min_self_distances = np.array(min_self_distances_list)
                centroid_distances = np.array(centroid_distances_list)
                # if confidence_model is not None:
                confidences = np.array(confidences_list)
                # else:
                #     confidences = None
                min_cross_distances = np.array(min_cross_distances_list)
                base_min_cross_distances = np.array(base_min_cross_distances_list)
                names = np.array(names_list)
            names = np.array([idx_map_name[idx] for idx in names])

            run_times = np.array(run_times)
            np.save(f'{args.out_dir}/{overlap}min_cross_distances.npy', min_cross_distances)
            np.save(f'{args.out_dir}/{overlap}min_self_distances.npy', min_self_distances)
            np.save(f'{args.out_dir}/{overlap}base_min_cross_distances.npy', base_min_cross_distances)
            np.save(f'{args.out_dir}/{overlap}rmsds.npy', rmsds)
            np.save(f'{args.out_dir}/{overlap}centroid_distances.npy', centroid_distances)
            np.save(f'{args.out_dir}/{overlap}confidences.npy', confidences)
            np.save(f'{args.out_dir}/{overlap}run_times.npy', run_times)
            np.save(f'{args.out_dir}/{overlap}complex_names.npy', np.array(names))

            performance_metrics.update({
                f'{overlap}run_times_std': run_times.std().__round__(2),
                f'{overlap}run_times_mean': run_times.mean().__round__(2),
                f'{overlap}steric_clash_fraction': (
                            100 * (min_cross_distances < 0.4).sum() / len(min_cross_distances) / N).__round__(2),
                f'{overlap}self_intersect_fraction': (
                            100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / N).__round__(2),
                f'{overlap}mean_rmsd': rmsds.mean(),
                f'{overlap}rmsds_below_1': (100 * (rmsds < 1).sum() / len(rmsds) / N),
                f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
                f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N),
                f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
                f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
                f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),

                f'{overlap}mean_centroid': centroid_distances.mean().__round__(2),
                f'{overlap}centroid_below_2': (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
                f'{overlap}centroid_below_5': (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),
                f'{overlap}centroid_percentile_25': np.percentile(centroid_distances, 25).round(2),
                f'{overlap}centroid_percentile_50': np.percentile(centroid_distances, 50).round(2),
                f'{overlap}centroid_percentile_75': np.percentile(centroid_distances, 75).round(2),
            })

            if N >= 5:
                top5_rmsds = np.min(rmsds[:, :5], axis=1)
                top5_centroid_distances = centroid_distances[
                                            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
                top5_min_cross_distances = min_cross_distances[
                                            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
                top5_min_self_distances = min_self_distances[
                                            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top5_steric_clash_fraction': (
                                100 * (top5_min_cross_distances < 0.4).sum() / len(top5_min_cross_distances)).__round__(2),
                    f'{overlap}top5_self_intersect_fraction': (
                                100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),
                    f'{overlap}top5_rmsds_below_1': (100 * (top5_rmsds < 1).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
                    f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
                    f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

                    f'{overlap}top5_centroid_below_2': (
                                100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
                    f'{overlap}top5_centroid_below_5': (
                                100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
                    f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
                    f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
                    f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
                })

            if N >= 10:
                top10_rmsds = np.min(rmsds[:, :10], axis=1)
                top10_centroid_distances = centroid_distances[
                                            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
                top10_min_cross_distances = min_cross_distances[
                                                np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
                top10_min_self_distances = min_self_distances[
                                            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top10_steric_clash_fraction': (
                                100 * (top10_min_cross_distances < 0.4).sum() / len(top10_min_cross_distances)).__round__(2),
                    f'{overlap}top10_self_intersect_fraction': (
                                100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),
                    f'{overlap}top10_rmsds_below_1': (100 * (top10_rmsds < 1).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
                    f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
                    f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

                    f'{overlap}top10_centroid_below_2': (
                                100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
                    f'{overlap}top10_centroid_below_5': (
                                100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
                    f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
                    f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
                    f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
                })

            # if confidence_model is not None:
            if confidences is not None:
                confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]

                filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                filtered_min_cross_distances = min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:,
                                            0]
                filtered_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                performance_metrics.update({
                    f'{overlap}filtered_self_intersect_fraction': (
                                100 * (filtered_min_self_distances < 0.4).sum() / len(filtered_min_self_distances)).__round__(
                        2),
                    f'{overlap}filtered_steric_clash_fraction': (
                                100 * (filtered_min_cross_distances < 0.4).sum() / len(filtered_min_cross_distances)).__round__(
                        2),
                    f'{overlap}filtered_rmsds_below_1': (100 * (filtered_rmsds < 1).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_below_2': (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_below_5': (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_percentile_25': np.percentile(filtered_rmsds, 25).round(2),
                    f'{overlap}filtered_rmsds_percentile_50': np.percentile(filtered_rmsds, 50).round(2),
                    f'{overlap}filtered_rmsds_percentile_75': np.percentile(filtered_rmsds, 75).round(2),

                    f'{overlap}filtered_centroid_below_2': (
                                100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)).__round__(2),
                    f'{overlap}filtered_centroid_below_5': (
                                100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)).__round__(2),
                    f'{overlap}filtered_centroid_percentile_25': np.percentile(filtered_centroid_distances, 25).round(2),
                    f'{overlap}filtered_centroid_percentile_50': np.percentile(filtered_centroid_distances, 50).round(2),
                    f'{overlap}filtered_centroid_percentile_75': np.percentile(filtered_centroid_distances, 75).round(2),
                })

                if N >= 5:
                    top5_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)
                    top5_filtered_centroid_distances = \
                    centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    top5_filtered_min_cross_distances = \
                    min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    top5_filtered_min_self_distances = \
                    min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    performance_metrics.update({
                        f'{overlap}top5_filtered_self_intersect_fraction': (
                                    100 * (top5_filtered_min_cross_distances < 0.4).sum() / len(
                                top5_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top5_filtered_steric_clash_fraction': (
                                    100 * (top5_filtered_min_cross_distances < 0.4).sum() / len(
                                top5_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_1': (
                                    100 * (top5_filtered_rmsds < 1).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_2': (
                                    100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_5': (
                                    100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_percentile_25': np.percentile(top5_filtered_rmsds, 25).round(2),
                        f'{overlap}top5_filtered_rmsds_percentile_50': np.percentile(top5_filtered_rmsds, 50).round(2),
                        f'{overlap}top5_filtered_rmsds_percentile_75': np.percentile(top5_filtered_rmsds, 75).round(2),

                        f'{overlap}top5_filtered_centroid_below_2': (100 * (top5_filtered_centroid_distances < 2).sum() / len(
                            top5_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top5_filtered_centroid_below_5': (100 * (top5_filtered_centroid_distances < 5).sum() / len(
                            top5_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top5_filtered_centroid_percentile_25': np.percentile(top5_filtered_centroid_distances,
                                                                                        25).round(2),
                        f'{overlap}top5_filtered_centroid_percentile_50': np.percentile(top5_filtered_centroid_distances,
                                                                                        50).round(2),
                        f'{overlap}top5_filtered_centroid_percentile_75': np.percentile(top5_filtered_centroid_distances,
                                                                                        75).round(2),
                    })
                if N >= 10:
                    top10_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
                                                axis=1)
                    top10_filtered_centroid_distances = \
                    centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    top10_filtered_min_cross_distances = \
                    min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    top10_filtered_min_self_distances = \
                    min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    performance_metrics.update({
                        f'{overlap}top10_filtered_self_intersect_fraction': (
                                    100 * (top10_filtered_min_cross_distances < 0.4).sum() / len(
                                top10_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top10_filtered_steric_clash_fraction': (
                                    100 * (top10_filtered_min_cross_distances < 0.4).sum() / len(
                                top10_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_1': (
                                    100 * (top10_filtered_rmsds < 1).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_2': (
                                    100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_5': (
                                    100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_percentile_25': np.percentile(top10_filtered_rmsds, 25).round(2),
                        f'{overlap}top10_filtered_rmsds_percentile_50': np.percentile(top10_filtered_rmsds, 50).round(2),
                        f'{overlap}top10_filtered_rmsds_percentile_75': np.percentile(top10_filtered_rmsds, 75).round(2),

                        f'{overlap}top10_filtered_centroid_below_2': (100 * (top10_filtered_centroid_distances < 2).sum() / len(
                            top10_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top10_filtered_centroid_below_5': (100 * (top10_filtered_centroid_distances < 5).sum() / len(
                            top10_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top10_filtered_centroid_percentile_25': np.percentile(top10_filtered_centroid_distances,
                                                                                        25).round(2),
                        f'{overlap}top10_filtered_centroid_percentile_50': np.percentile(top10_filtered_centroid_distances,
                                                                                        50).round(2),
                        f'{overlap}top10_filtered_centroid_percentile_75': np.percentile(top10_filtered_centroid_distances,
                                                                                        75).round(2),
                    })

        for k in performance_metrics:
            logger.info(k, performance_metrics[k])

        if args.wandb:
            wandb.log(performance_metrics)
            histogram_metrics_list = [('rmsd', rmsds[:, 0]),
                                    ('centroid_distance', centroid_distances[:, 0]),
                                    ('mean_rmsd', rmsds.mean(axis=1)),
                                    ('mean_centroid_distance', centroid_distances.mean(axis=1))]
            if N >= 5:
                histogram_metrics_list.append(('top5_rmsds', top5_rmsds))
                histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))
            if N >= 10:
                histogram_metrics_list.append(('top10_rmsds', top10_rmsds))
                histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))
            # if confidence_model is not None:
            if confidences is not None:
                histogram_metrics_list.append(('filtered_rmsd', filtered_rmsds))
                histogram_metrics_list.append(('filtered_centroid_distance', filtered_centroid_distances))
                if N >= 5:
                    histogram_metrics_list.append(('top5_filtered_rmsds', top5_filtered_rmsds))
                    histogram_metrics_list.append(('top5_filtered_centroid_distances', top5_filtered_centroid_distances))
                if N >= 10:
                    histogram_metrics_list.append(('top10_filtered_rmsds', top10_filtered_rmsds))
                    histogram_metrics_list.append(('top10_filtered_centroid_distances', top10_filtered_centroid_distances))
        if args.wandb:
            wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    from accelerate.utils import set_seed
    device = accelerator.device
    set_seed(1024)
    accelerator.logger.info(f'device {str(accelerator.device)} is used!')
    main_function()
    # sys.exit()