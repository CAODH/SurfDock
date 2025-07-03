"""
caoduanhua : we should to implemented a parapllel version of evaluate.py for a large dataset
"""

import copy
import os
import torch
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
import time
import numpy as np
import pandas as pd
import wandb
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from score_in_place_dataset.score_dataset import ScreenDataset
# from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from tqdm import tqdm
from loguru import logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
RDLogger.DisableLog('rdApp.*')
import yaml
cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()
# '~/diffScreen/workdir/mdn_model_ns_48_nv_10_layer_62023-07-04_04-16-12'
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--data_csv', type=str, default='~/DeepLearningForDock/dataset/DEKOIS2.csv', help='Path to folder with dataset for score in place')
# parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--esm_embeddings_path', type=str, default="~/esm2_3billion_pdbbind_embeddings.pt", help='Path to folder with esm embeddings for screen proteins')
# parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--model_version', type=str, default='version4', help='version of mdn model')
parser.add_argument('--mdn_dist_threshold_test', type=float, default=None, help='mdn_dist_threshold_test')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test_ns_48_nv_10_layer_62023-06-25_07-54-08_model', help='')
parser.add_argument('--project', type=str, default='ligbind_inf_test_mdn', help='')

parser.add_argument('--out_dir', type=str, default='~/test_workdir/mdn_result_40', help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')

parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--wandb_dir', type=str, default='~/test_workdir', help='Folder in which to save wandb logs')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')

args = parser.parse_args()


def main_function():
    """
    Main function for evaluating scores in place using a confidence model.
    This function performs the following tasks:
    1. Loads configuration settings from a YAML file if provided.
    2. Sets up output directories and initializes logging.
    3. Loads a pre-trained confidence model and its parameters.
    4. Reads input data from a CSV file and processes it using a dataset and dataloader.
    5. Evaluates the confidence model on the input data and generates predictions.
    6. Saves the results to a CSV file and optionally logs them to Weights & Biases (wandb).
    Result CSV Columns:
    - `sdf_name`: The name of the SDF file associated with the molecule.
    - `screen_confidence(for molecule rank)`: The confidence score for ranking molecules.
    - `confidence_name`: The name of the confidence prediction, including detailed metadata.
    - `pose_prediction_confidence(for pose rank)`: The confidence score for ranking poses, extracted from `confidence_name`.
    - `pose_sample_idx`: The sample index of the pose, extracted from `confidence_name`.
    - `pose_rank`: The rank of the pose, extracted from `confidence_name`.
    - `molecule_name`: The name of the molecule, extracted from `confidence_name`.
    - `molecule_idx_in_input_file`: The index of the molecule in the input file, extracted from `confidence_name`.
    Notes:
    - The function uses the `accelerator` library for distributed processing.
    - The `wandb` library is used for experiment tracking if enabled.
    - The confidence model is loaded and evaluated in a no-gradient mode (`torch.no_grad()`).
    - Errors during processing of individual ligands are logged and skipped.
    Outputs:
    - A CSV file containing the results is saved in the specified output directory.
    - A `Readme.txt` file is generated with metadata about the run.
    - Optionally, results are logged to Weights & Biases (wandb).
    Raises:
    - Exceptions during ligand processing are caught and logged without halting the execution.
    """
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
  
    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            args_dicts = yaml.full_load(f)
            if 'topN' not in args_dicts.keys():
                args_dicts['topN'] = 1
            confidence_args = Namespace(**args_dicts)
            # 
            confidence_args.transfer_weights = False
            confidence_args.use_original_model_cache = True
            confidence_args.original_model_dir = None
    # load model param & weight bias
    if args.confidence_model_dir is not None:
        if confidence_args.transfer_weights:
            with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                args_dicts = yaml.full_load(f)
                if 'topN' not in args_dicts.keys():
                    args_dicts['topN'] = 1
                confidence_model_args = Namespace(**args_dicts)
        else:
            confidence_model_args = confidence_args
            # confidence_model_args.add_argument('--topN', type=int, default=1, help='Number of atoms to calculate confidence')
        confidence_model_args.mdn_dist_threshold_test = args.mdn_dist_threshold_test if args.mdn_dist_threshold_test is not None else 5.0
        if not hasattr(confidence_model_args,'mdn_dist_threshold_train'):
                confidence_model_args.mdn_dist_threshold_train =7.0

        confidence_model = get_model(confidence_model_args, device, t_to_sigma=None, no_parallel=True,
                                    model_type = 'mdn_model')
        
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()

    confidence_model = accelerator.prepare(confidence_model)

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
    df = pd.read_csv(args.data_csv)
    pocket_paths = df['pocket_path'].tolist()
    ligands_paths = df['ligand_path'].tolist()
    ref_ligands = df['ref_ligand'].tolist()
    surface_paths = df['protein_surface'].tolist()
    esm_embeddings_dict = torch.load(args.esm_embeddings_path)
    confidence = []
    confidence_names = []
    sdf_names = []
    start_time = time.time()
    pbar = tqdm(zip(pocket_paths,ligands_paths,ref_ligands,surface_paths),total=len(pocket_paths))
    for pocket_path,ligands_path,ref_ligand,surface_path in pbar:
        try:
            esm_embeddings = esm_embeddings_dict[os.path.splitext(os.path.basename(pocket_path))[0]]
            # assert False, f'esmembedding shape : {esm_embeddings.shape}'
            test_dataset = ScreenDataset(pocket_path,ligands_path,ref_ligand,surface_path,transform=None,
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
                                num_workers=args.num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
            if len(test_dataset) == 0:
                continue
            test_loader= accelerator.prepare(test_loader)
            logger.info('Size of test dataset: ', len(test_dataset))
            with torch.no_grad():
                confidence_model.eval()
                for confidence_complex_graph_batch in tqdm(test_loader,total = len(test_loader)):
                    confidence += confidence_model(confidence_complex_graph_batch)[-1].cpu().detach().numpy().tolist()
                    confidence_names += confidence_complex_graph_batch['name']
                    sdf_names += [os.path.basename(ligands_path)]*len(confidence_complex_graph_batch['name'])
                    assert len(confidence)==len(confidence_names)==len(sdf_names)
                    # logger.info(len(confidence_complex_graph_batch['name'][0]),len(confidence_complex_graph_batch['name']),confidence_complex_graph_batch['name'][0])
        except Exception as e:
            logger.info(e,'some error failed for : ',ligands_path)
            continue
            # if accelerator.is_local_main_process:
        pbar.set_description('screen time used: {:.2f} '.format(time.time()-start_time))
    logger.info('screen time used: ',time.time()-start_time)
    if accelerator.is_local_main_process:
        result = pd.DataFrame({'sdf_name':sdf_names,'screen_confidence(for molecule rank)':confidence,'pose_file_path':confidence_names})
        csv_flag = os.path.basename(args.data_csv).split('.')[0]
        result['pose_prediction_confidence(for pose rank)'] = result['pose_file_path'].apply(lambda x: float(x.split('_')[-1].split('.sdf')[0]))
        result['pose_sample_idx'] = result['pose_file_path'].apply(lambda x: float(x.split('sample_idx_')[-1].split('_rank')[0]))
        result['pose_rank'] = result['pose_file_path'].apply(lambda x: float(x.split('_rank_')[-1].split('_confidence_')[0]))
        result['molecule_name'] = result['pose_file_path'].apply(lambda x: x.split('.sdf_file_inner_idx_')[0].split('/')[-1])
        result['molecule_idx_in_input_file'] = result['pose_file_path'].apply(lambda x: float(x.split('sdf_file_inner_idx_')[-1].split('_sample_idx_')[0]))
        result.to_csv(f'{args.out_dir}/{csv_flag}_confidence.csv',index=False)
        with open(f"{args.out_dir}/Readme.txt", 'w') as f:
            f.write(f"""Result CSV Columns:
        - `sdf_name`: The name of the SDF file associated with the molecule.
        - `screen_confidence(for molecule rank)`: The confidence score for ranking molecules to screen a library.
        - `pose_file_path`: The path of the SDF file associated with the molecule, including detailed metadata.
        - `pose_prediction_confidence(for pose rank)`: The confidence score for ranking poses, extracted from `pose_file_path`.
        - `pose_sample_idx`: The sample index of the pose, extracted from `pose_file_path`.
        - `pose_rank`: The rank of the pose, extracted from `pose_file_path`.
        - `molecule_name`: The name of the molecule, extracted from `pose_file_path`.
        - `molecule_idx_in_input_file`: The index of the molecule in the input file, extracted from `pose_file_path`.
        """)
            # np.save(f'{args.out_dir}/confidence.npy', confidence)
    if args.wandb:
            wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    from accelerate.utils import set_seed
    import sys
    device = accelerator.device
    set_seed(42)
    accelerator.print(f'device {str(accelerator.device)} is used!')
    main_function()
    sys.exit()
