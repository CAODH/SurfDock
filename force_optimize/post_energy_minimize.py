import os
from minimize_utils import GetfixedPDB,GetFFGenerator,UpdatePose,GetPlatformPara,GetPlatform,Molecule,trySystem,read_molecule,run_command,read_abs_file_mol
import sys
from openmm.app import Modeller
from joblib import Parallel,delayed
import argparse
from tqdm import tqdm
from glob import glob
import warnings
import traceback
import time
import pandas as pd
import numpy as np
import logging


""""
This Script will help user to do energy minimized for protein-ligand complex by openmm
Of course , you can use force_optimize args in docking step if you want to minimized all docking pose!,but may be it will be slowly
So , I think you can use this script to do energy minimized for protein-ligand complex that ranking topN in docking step,this will save more time,without performance loss
Enjoy it!

"""
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Process protein-ligand files.')
    parser.add_argument('--head_num', type=int, default=20, help='Number of top pose to be minimized.')
    parser.add_argument('--num_process', type=int, default=20, help='Number of parallel workers.')
    parser.add_argument('--cuda', type=int, default=0, help='Number of parallel workers.')
    parser.add_argument('--path_csv', type=str, default='~/Screen_dataset/dataset/DEKOIS2_SurfDock_pose.csv', help='path csv file')
    parser.add_argument('--out_dir', type=str, default='~/Screen_dataset/SurfDock_multi_pose_minimized', help='save_dir')
    parser.add_argument('--head_index', type=int, default=0, help='the head index to start minimized,this optinal to minimized use multi-GPU every GPU minimized a part of sdfs')
    parser.add_argument('--tail_index', type=int, default=-1, help='the tail index to start minimized,this optinal to minimized use multi-GPU every GPU minimized a part of sdfs')
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    """Init force field"""
    start_time = time.time()
    platform = GetPlatformPara()
    system_generator = GetFFGenerator(ignoreExternalBonds=True)
    system_generator_gaff = GetFFGenerator(small_molecule_forcefield = 'gaff-2.11',ignoreExternalBonds=True)
    paths = pd.read_csv(args.path_csv)
    for protein_path,sdf_dir in zip(paths['protein_path'],paths['ligand_path']):
        pdbid = os.path.basename(protein_path).split('_')[0]
        try:
            logger.info(f'minimized for target {pdbid}.......')
            logger.info('Use default forcefield')
            receptor_path = protein_path
            fixer = GetfixedPDB(receptor_path)
            modeller = Modeller(fixer.topology, fixer.positions)
            if os.path.isdir(sdf_dir):
                logger.info(f" {sdf_dir} is a Dir path,if you want to minimized just a file like relax for esmfold-ligand complex,please check the ligand_path !")
                # pass
                sdf_paths = glob(os.path.join(sdf_dir, '*.sdf'))
                # this code use to energy minimized the top N pose for every molecule
                # select confidence topN pose to minimized
                sdf_pd = pd.DataFrame({'pred_sdf_name':sdf_paths})
                sdf_pd['molecule_name'] = sdf_pd['pred_sdf_name'].apply(lambda x: os.path.basename(x).split('_sample_idx_')[0])
                sdf_pd['confidence'] = sdf_pd['pred_sdf_name'].apply(lambda x: float(os.path.basename(x).split('_confidence_')[-1].split('.sdf')[0]))
                # selected the topN confidence pose
                result = sdf_pd.sort_values('confidence',ascending=False)
                result_group = result.groupby('molecule_name')
                result = result_group.head(args.head_num)
                top1_sdfs = result['pred_sdf_name'].tolist()[args.head_index:args.tail_index]
            else:
                logger.info(f"Only minimized file {sdf_dir},if you want to minimized docking result from a Dir ,please check the ligand_path !")
                logger.info(f"Only minimized file {sdf_dir},head_num,head_index, tail_index, out_dir params will unable!")
                args.out_dir = os.path.dirname(os.path.dirname(sdf_dir))
                top1_sdfs = [sdf_dir]
            
            
            logger.info(f"ALL About {len(top1_sdfs)} sdfs to minimize , try to skip files have done!")
            if os.path.isdir(sdf_dir):
                # check out_dir done have optimized file and filter optimized files
                if os.path.exists(os.path.join(args.out_dir ,os.path.basename(sdf_dir))):
                    finished_files=os.listdir(os.path.join(args.out_dir ,os.path.basename(sdf_dir)))
                else:
                    finished_files = []
                if os.path.exists(os.path.join(args.out_dir ,os.path.basename(sdf_dir) + '_tmp')):
                    finished_files.extend(os.listdir(os.path.join(args.out_dir ,os.path.basename(sdf_dir) + '_tmp')))

                top1_sdfs = list(filter(lambda x:os.path.splitext(os.path.basename(x))[0]+ '_minimized.sdf' not in finished_files and \
                    os.path.splitext(os.path.basename(x))[0]+ '_unminimized.sdf' not in finished_files
                    ,top1_sdfs))
            else:
                
                if os.path.exists(os.path.splitext(os.path.basename(top1_sdfs[0]))[0] + '_minimized.sdf') or os.path.exists(os.path.splitext(os.path.basename(top1_sdfs[0]))[0] + '_unminimized.sdf') :
                    logger.info(f"{os.path.splitext(os.path.basename(top1_sdfs[0]))[0]} have been minimized,skip it!")
                    # finished_files = []
                    continue
                else:
                    finished_files = []

               

            logger.info(f"Minimizeing...... {len(finished_files)} sdfs have Minimized || left {len(top1_sdfs)} sdfs to Minimizing......")

            logger.info(f"Trying...... create system for protein!")
            failed_create_system = False
            
            for test_idx in range(len(top1_sdfs)):
                try:
                    dockingpose = read_abs_file_mol(top1_sdfs[test_idx], remove_hs=True, sanitize=True)
                    lig_mol = Molecule.from_rdkit(dockingpose,allow_undefined_stereo=True)
                    # set formal_charge use gasteiger
                    lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
                    modeller = trySystem(system_generator_gaff,modeller,lig_mol,top1_sdfs[test_idx])
                    failed_create_system = False
                    break
                except:
                    logger.info(f"ERROR in create system step! try anather molecule ing....., or you can check the protein please!")
                    failed_create_system = True
                    continue
            
            if failed_create_system:
                logger.info(f"ERROR For create system for protein!,check in error_for_create_system.txt")
                with open('error_for_create_system.txt','a') as f:
                    f.write(receptor_path +': Create system error! by :' + '\n')
                continue


            if modeller is None:
                print('Create system error!')
                with open('error_for_create_system.txt','a') as f:
                    f.write(receptor_path +': Create system error! by :' + '\n')
                logger.info(f"ERROR For create system for protein!,check in error_for_create_system.txt")
                continue
            logger.info(f"Done For create system for protein!,Start to Minimize sdf file")
            
            protein_atoms = list(modeller.topology.atoms())

            with Parallel(n_jobs=args.num_process,) as parallel:
                new_data_list = parallel(delayed(UpdatePose)(lig_path,system_generator,modeller,protein_atoms,args.out_dir) for lig_path in top1_sdfs)
            # selected the failed samples and try to use gaff-2.11 forcefield
            if sum(new_data_list) != 0:
                result = np.array(new_data_list)
                indices = np.where(result == 1)
                failed_sdfs = [top1_sdfs[i] for i in indices[0]]
                logger.info(f'Minimized not Completed:{pdbid}, {len(failed_sdfs)} sdf not be minimized by default forcefield , try use gaff-2.11 forcefield!')
                with Parallel(n_jobs=args.num_process) as parallel:
                    new_data_list = parallel(delayed(UpdatePose)(lig_path,system_generator_gaff,modeller,protein_atoms,args.out_dir) for lig_path in failed_sdfs)
            
            if sum(new_data_list) != 0:
                
                logger.info(f'Minimized not Completed:{pdbid}, {sum(new_data_list)} sdf not be minimized,use unminimized conformers for later stage')
                # save unminimized conformers
                result = np.array(new_data_list)
                indices = np.where(result == 1)
                failed_sdfs = [top1_sdfs[i] for i in indices[0]]
                out_base_dir = os.path.join(args.out_dir,failed_sdfs[0].split('/')[-2])
                cwd_path = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(out_base_dir,exist_ok=True)
                for lig_path in failed_sdfs:
                    out_file = os.path.join(out_base_dir,os.path.splitext(os.path.basename(lig_path))[0] + '_unminimized.sdf')
                    command = f"cp {lig_path} {out_file}"
                    run_command(command=command,cwd_path = cwd_path )
            logger.info(f'Finish minimized target {pdbid}')
        except Exception as e:
            warnings.warn(f'{pdbid} faild with {str(e)}')
            error_info = traceback.format_exc()
            print(error_info)
    end_time = time.time()
    logger.info(f"Time taken for optimizing {len(paths)} molecules: {end_time - start_time:.2f} seconds")
