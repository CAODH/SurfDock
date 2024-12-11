import pandas as pd
# from defaultdict import defaultdict
from collections import defaultdict
import os
from argparse import ArgumentParser, Namespace, FileType
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='~/SurfDock/data/test_samples', help='')
parser.add_argument('--surface_out_dir', type=str, default='~/SurfDock/data/test_samples_8A_surface', help='')
parser.add_argument('--Screen_ligand_library_file', type=str, default=None, help='')
parser.add_argument('--output_csv_file', type=str, default='~/SurfDock/data/test_samples_8A_surface', help='')
parser.add_argument('--is_docking_result_dir', action='store_true', default=False, help='')
parser.add_argument('--docking_result_dir', type=str, default='', help='')
# dirname = os.path.splitext(pocket_path.split('/')[-1])[0] + '_'+ os.path.splitext(ligands_path.split('/')[-1])[0] 
# write_dir =  os.path.join(args.out_dir,'SurfDock_docking_result',dirname)#f'{args.out_dir}/SurfDock_docking_result/{dirname}'
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_csv_file),exist_ok=True)
from tqdm import tqdm

args_list=defaultdict(list)
proteins = [i for i in os.listdir(args.surface_out_dir) if os.path.isdir(os.path.join(args.surface_out_dir, i)) ]
for protein in tqdm(proteins ):
        target_filename = os.path.join(args.data_dir,protein,f'{protein}_protein_processed_obabel_reduce_obabel.pdb')
        if not os.path.exists(target_filename):
            target_filename = os.path.join(args.data_dir,protein,f'{protein}_protein_processed.pdb')
        if not os.path.exists(target_filename):
            raise ValueError(f'{target_filename} not exists , Please check file name or path')

        ref_ligand_filename = os.path.join(args.data_dir,protein,f'{protein}_ligand.sdf')
        ligand_filename = os.path.join(args.data_dir,protein,f'{protein}_ligand.sdf')
        if args.Screen_ligand_library_file is not None:
            print(f'Using Screen ligands library file: {args.Screen_ligand_library_file}')
            ligand_filename = args.Screen_ligand_library_file
        
        if os.path.exists(ref_ligand_filename):

            pocket = os.path.join(args.surface_out_dir, protein, f'{protein}_protein_processed_8A.pdb')
            surface = os.path.join(args.surface_out_dir, protein, f'{protein}_protein_processed_8A.ply')
            
            if os.path.exists(pocket) and os.path.exists(surface):

                args_list['protein_path'].append(target_filename)
                args_list['pocket_path'].append(pocket)
                args_list['ref_ligand'].append(ref_ligand_filename)
                
                if args.is_docking_result_dir:
                    dirname = os.path.splitext(pocket.split('/')[-1])[0] + '_'+ os.path.splitext(ligand_filename.split('/')[-1])[0] 
                    # write_dir =  os.path.join(args.docking_result_dir,'SurfDock_docking_result',dirname)#f'{args.out_dir}/SurfDock_docking_result/{dirname}'
                    args_list['ligand_path'].append(os.path.join(args.docking_result_dir,'SurfDock_docking_result',dirname))
                else:
                    args_list['ligand_path'].append(ligand_filename)
                args_list['protein_surface'].append(surface)
            else:
                pass
                print(pocket)
        else:
            
            print(protein)
pd.DataFrame(args_list).to_csv(args.output_csv_file,index=False)