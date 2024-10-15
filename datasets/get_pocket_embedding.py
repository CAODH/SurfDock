# by caoduanhua email : caodh@zju.edu.cn
# cycle chain and residues
import pickle
import os
from Bio.PDB import *
import warnings
warnings.filterwarnings('ignore')
import tqdm
from Bio.PDB import PDBParser
biopython_parser = PDBParser()
import torch
import pandas as pd
from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument('--pocket_dir', type=str, default='~/PDBBind_pocket_8A', help='pocket dir locations')
# parser.add_argument('--full_protein_dir', type=str, default='~/PDBBind_processed', help='full protein dir locations')
parser.add_argument('--protein_pocket_csv', type=str, default='~/processsed/PDBBIND.csv', help='save pocket and full protein csv locations')
parser.add_argument('--embeddings_dir', type=str, default='~/esm_embedding/esm_embedding_output', help='full protein embedding dir locations')
parser.add_argument('--pocket_emb_save_dir', type=str, default='~/esm_embedding/esm_embedding_output_pocket_new', help='')
args = parser.parse_args()

df = pd.read_csv(args.protein_pocket_csv)
full_protein_paths = list(df['protein_path'].tolist())
pocket_paths = list(df['pocket_path'].tolist())
# pocket_dir = args.pocket_dir
# full_protein_dir = args.full_protein_dir
protein_pocket_csv = args.protein_pocket_csv
embeddings_dir = args.embeddings_dir
pocket_emb_save_dir = args.pocket_emb_save_dir

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}
# if os.path.exists(
Assertion_list = []
os.makedirs(pocket_emb_save_dir,exist_ok = True)
# pbar = tqdm.tqdm(os.listdir(pocket_dir),total=len(os.listdir(pocket_dir)))
pbar = tqdm.tqdm(zip(full_protein_paths,pocket_paths),total=len(full_protein_paths))

for pbar_idx,(full_protein_path,pocket_path) in enumerate(pbar):
    protein_name = os.path.splitext(os.path.basename(pocket_path))[0]
    # raise  AssertionError(protein_name)# if False else None
    if os.path.exists(os.path.join(pocket_emb_save_dir,f'{protein_name}.pt')):
        pbar.set_description(f'have done ,just skip!')
        continue
    try:
        pocket = pocket_path
        full_protein = full_protein_path
        # full_protein = f'{full_protein_dir}/{pdb_id}.pdb'
        pocket_structure = biopython_parser.get_structure(f"{protein_name}", pocket)[0]
        full_structure = biopython_parser.get_structure(f"{protein_name}", full_protein)[0]
        pocket_embeddings =[]
        pocket_infos_all = []
        for i,chain in enumerate(full_structure.get_chains()):
            chain_id = chain.get_id()
            try:
                pocket_chain = pocket_structure[chain_id]
            except KeyError:
                pbar.set_description(f'{chain_id} not in {protein_name} pocket skip this chain')
                continue
            try:

                embeddings_path_chain = os.path.join(embeddings_dir,f'{os.path.basename(full_protein_path)}_chain_{i}.pt')
                # embeddings_path_chain = os.path.join(embeddings_dir,f'{pdb_id}.pdb_chain_{i}.pt')
                embeddings = torch.load(embeddings_path_chain)['representations'][33]
                assert len(list(chain.get_residues())) == len(embeddings),'embedding must equal to res nums!'
            except AssertionError:
                # pbar.set_description(f'{pdb_id} has error!,{len(list(chain.get_residues()))},{len(embeddings)}')
                # Assertion_list.append(pdb_id)
                residue_list = list(chain.get_residues())
                for res_idx, residue in enumerate(residue_list):
                # for res_idx, residue in enumerate(chain):
                    if residue.get_resname() == 'HOH':
                        chain.detach_child(residue.get_id())
                        continue
                    c_alpha, n, c = None, None, None
                    for atom in residue:
                        if atom.name == 'CA':
                            c_alpha = list(atom.get_vector())
                        if atom.name == 'N':
                            n = list(atom.get_vector())
                        if atom.name == 'C':
                            c = list(atom.get_vector())
                    if c_alpha != None and n != None and c != None:  
                        continue
                    else:
                        chain.detach_child(residue.get_id())
                        continue

                assert len(list(chain.get_residues())) == len(embeddings),f'embedding must equal to res nums! {len(list(chain.get_residues()))},{len(embeddings)}'
            pocket_infos = []
            pocket_residue_list = list(pocket_chain.get_residues())
            for res_idx, residue in enumerate(pocket_residue_list):

                if residue.get_resname() == 'HOH':

                    continue
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N':
                        n = list(atom.get_vector())
                    if atom.name == 'C':
                        c = list(atom.get_vector())
                if c_alpha != None and n != None and c != None:  
                    pocket_infos += [residue.get_id()]
            #         continue
                else:

                    print(residue.get_resname())
                    continue

            pocket_infos_all += pocket_infos
            # check the res in pocket 
            pocket_idx_list = []
            for res_idx,res in enumerate(chain.get_residues()):
                if res.get_id() in pocket_infos:
                    pocket_idx_list.append(res_idx)
                # else:

            pocket_embeddings.append(embeddings[pocket_idx_list])
        pocket_embeddings = torch.cat(pocket_embeddings,dim = 0)
        assert len(pocket_embeddings) == len(pocket_infos_all),f'pocket embedding must equal to res nums! {len(pocket_embeddings)},{len(pocket_infos_all)}'
        torch.save(pocket_embeddings,os.path.join(pocket_emb_save_dir,f'{protein_name}.pt'))
    except AssertionError as e:
        print(e,protein_name)

        Assertion_list.append(protein_name)

        continue
    except FileNotFoundError as e:
        Assertion_list.append(protein_name)

        continue
    except Exception as e:
        Assertion_list.append(protein_name)

        continue
    pbar.set_description(f'{pbar_idx}/{len(full_protein_paths)} done!')
print('Assertion_list:',Assertion_list)