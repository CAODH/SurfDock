import pickle
import os
import glob
from multiprocessing import Pool
import numpy as np
from rdkit import Chem
from scipy.spatial import distance_matrix
from Bio.PDB import *
from Bio.PDB.PDBIO import Select
import warnings
warnings.filterwarnings('ignore')
from rdkit.Chem import AllChem
def extract(ligand, pdb,key):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb)
    ligand_positions = ligand.GetConformer().GetPositions()
    # Get distance between ligand positions (N_ligand, 3) and
    # residue positions (N_residue, 3) for each residue
    # only select residue with minimum distance of it is smaller than 8A
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                for atom in residue.get_atoms()])    # if "H" not in atom.get_id()
            if len(residue_positions.shape) < 2:
                print(residue)
                return 0
            min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
            if min_dis < 8.0:
                return 1
            else:
                return 0
    
    io = PDBIO()
    io.set_structure(structure)
    fn = "BS_tmp_"+str(key)+".pdb"
    io.save(fn, ResidueSelect())
    try:
        m2 = Chem.MolFromPDBFile(fn)
        # may contain metal atom, causing MolFromPDBFile return None
        if m2 is None:
            print("first read PDB fail",fn)
            # copy file to tmp dir 
            remove_zn_dir="./docker_result_remove_ZN"
            if not os.path.exists(remove_zn_dir):
                os.mkdir(remove_zn_dir)
            cmd=f"cp {fn}   {remove_zn_dir}"
            print(cmd)
            os.system(cmd)
            fn_remove_zn=os.path.join(remove_zn_dir,fn.replace('.pdb','_remove_ZN.pdb'))
            cmd=f"sed -e '/ZN/d'  {fn}  > {fn_remove_zn}"
            os.system(cmd)
            print("delete metal atom and get new pdb file",fn_remove_zn)
            m2 = Chem.MolFromPDBFile(fn_remove_zn)
        else:
            os.system("rm -f " + fn)
    except:
        print("Read PDB fail for other unknow reason",fn)
    return m2

def preprocessor(ligand_dir,data_dir):
    """
    get pocket from docking result and save to file:(m1,m2)

    input:
        docking_result_sdf_fn: docking result sdf file, one ligand in sdf file will speed up this process in multi-process
        origin_recptor_pdb: receptor pdb file
        data_dir: path for save pocket file
    output:
        0: success
        -1: fail
    """
    file_flag = os.path.basename(ligand_dir)
    try:
        m1 = read_molecule(os.path.join(ligand_dir, f'{file_flag}_ligand.sdf'), remove_hs=True, sanitize=True)
        if m1 is None:  # read mol2 file if sdf file cannot be sanitized
            print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
            m1 = read_molecule(os.path.join(ligand_dir, f'{file_flag}_ligand.mol2'), remove_hs=True, sanitize=True)
    except Exception as e:
        print(e)
        return -1
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if m1 is not None: #docking ligand file may be 0 size
            if not os.path.exists(os.path.join(data_dir,file_flag)):
                os.mkdir(os.path.join(data_dir,file_flag))

                if len(m1.GetConformers())==0:
                    print(f"{file_flag} mol no conformer!")
                    return -1
                try:
                    pdb_path = os.path.join(ligand_dir, f'{file_flag}_protein_processed.pdb')
                    m2 = extract(m1,pdb_path ,file_flag)
                except:
                    print(f'extract m2 failed {file_flag}')
                    return -1

                if m2 is None :
                    print(f"{file_flag} no extracted binding pocket!")
                    # continue
                    return -1
                if len(m2.GetConformers())==0:
                    print(f"{file_flag} receptor no conformer!")
                    return -1
                # save pdb pocket
                Chem.MolToPDBFile(m2, os.path.join(data_dir,file_flag,f'{file_flag}_pocket.pdb'))

            else:
                print(f'file done before so skip it  {file_flag}')
                
                return 0
        # return 0
            
    else:
        print("read mol fail")
        return -1
def out_sdf(lig,filename):
    writer = Chem.SDWriter(filename)
    writer.write(lig)
    writer.close()
    return
def get_pocket_with_water(complex_sample):
    status=preprocessor(complex_sample,out_data_dir)
    # print(status)
def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

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
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol
if __name__ == '__main__':

    import time
    from multiprocessing import Pool
    import os
    import gzip
    import tqdm
    # get pocket and save to file
    import argparse
    parser = argparse.ArgumentParser(description='Process data from docking result')
    parser.add_argument("--PDBbind_path", help="file path for save compounds from docking result.", type=str, \
        default='/home/house/caoduanhua/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/PDBBind_processed/',required=False)
    # parser.add_argument("--docking_result", help="docking result filname.maegz,filename.mae or filename.sdf.", type=str,default=None,required=True)
    # parser.add_argument("--recptor_pdb", help="receptor pdb file.", type=str,default=None,required=True)
    parser.add_argument("--save_dir", help="save pocket file dir.", type=str,default='/home/house/caoduanhua/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/PDBBind_pocket_8A',required=False)
    # parser.add_argument("--prefix", help="Anything that helps you distinguish between compounds.", type=str,default='')
    parser.add_argument("--process_num", help="process num for multi process ", type=int,default=60)
    args = parser.parse_args()
    
    total_sdfs = [os.path.join(args.PDBbind_path,filename) for filename in os.listdir(args.PDBbind_path)]

    file_tuple_list = []
    for complex_sample in total_sdfs:
        # receptor_fn=args.recptor_pdb
        file_tuple_list.append(complex_sample)
    print('num compounds to get pocket',len(file_tuple_list))
    out_data_dir = args.save_dir
    p = Pool(args.process_num)
    pbar = tqdm.tqdm(total=len(file_tuple_list))
    pbar.set_description('get_pocket:')
    update = lambda *args: pbar.update() # set callback function to update pbar state when process end
    for file_tuple in file_tuple_list:
        p.apply_async(get_pocket_with_water,args = (file_tuple,),callback=update)
    print('waiting for processing!')
    p.close()
    p.join()
    print("all pocket done! check the outdir plz!")