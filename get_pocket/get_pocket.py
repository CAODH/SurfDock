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

def preprocessor(docking_result_sdf_fn,origin_recptor_pdb,data_dir):
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
    sdf_fn = docking_result_sdf_fn.split("/")[-1].split(".")[0] 
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if os.path.getsize(docking_result_sdf_fn): #docking ligand file may be 0 size
        total=Chem.SDMolSupplier(docking_result_sdf_fn)
        for i,m1 in enumerate(total):
            key=sdf_fn 
            if not os.path.exists(os.path.join(data_dir,key)):

                if len(m1.GetConformers())==0:
                    print(f"{key} mol no conformer!")
                    continue
                try:
                    m2 = extract(m1, origin_recptor_pdb,key)
                except:
                    print(f'extract m2 failed {sdf_fn}')
                    continue

                if m2 is None :
                    print(f"{key} no extracted binding pocket!")
                    continue
                if len(m2.GetConformers())==0:
                    print(f"{key} receptor no conformer!")
                    continue

                with open(os.path.join(data_dir,key), "wb") as fp:
                    pickle.dump((m1, m2), fp, pickle.HIGHEST_PROTOCOL)
            else:
                print(f'file done before so skip it  {sdf_fn}')
                continue
        return 0
            
    else:
        print("docking result file size is 0!")
        return -1
def out_sdf(lig,filename):
    writer = Chem.SDWriter(filename)
    writer.write(lig)
    writer.close()
    return
def get_pocket_with_water(complex_sample,receptor_fn):
    status=preprocessor(complex_sample,receptor_fn,out_data_dir)
    # print(status)
if __name__ == '__main__':

    import time
    from multiprocessing import Pool
    import os
    import gzip
    import tqdm
    # get pocket and save to file
    import argparse
    parser = argparse.ArgumentParser(description='Process data from docking result')
    parser.add_argument("--single_sdf_save_path", help="file path for save compounds from docking result.", type=str, \
        default=None,required=True)
    parser.add_argument("--docking_result", help="docking result filname.maegz,filename.mae or filename.sdf.", type=str,default=None,required=True)
    parser.add_argument("--recptor_pdb", help="receptor pdb file.", type=str,default=None,required=True)
    parser.add_argument("--pocket_save_dir", help="save pocket file dir.", type=str,default=None,required=True)
    parser.add_argument("--prefix", help="Anything that helps you distinguish between compounds.", type=str,default='Compound')
    parser.add_argument("--process_num", help="process num for multi process ", type=int,default=1)
    args = parser.parse_args()
    os.makedirs(args.single_sdf_save_path,exist_ok=True)
    if args.docking_result.endswith('maegz'):
        total=Chem.rdmolfiles.MaeMolSupplier(gzip.open(args.docking_result))
    elif args.docking_result.endswith('sdf'):
        total=Chem.SDMolSupplier(args.docking_result)
    elif args.docking_result.endswith('mae'):
        total=Chem.rdmolfiles.MaeMolSupplier(args.docking_result)
    else:
        print('docking result file format error! only support maegz,mae or sdf')
        exit()
   
    for i,sample in enumerate(total):
        if i==0 and len(sample.GetAtoms()) > 500:
            print('atoms nums',len(sample.GetAtoms()),'may you not split protein and compounds ? save protein in a file in this dir')
            Chem.MolToPDBFile(sample,f'./data/protein.pdb')
            print('save protein success')
        else:
            if sample is  not None:
                name = '{}_{}_{}.sdf'.format(os.path.basename(args.docking_result).split('.')[0],args.prefix,i)
                out_sdf(sample,os.path.join(args.single_sdf_save_path,name))
    
    total_sdfs = [os.path.join(args.single_sdf_save_path,filename)for filename in os.listdir(args.single_sdf_save_path)]
    file_tuple_list = []
    for complex_sample in total_sdfs:
        receptor_fn=args.recptor_pdb
        file_tuple_list.append((complex_sample,receptor_fn))
    print('num compounds to get pocket',len(file_tuple_list))
    out_data_dir = args.pocket_save_dir
    p = Pool(args.process_num)
    pbar = tqdm.tqdm(total=len(file_tuple_list))
    pbar.set_description('get_pocket:')
    update = lambda *args: pbar.update() # set callback function to update pbar state when process end
    for file_tuple in file_tuple_list:
        p.apply_async(get_pocket_with_water,args = (file_tuple[0],file_tuple[1]),callback=update)
    print('waiting for processing!')
    p.close()
    p.join()
    print("all pocket done! check the outdir plz!")