import os
import subprocess
import time
from tqdm import tqdm
from openbabel import openbabel

def main(data_path,save_path):
    # Create a molecule
    mol = openbabel.OBMol()
    # Create a conversion object
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdb")

    start_time = time.time()
    overwrite = False
    names = sorted(os.listdir(data_path))
    sucessed_names = []
    for i, name in tqdm(enumerate(names)):
        # if name in ['6gdy', '6gdy', '6i41', '6i41', '6i5p', '6i5p', '6i7a', '6i7a', '6m7h', '6m7h', '6ny0', '6ny0', '6qge', '6qge', '6qr3', '6qr3', '6qsz', '6qsz', '6qtm', '6qtm', '6qto', '6qto', '6qtq', '6qtq', '6qtr', '6qtr', '6qts', '6qts', '6qtw', '6qtw', '6qtx', '6qtx', '6t6a', '6t6a']:
            os.makedirs(os.path.join(save_path, name), exist_ok=True)
            result_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
            if os.path.exists(result_path):
                continue
            # step 1 openbabel
            rec_path = os.path.join(data_path, name, f'{name}_protein_processed.pdb')
            conv.ReadFile(mol, rec_path)

            out_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel.pdb')
            conv.WriteFile(mol, out_path)
            # step 2 reduce 
            rec_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel.pdb')
            return_code = subprocess.run(
                f"reduce -Trim {rec_path} > {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}", shell=True)

            return_code2 = subprocess.run(
                f"reduce -HIS {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')} > {os.path.join(data_path, name, f'{name}_protein_processed_obabel_reduce.pdb')}", shell=True)

            return_code2 = subprocess.run(
                f"rm {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}",
                shell=True)
            
            # step 3 openbabel
            rec_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce.pdb')
            conv.ReadFile(mol, rec_path)

            out_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
            conv.WriteFile(mol, out_path)
            sucessed_names.append(name)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"sucessed_names: {sucessed_names}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='Path to the data directory')
    parser.add_argument('--save_path', type=str, default='', help='Path to the data directory')
    args = parser.parse_args()
    # return args
    os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist.")
    main(args.data_path,args.save_path)