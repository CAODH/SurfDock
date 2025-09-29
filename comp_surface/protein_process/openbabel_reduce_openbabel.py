import os
import subprocess
import time
from tqdm import tqdm
from openbabel import openbabel
from joblib import Parallel, delayed

def process_protein(name, data_path, save_path):
    mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdb")

    os.makedirs(os.path.join(save_path, name), exist_ok=True)
    result_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
    if os.path.exists(result_path):
        return name

    # step 1 openbabel
    rec_path = os.path.join(data_path, name, f'{name}_protein_processed.pdb')
    conv.ReadFile(mol, rec_path)

    out_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel.pdb')
    conv.WriteFile(mol, out_path)

    # step 2 reduce 
    rec_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel.pdb')
    subprocess.run(
        f"reduce -Trim {rec_path} > {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}", shell=True)

    subprocess.run(
        f"reduce -HIS {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')} > {os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce.pdb')}", shell=True)

    subprocess.run(
        f"rm {os.path.join(save_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}",
        shell=True)

    # step 3 openbabel
    rec_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce.pdb')
    conv.ReadFile(mol, rec_path)

    out_path = os.path.join(save_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
    conv.WriteFile(mol, out_path)

    return name

def main(data_path, save_path, n_jobs):
    start_time = time.time()
    names = sorted(os.listdir(data_path))
    os.makedirs(save_path, exist_ok=True)

    sucessed_names = Parallel(n_jobs=n_jobs)(
        delayed(process_protein)(name, data_path, save_path) for name in tqdm(names)
    )

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"sucessed_names: {list(filter(None, sucessed_names))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='Path to the data directory')
    parser.add_argument('--save_path', type=str, default='', help='Path to the save directory')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (-1 for all CPUs)')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist.")
    main(args.data_path, args.save_path, args.n_jobs)