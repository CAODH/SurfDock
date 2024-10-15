import os
import subprocess
import time
from tqdm import tqdm
from openbabel import openbabel

start_time = time.time()
data_path =  '~/DockGen-sdf/processed_files/'
overwrite = True

names = sorted(os.listdir(data_path))
sucessed_names = []
for i, name in tqdm(enumerate(names)):
    # if name in ['6gdy', '6gdy', '6i41', '6i41', '6i5p', '6i5p', '6i7a', '6i7a', '6m7h', '6m7h', '6ny0', '6ny0', '6qge', '6qge', '6qr3', '6qr3', '6qsz', '6qsz', '6qtm', '6qtm', '6qto', '6qto', '6qtq', '6qtq', '6qtr', '6qtr', '6qts', '6qts', '6qtw', '6qtw', '6qtx', '6qtx', '6t6a', '6t6a']:
        
        result_path = os.path.join(data_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
        if os.path.exists(result_path) and not overwrite:
            continue
        # step 1 openbabel
        rec_path = os.path.join(data_path, name, f'{name}_protein_processed.pdb')
        # Create a molecule
        mol = openbabel.OBMol()
        # Create a conversion object
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdb")
        conv.ReadFile(mol, rec_path)
        out_path = os.path.join(data_path, name, f'{name}_protein_processed_obabel.pdb')
        conv.WriteFile(mol, out_path)
        # step 2 reduce 
        rec_path = os.path.join(data_path, name, f'{name}_protein_processed_obabel.pdb')
        return_code = subprocess.run(
            f"reduce -Trim {rec_path} > {os.path.join(data_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}", shell=True)

        return_code2 = subprocess.run(
            f"reduce -HIS {os.path.join(data_path, name, f'{name}_protein_processed_obabel_tmp.pdb')} > {os.path.join(data_path, name, f'{name}_protein_processed_obabel_reduce.pdb')}", shell=True)

        return_code2 = subprocess.run(
            f"rm {os.path.join(data_path, name, f'{name}_protein_processed_obabel_tmp.pdb')}",
            shell=True)
        
        # step 3 openbabel
        rec_path = os.path.join(data_path, name, f'{name}_protein_processed_obabel_reduce.pdb')
        # Create a molecule
        mol1 = openbabel.OBMol()
        # Create a conversion object
        conv1 = openbabel.OBConversion()
        conv1.SetInAndOutFormats("pdb", "pdb")
        conv1.ReadFile(mol1, rec_path)

        out_path = os.path.join(data_path, name, f'{name}_protein_processed_obabel_reduce_obabel.pdb')
        conv.WriteFile(mol1, out_path)
        sucessed_names.append(name)
print("--- %s seconds ---" % (time.time() - start_time))
# print(f"sucessed_names: {sucessed_names}")