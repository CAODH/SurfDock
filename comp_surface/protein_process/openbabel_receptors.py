# you need openbabel installed to use this (can be installed with anaconda)
import os
import subprocess

import time

from tqdm import tqdm

from openbabel import openbabel
# Create a molecule
mol = openbabel.OBMol()
# Create a conversion object
conv = openbabel.OBConversion()
conv.SetInAndOutFormats("pdb", "pdb")


start_time = time.time()
data_path =  '~/timesplit_test_/'
overwrite = False
names = sorted(os.listdir(data_path))

for i, name in tqdm(enumerate(names)):
    # if 
    rec_path = os.path.join(data_path, name, f'{name}_esmfold_aligned_tr_obabel_reduce.pdb')
    conv.ReadFile(mol, rec_path)
    # # Add hydrogens
    # mol.AddHydrogens()
    out_path = os.path.join(data_path, name, f'{name}_esmfold_aligned_tr_obabel_reduce_obabel.pdb')
    conv.WriteFile(mol, out_path)
# 


print("--- %s seconds ---" % (time.time() - start_time))
