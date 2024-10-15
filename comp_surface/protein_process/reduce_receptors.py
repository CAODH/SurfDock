# you need reduce installed to use this: https://github.com/rlabduke/reduce
import os
import subprocess

import time

from tqdm import tqdm

start_time = time.time()
data_path =  'dfgh/'
overwrite = False
names = sorted(os.listdir(data_path))

for i, name in tqdm(enumerate(names)):
    rec_path = os.path.join(data_path, name, f'{name}_esmfold_aligned_tr.pdb')
    return_code = subprocess.run(
        f"reduce -Trim {rec_path} > {os.path.join(data_path, name, f'{name}_esmfold_aligned_tr.pdb')}", shell=True)
    print(return_code)
    return_code2 = subprocess.run(
        f"reduce -HIS {os.path.join(data_path, name, f'{name}_esmfold_aligned_tr.pdb')} > {os.path.join(data_path, name, f'{name}_esmfold_aligned_tr_reduce.pdb')}", shell=True)
    print(return_code2)
    return_code2 = subprocess.run(
        f"rm {os.path.join(data_path, name, f'{name}_esmfold_aligned_tr.pdb')}",
        shell=True)
    print(return_code2)



print("--- %s seconds ---" % (time.time() - start_time))
