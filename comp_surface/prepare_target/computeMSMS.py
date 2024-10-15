import os
from subprocess import Popen, PIPE

from input_output.read_msms import read_msms
from triangulation.xyzrn import output_pdb_as_xyzrn
from default_config.global_vars import msms_bin 
from default_config.masif_opts import masif_opts
import random

"""
Modified from: 
computeMSMS.py - MaSIF
Pablo Gainza - LPDI STI EPFL 2019
"""

# Pablo Gainza LPDI EPFL 2017-2019
# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.
import time
def computeMSMS(pdb_file,  protonate=True, one_cavity=None):
    randnum = random.randint(1,10000000) #+ time.time() + os.getpid()
    file_base = masif_opts['tmp_dir']+"/msms_"+str(randnum)
    out_xyzrn = file_base+".xyzrn"

    if protonate:        
        output_pdb_as_xyzrn(pdb_file, out_xyzrn)
    else:
        print("Error - pdb2xyzrn is deprecated.")
        sys.exit(1)
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    if one_cavity is not None:
        args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe", "1.5",\
                "-one_cavity", str(1), str(one_cavity),\
                "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    else:
        args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-all_components", "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    print(stdout, stderr)

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]
    # os.system("rm " + file_base + "*")
    return vertices, faces, normals, names, areas

