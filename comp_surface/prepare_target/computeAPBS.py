import os
import numpy
from subprocess import Popen, PIPE
import pymesh
import tempfile

from default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin
import random

""" 
Modified from:
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
"""

def computeAPBS(vertices, pdb_file, tmp_file_base,clear=False):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """    
    if not clear:
        pdb2pqr = pdb2pqr_bin + " --ff=parse --whitespace --noopt --apbs-input %s %s"# + tempfile.mktemp() 
        # pdb2pqr = pdb2pqr_bin + " --clean --whitespace --noopt --apbs-input %s %s"# + tempfile.mktemp() # 上一行由于确实太多办法计算表面的时候再用这个
    else:
        pdb2pqr = pdb2pqr_bin + " --clean --whitespace --noopt --apbs-input %s %s"# + tempfile.mktemp() # 上一行由于确实太多办法计算表面的时候再用这个
    make_pqr = pdb2pqr % (pdb_file, tmp_file_base) 
    os.system(make_pqr)
    print('os.system(make_pqr)',os.system(make_pqr))
    
    apbs = apbs_bin + " %s"
    make_apbs = apbs % (tmp_file_base+".in") 
    # os.system(make_apbs)
    print(make_apbs)
    print('os.system(make_apbs)',os.system(make_apbs))
    
    vertfile = open(tmp_file_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()
    
    multivalue = multivalue_bin + " %s %s %s"
    make_multivalue = multivalue % (tmp_file_base+".csv", tmp_file_base+".dx", tmp_file_base+"_out.csv") 
    # print(make_multivalue)
    try:
        os.system(make_multivalue)
    except Exception as e:
        print(e)


    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])
    # os.system("rm " + tmp_file_base + "*")
    # os.system("rm io.mc")

    return charges



"""  ORIGINAL FUNCTION
'''
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
'''

def computeAPBS(vertices, pdb_file, tmp_file_base = tempfile.mktemp()):
    
        #Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    
    #fields = tmp_file_base.split("/")[0:-1]
    #directory = "/".join(fields) + "/"
    fields = tmp_file_base
    directory = str(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]
    args = [
        pdb2pqr_bin,
        "--ff=parse",
        "--whitespace",
        "--noopt",
        "--apbs-input",
        pdbname,
        filename_base,
    ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    args = [apbs_bin, filename_base + ".in"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    vertfile = open(directory + "/" + filename_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    args = [
        multivalue_bin,
        filename_base + ".csv",
        filename_base + ".dx",
        filename_base + "_out.csv",
    ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])

    return charges
"""