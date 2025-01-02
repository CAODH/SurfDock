##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Jan  2 07:04:34 2025

##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path 1a0q_temp
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 4, dime = (97, 97, 97)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 802 atoms
Valist_getStatistics:  Max atom coordinate:  (28.002, 34.321, 69.127)
Valist_getStatistics:  Min atom coordinate:  (0.325, 5.03, 44.364)
Valist_getStatistics:  Molecule center:  (14.1635, 19.6755, 56.7455)
NOsh_setupCalcMGAUTO(/home/runner/work/apbs/apbs/src/generic/nosh.c, 1868):  coarse grid center = 14.1635 19.6755 56.7455
NOsh_setupCalcMGAUTO(/home/runner/work/apbs/apbs/src/generic/nosh.c, 1873):  fine grid center = 14.1635 19.6755 56.7455
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1885):  Coarse grid spacing = 0.539697, 0.560168, 0.475398
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1887):  Fine grid spacing = 0.525802, 0.537844, 0.475398
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1889):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.974254, 0.960148, 1 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1983):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1985):  coarse mesh center = 14.1635 19.6755 56.7455
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1990):  coarse mesh upper corner = 40.069 46.5635 79.5646
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 1995):  coarse mesh lower corner = -11.7419 -7.21255 33.9264
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 2000):  initial fine mesh upper corner = 39.402 45.492 79.5646
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 2005):  initial fine mesh lower corner = -11.075 -6.141 33.9264
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 2066):  final fine mesh upper corner = 39.402 45.492 79.5646
NOsh_setupCalcMGAUTO (/home/runner/work/apbs/apbs/src/generic/nosh.c, 2071):  final fine mesh lower corner = -11.075 -6.141 33.9264
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 18.1075
Vpbe_ctor2:  solute dimensions = 30.477 x 31.633 x 26.846
Vpbe_ctor2:  solute charge = 1
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 60 x 63 x 53 table
Vclist_ctor2:  Using 60 x 63 x 53 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (38.753, 40.367, 35.839)
Vclist_setupGrid:  Grid lower corner = (-5.213, -0.508, 38.826)
Vclist_assignAtoms:  Have 1174850 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.075197
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 3.289760e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (097, 097, 097)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 3.623700e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (049, 049, 049)
Vbuildops: Galer: (025, 025, 025)
Vbuildops: Galer: (013, 013, 013)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 9.478600e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.874140e-01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.400615e-01
Vprtstp: contraction number = 1.400615e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.942548e-02
Vprtstp: contraction number = 1.386925e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.923254e-03
Vprtstp: contraction number = 1.504855e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 4.599695e-04
Vprtstp: contraction number = 1.573485e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 7.601479e-05
Vprtstp: contraction number = 1.652605e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.302875e-05
Vprtstp: contraction number = 1.713976e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 2.344366e-06
Vprtstp: contraction number = 1.799379e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 4.438507e-07
Vprtstp: contraction number = 1.893266e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 4.554880e-01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 6.049670e-01
Vpmg_setPart:  lower corner = (-11.7419, -7.21255, 33.9264)
Vpmg_setPart:  upper corner = (40.069, 46.5635, 79.5646)
Vpmg_setPart:  actual minima = (-11.7419, -7.21255, 33.9264)
Vpmg_setPart:  actual maxima = (40.069, 46.5635, 79.5646)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.215489098079E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 1.458000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 2.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 18.1075
Vpbe_ctor2:  solute dimensions = 30.477 x 31.633 x 26.846
Vpbe_ctor2:  solute charge = 1
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 60 x 63 x 53 table
Vclist_ctor2:  Using 60 x 63 x 53 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (38.753, 40.367, 35.839)
Vclist_setupGrid:  Grid lower corner = (-5.213, -0.508, 38.826)
Vclist_assignAtoms:  Have 1174850 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = -11.075, -6.141, 33.9264
VPMG::focusFillBound -- New mesh maxs = 39.402, 45.492, 79.5646
VPMG::focusFillBound -- Old mesh mins = -11.7419, -7.21255, 33.9264
VPMG::focusFillBound -- Old mesh maxs = 40.069, 46.5635, 79.5646
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (-11.075, -6.141, 33.9264)
Vpmg_setPart:  upper corner = (39.402, 45.492, 79.5646)
Vpmg_setPart:  actual minima = (-11.7419, -7.21255, 33.9264)
Vpmg_setPart:  actual maxima = (40.069, 46.5635, 79.5646)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (-11.075, -6.141, 33.9264)
VPMG::extEnergy    Disj part upper corner = (39.402, 45.492, 79.5646)
VPMG::extEnergy    Old lower corner = (-11.7419, -7.21255, 33.9264)
VPMG::extEnergy    Old upper corner = (40.069, 46.5635, 79.5646)
Vpmg_qmEnergy:  Zero energy for zero ionic strength!
VPMG::extEnergy: extQmEnergy = 0 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 0.0171746 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.072583
Vpmg_fillco:  done filling coefficient arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 3.741580e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (097, 097, 097)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 3.613700e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (049, 049, 049)
Vbuildops: Galer: (025, 025, 025)
Vbuildops: Galer: (013, 013, 013)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 9.354200e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.469380e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.412028e-01
Vprtstp: contraction number = 1.412028e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.912901e-02
Vprtstp: contraction number = 1.354719e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.814956e-03
Vprtstp: contraction number = 1.471564e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 4.346573e-04
Vprtstp: contraction number = 1.544100e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 7.188343e-05
Vprtstp: contraction number = 1.653796e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.284600e-05
Vprtstp: contraction number = 1.787060e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 2.615661e-06
Vprtstp: contraction number = 2.036167e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 5.807000e-07
Vprtstp: contraction number = 2.220089e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 4.556370e-01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 6.018910e-01
Vpmg_setPart:  lower corner = (-11.075, -6.141, 33.9264)
Vpmg_setPart:  upper corner = (39.402, 45.492, 79.5646)
Vpmg_setPart:  actual minima = (-11.075, -6.141, 33.9264)
Vpmg_setPart:  actual maxima = (39.402, 45.492, 79.5646)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.301018949492E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 1.398000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 2.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
printEnergy:  Performing global reduction (sum)
Vcom_reduce:  Not compiled with MPI, doing simple copy.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 2.234311e+00
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Jan  2 07:04:36 2025

##############################################################################
Vgrid_readDX:  Grid dimensions 97 x 97 x 97 grid
Vgrid_readDX:  Grid origin = (-11.075, -6.141, 33.9264)
Vgrid_readDX:  Grid spacings = (0.525802, 0.537844, 0.475398)
Vgrid_readDX:  allocating 97 x 97 x 97 doubles for storage
