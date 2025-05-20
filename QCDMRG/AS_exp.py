from pyscf import gto, scf, lib, dmrgscf, mcscf
import os
import numpy as np  
from time import time
from pyscf.geomopt.berny_solver import optimize
from pyscf.mcscf import avas

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

"""
Systematic expansion of active space for N2
"""

# output file 'output/expand_AS.txt'


# Settings
basis = 'ccpvdz'
bond_dim = 200 
N_active_orbitals = [6, 8, 10, 12, 14]

# Optimize geometry and initialize object
mol = gto.M(atom = f'N 0 0 0; N 1.1 0 0', basis = basis)
mf = scf.RKS(mol)
mol_eq = optimize(mf)
mf = scf.RHF(mol_eq)
mf.kernel()

# Solve and write to file
f = open('output/expand_AS.txt', 'w')
f.write('AS size' + ' ' + 'CASSCF' + ' ' + 'DMRG' + ' ' + 't-CASSCF' + ' ' + 't-DMRG' + '\n')
for nactorb in N_active_orbitals:
	print('orbs:', nactorb)

	# CASSCF
	mycas = mcscf.CASSCF(mf, nactorb, nactorb)
	tic1 = time()
	mycas.kernel()
	toc1 = time() - tic1

	# DMRG-CASSCF
	mc = dmrgscf.DMRGSCF(mf, nactorb, nactorb, maxM = bond_dim)
	tic2 = time()
	mc.kernel()
	toc2 = time() - tic2

	f.write(str(nactorb) + ' ' + str(mycas.e_tot) + ' ' + str(mc.e_tot) + ' ' + str(toc1) + ' ' + str(toc2) + '\n')
f.close()










