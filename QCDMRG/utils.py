from pyscf import gto, scf, lib, dmrgscf, mcscf, cc, mrpt, fci
import os
import numpy as np  
from pyscf.geomopt.berny_solver import optimize
from ast import literal_eval
from time import time

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

"""
Initialize molecule objects
"""

# Initiale nitrogen dimer object
def initialize_N2(R, basis):
    mol = gto.M(atom = f'N 0 0 0; N {R} 0 0', basis = basis)
    return mol

# Initiale beryllium dihydride object
def initialize_BeH2(x, basis):
	y = 2.54 - 0.46 * x
	mol = gto.M(atom = f'Be {x} 0 0; H 0 {-y} 0; H 0 {y} 0', basis = basis)
	return mol

# Initialize chain of N H-atoms with a spacing of R
def initialize_H_chain(R, basis, N):
	for i in range(N):
		if i == 0:
			atom = f'H 0 0 0'
		else:
			atom += f'; H {i * R} 0 0'
	if N % 2 == 0:
		spin = 0
	else: 
		spin = 1
	mol = gto.M(atom = atom, basis = basis, spin = spin)
	return mol

# Initialize circle of N H-atoms with a spacing of R
def initialize_H_circle(R, basis, N):
	if N == 1:
		return gto.M(atom = 'H 0 0 0', basis = basis, spin = 1)
	elif N == 2:
		return gto.M(atom = f'H {- R / 2} 0 0; H {R / 2} 0 0', basis = basis)
	theta = 2 * np.pi / N  # angle between atoms
	r = R / (2 * np.sin(np.pi / N)) # radius
	coords = []
	for i in range(N):
		if i == 0:
			x = r * np.cos(i * theta)
			y = r * np.sin(i * theta)
			atom = f'H {x} {y} 0'
		else: 
			x = r * np.cos(i * theta)
			y = r * np.sin(i * theta)
			atom += f'; H {x} {y} 0'
	if N % 2 == 0:
		spin = 0
	else: 
		spin = 1
	mol = gto.M(atom = atom, basis = basis, spin = spin)
	return mol

"""
Single calculations
"""

# make RKS object (in case for switching RHF mf with this)
def RKS_calc(mol):
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'
    return mf

# return CCSD energy
def CCSD_calc(mf):
    mycc = cc.CCSD(mf).run()
    return mycc.e_tot

# return CASSCF energy
def CASSCF_calc(mf, nactorb, nactelec, mo_indices):
    mycas = mcscf.CASSCF(mf, nactorb, nactelec)
    mo = mycas.sort_mo(mo_indices)
    mycas.kernel(mo)
    return mycas.e_tot

# return DMRG-CASSCF energy
def DMRG_calc(mf, nactorb, nactelec, bond_dim, mo_indices):
    mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim)
    mo = mc.sort_mo(mo_indices)
    mc.kernel(mo)
    return mc.e_tot

# return CASSCF energy when mo_indices are not specified
def CASSCF_unspecified_orbs(mf, nactorb, nactelec):
	mycas = mcscf.CASSCF(mf, nactorb, nactelec).run()
	return mycas.e_tot

# return DMRG-CASSCF energy when mo_indices are not specified
def DMRG_unspecified_orbs(mf, nactorb, nactelec, bond_dim):
	mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim).run()
	return mc.e_tot


# Inspect orbital energies and occupations
def inspect_orbs(molecule, basis, R):
	if molecule == 'N2':
		mol = initialize_N2(R, basis)
	elif molecule == 'BeH2':
		mol = initialize_BeH2(R, basis)
	elif molecule == 'Hchain':
		mol = initialize_H_chain(R, basis, N)
	elif molecule == 'Hcircle':
		mol = initialize_H_circle(R, basis, N)

	mf = scf.RHF(mol).run()
	for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
		print(f"MO {i + 1:2d} | Energy: {e:8.4f} | Occ: {occ}")


"""
Data gathering
"""

# Compute PES (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF)
def compute_PES(basis, nactorb, nactelec, bond_dim, bond_lengths, molecule, mo_indices = [1,2], N = 2):
    energy_RHF = []
    energy_RKS = []
    energy_CCSD = []
    energy_CASSCF = []
    energy_DMRG = []

    for R in bond_lengths:
    	if molecule == 'N2':
    		mol = initialize_N2(R, basis)
    	elif molecule == 'BeH2':
    		mol = initialize_BeH2(R, basis)
    	elif molecule == 'Hchain':
    		mol = initialize_H_chain(R, basis, N)
    	elif molecule == 'Hcircle':
    		mol = initialize_H_circle(R, basis, N)

    	# RHF
    	mf = scf.RHF(mol).run()
    	energy_RHF.append(mf.e_tot)

    	# RKS
    	mf_RKS = RKS_calc(mol)
    	energy_RKS.append(mf_RKS.kernel())

    	# CCSD 
    	energy_CCSD.append(CCSD_calc(mf))

    	if molecule == 'Hchain' or molecule == 'Hcircle':
    		# If system has unspecified orbital indices
    		energy_CASSCF.append(CASSCF_unspecified_orbs(mf, nactorb, nactelec))
    		energy_DMRG.append(DMRG_unspecified_orbs(mf, nactorb, nactelec, bond_dim))
    	else: 
    	    # If orbital indices are specified
    	    energy_CASSCF.append(CASSCF_calc(mf, nactorb, nactelec, mo_indices))
    	    energy_DMRG.append(DMRG_calc(mf, nactorb, nactelec, bond_dim, mo_indices))

    # Write PES to file
    filename = 'output/' + molecule + '_PES.txt'
    g = open(filename, 'w')
    g.write('R' + ' ' + 'RHF' + ' ' + 'RKS' + ' ' + 'CCSD' + ' ' + 'CASSCF' + ' ' + 'DMRG' + '\n')
    for i in range(len(bond_lengths)):
    	g.write(str(bond_lengths[i]) + ' ' + str(energy_RHF[i]) + ' ' + str(energy_RKS[i]) + ' ' + str(energy_CCSD[i]) + ' ' + str(energy_CASSCF[i]) + ' ' + str(energy_DMRG[i]) + '\n')
    g.close()

    # Write last energies (dissociation) to file
    filename = 'output/energy_at_max_separation_' + molecule + '.txt'
    f = open(filename, 'w')
    f.write('RHF' + ' ' + 'RKS' + ' ' + 'CCSD' + ' ' + 'CASSCF' + ' ' + 'DMRG' + '\n')
    f.write(str(energy_RHF[-1]) + ' ' + str(energy_RKS[-1]) + ' ' + str(energy_CCSD[-1]) + ' ' + str(energy_CASSCF[-1]) + ' ' + str(energy_DMRG[-1]) + '\n')
    f.close()


# Compute PES curves with varying bond dim (only CASSCF and DMRG-CASSCF)
def bond_dim_PES(bond_lengths, nactorb, nactelec, basis, bond_dims, molecule, mo_indices):
    energy_CASSCF = np.zeros(len(bond_lengths))
    energy_DMRG = np.zeros((len(bond_lengths), len(bond_dims)))
    time_DMRG = np.zeros((len(bond_lengths), len(bond_dims)))
    time_CASSCF = np.zeros(len(bond_lengths))
    
    for i, R in enumerate(bond_lengths):
    	if molecule == 'N2':
    		mol = initialize_N2(R, basis)
    	elif molecule == 'BeH2':
    		mol = initialize_BeH2(R, basis)
    	else:
    		print('"molecule" parameter must be string: "N2" or "BeH2".')
    		break

    	mf = scf.RHF(mol).run()

    	# CASSCF
    	tic1 = time()
    	energy_CASSCF[i] = CASSCF_calc(mf, nactorb, nactelec, mo_indices)
    	time_CASSCF[i] = time() - tic1

    	# DMRG-CASSCF for all bond dimensions
    	for j, D in enumerate(bond_dims):
    		print('D:', D)
    		tic2 = time()
    		energy_DMRG[i, j] = DMRG_calc(mf, nactorb, nactelec, D, mo_indices)
    		time_DMRG[i, j] = time() - tic2

    # Write to file
    filename = 'output/' + molecule + 'PES_bond_dim.npz'
    np.savez_compressed(filename,
                       bond_lengths=bond_lengths,
                       bond_dims=bond_dims,
                       energy_CASSCF=energy_CASSCF,
                       energy_DMRG=energy_DMRG, 
                       time_CASSCF = time_CASSCF, 
                       time_DMRG = time_DMRG)


# Compute energies at equlibrium distance (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF)
def eq_energies(basis, nactorb, nactelec, bond_dim, molecule, mo_indices):
	if molecule == 'N2':
		mol = initialize_N2(1.1, basis)
	elif molecule == 'BeH2':
		mol = initialize_BeH2(1.3, basis)

	# Optimize geometry
	mf = RKS_calc(mol)
	mol_eq = optimize(mf)

	# Solve RHF
	mf = scf.RHF(mol_eq)
	mf.kernel()
	RHF_eq = mf.e_tot

	# RKS
	RKS = RKS_calc(mol_eq)
	RKS.kernel()
	RKS_eq = RKS.e_tot

	# CCSD
	CCSD_eq = CCSD_calc(mf)

	# CASSCF
	mycas = mcscf.CASSCF(mf, nactorb, nactelec)
	mo = mycas.sort_mo(mo_indices)
	mycas.kernel(mo)
	CASSCF_eq = mycas.e_tot 

	# DMRG-CASSCF
	mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim)
	mo = mc.sort_mo(mo_indices)
	mc.kernel(mo)
	DMRG_eq = mc.e_tot

	#MRPT (NEVPT2)
	nevpt_e = mrpt.NEVPT(mc).kernel()

	# Write to file
	filename = 'output/eq_energies_' + molecule + '.txt'
	f = open(filename, 'w')
	f.write('RHF' + ' ' + 'RKS' + ' ' + 'CCSD' + ' ' + 'CASSCF' + ' ' + 'DMRG' + ' ' + 'NEVPT2' + '\n')
	f.write(str(RHF_eq) + ' ' + str(RKS_eq) + ' ' + str(CCSD_eq) + ' ' + str(CASSCF_eq) + ' ' + str(DMRG_eq) + ' ' + str(nevpt_e + DMRG_eq) + '\n')


# Compute isolated atom energies (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF)
# For nitrogen only
def isolated_atoms_energies(basis, bond_dim, nactorb = 4, nactelec = 5):
	mol_N = gto.M(atom = 'N 0 0 0', basis = basis, spin = 3)

	# RHF
	mf = scf.RHF(mol_N)
	mf.kernel()
	RHF_N = mf.e_tot * 2

	# RKS
	mf_RKS = scf.RKS(mol_N)
	mf_RKS.xc = 'b3lyp'
	mf_RKS.kernel()
	RKS_N = mf_RKS.e_tot * 2

	# CCSD
	mycc = cc.CCSD(mf).run()
	CCSD_N = mycc.e_tot * 2

	# CASSCF
	mycas = mcscf.CASSCF(mf, nactorb, nactelec).run()
	CASSCF_N = mycas.e_tot * 2 

	# DMRG-CASSCF
	mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim).run()
	DMRG_N = mc.e_tot * 2

	#MRPT (NEVPT2)
	nevpt_e = mrpt.NEVPT(mc).kernel() * 2

	#FCI
	cisolver = fci.FCI(mf)
	FCI_N = cisolver.kernel()[0] * 2

	# Write to file
	f = open('output/isolated_atom_energies_2N.txt', 'w')
	f.write('RHF' + ' ' + 'RKS' + ' ' + 'CCSD' + ' ' + 'CASSCF' + ' ' + 'DMRG' + ' ' + 'NEVPT2' + ' ' + 'FCI' + '\n')
	f.write(str(RHF_N) + ' ' + str(RKS_N) + ' ' + str(CCSD_N) + ' ' + str(CASSCF_N) + ' ' + str(DMRG_N) + ' ' + str(nevpt_e + DMRG_N) + ' ' + str(FCI_N) + '\n')


# PES using CASSCF and DMRG-CASSCF with increasing number of H-atoms
def N_increase(basis, bond_dim, shape, natoms, bond_lengths):
	energy_CASSCF = np.zeros((len(natoms), len(bond_lengths)))
	energy_DMRG = np.zeros((len(natoms), len(bond_lengths)))
	time_DMRG = np.zeros((len(natoms), len(bond_lengths)))
	time_CASSCF = np.zeros((len(natoms), len(bond_lengths)))

	for i, R in enumerate(bond_lengths):
		print('R:', R)
		for j, N in enumerate(natoms):
			print('N:', N)
			if shape == 'Hchain':
				mol = initialize_H_chain(R, basis, N)
			elif shape == 'Hcircle':
				mol = initialize_H_circle(R, basis, N)

			# RHF
			mf = scf.RHF(mol).run()

			nactorb = nactelec = N

			# CASSCF
			tic1 = time()
			mycas = mcscf.CASSCF(mf, nactorb, nactelec).run()
			energy_CASSCF[j, i] = mycas.e_tot # N * R array
			time_CASSCF[j, i] = time() - tic1
			print('time CASSCF:', time() - tic1)

			# DMRG
			tic2 = time()
			mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim).run()
			energy_DMRG[j, i] = mc.e_tot
			time_DMRG[j, i] = time() - tic2
			print('time DMRG:', time() - tic2)

	# Write to file
	filename = 'output/N_increase_PES_' + shape + '.npz'

	np.savez_compressed(filename,
                       bond_lengths=bond_lengths,
                       natoms = natoms,
                       energy_CASSCF=energy_CASSCF,
                       energy_DMRG=energy_DMRG, 
                       time_CASSCF = time_CASSCF, 
                       time_DMRG = time_DMRG)

# PES using only DMRG-CASSCF with increasing number of H-atoms
def N_increase_DMRG(basis, bond_dim, shape, natoms, bond_lengths):
	energy_DMRG = np.zeros((len(natoms), len(bond_lengths)))
	time_DMRG = np.zeros((len(natoms), len(bond_lengths)))

	for i, R in enumerate(bond_lengths):
		print('R:', R)
		for j, N in enumerate(natoms):
			print('N:', N)
			if shape == 'Hchain':
				mol = initialize_H_chain(R, basis, N)
			elif shape == 'Hcircle':
				mol = initialize_H_circle(R, basis, N)

			# RHF
			mf = scf.RHF(mol).run()

			nactorb = nactelec = N

			# DMRG
			tic2 = time()
			mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = bond_dim).run()
			energy_DMRG[j, i] = mc.e_tot
			time_DMRG[j, i] = time() - tic2
			print('time DMRG:', time() - tic2)

	# Write PES to file
	filename = 'output/N_increase_PES_DMRG' + shape + '.npz'

	np.savez_compressed(filename,
                       bond_lengths=bond_lengths,
                       natoms = natoms,
                       energy_DMRG=energy_DMRG,  
                       time_DMRG = time_DMRG)

# Solve DMRG-CASSCF for N H-atoms with different bond dimensions
def D_increase_N_H(basis, bond_dims, shape, natoms):
	energy_DMRG = np.zeros((len(natoms), len(bond_dims)))
	time_DMRG = np.zeros((len(natoms), len(bond_dims)))
	energy_CASSCF = np.zeros(len(natoms))
	time_CASSCF = np.zeros(len(natoms))

	for i, N in enumerate(natoms):
		print('N:', N)
		if shape == 'Hchain':
			mol = initialize_H_chain(0.74, basis, N)
		elif shape == 'Hcircle':
			mol = initialize_H_circle(0.74, basis, N)

		# Optimize geometry
		mf = RKS_calc(mol)
		mol_eq = optimize(mf)

		# Solve RHF
		mf = scf.RHF(mol_eq).run()

		nactorb = nactelec = N

		# CASSCF
		tic1 = time()
		energy_CASSCF[i] = CASSCF_unspecified_orbs(mf, nactorb, nactelec)
		time_CASSCF[i] = time() - tic1

		# DMRG-CASSCF
		for j, D in enumerate(bond_dims):
			print('D:', D)
			tic2 = time()
			mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM = D, tol = 1e-10).run()
			energy_DMRG[i, j] = mc.e_tot
			time_DMRG[i, j] = time() - tic2

	# Write to file
	filename = 'output/' + shape + '_N_H_bond_dim.npz'
	np.savez_compressed(filename,
                       bond_dims=bond_dims,
                       energy_DMRG=energy_DMRG, # len(natoms) * len(bond_dims)
                       energy_CASSCF = energy_CASSCF, # len(natoms)
                       time_CASSCF = time_CASSCF, # len(natoms)
                       time_DMRG = time_DMRG, # len(natoms) * len(bond_dims)
                       natoms = natoms)

def get_natural_occupation_numbers(basis, bond_lengths, molecule, N):
	f = open('output/occupation_numbers' + molecule + '.txt', 'w')
	for i, R in enumerate(bond_lengths):
		if molecule == 'N2':
			mol = initialize_N2(R, basis)
		elif molecule == 'BeH2':
			mol = initialize_BeH2(R, basis)
		elif molecule == 'Hchain':
			mol = initialize_H_chain(R, basis, N)

		mf = scf.RHF(mol).run()
		mc = mcscf.CASSCF(mf, N, N).run()

		dm1 = mc.make_rdm1()
		occ_numbers, _ = np.linalg.eigh(dm1)
		occ_numbers = occ_numbers[::-1]

		f.write(f'Geometry {i}: {R}\n')
		f.write('Occupations:\n')
		f.write(' '.join(f'{n:.6f}' for n in occ_numbers) + '\n\n')
	f.close()
