from utils import *

"""
Stretching bond in N2

Uncomment to run
"""

# Settings 
basis = 'ccpvdz'
bond_dim = 100
R = np.linspace(0.8, 2.9, 100)

molecule = 'N2'

# Inspect orbital energies and occupations, prints energy and occupation number of all MO's in terminal
inspect_orbs(molecule, basis, 1.1)

# Set active space
nactorb = nactelec = 6
mo_indices = [5, 6, 7, 8, 9, 10] # interpret inspect_orbs() output and set manually. OBS: sort_mo used for DMRG-CASSCF and CASSCF by default take the 1-based orbital indices!!

# Compute PES using different methods (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF) and write to file 'output/N2_PES.txt'
# Simulataneously writes to a separate file 'output/energy_at_max_separation_N2.txt' the energies at maximal chosen separation
compute_PES(basis, nactorb, nactelec, bond_dim, R, molecule, mo_indices)

# Compute PES curves with varying bond dim (only CASSCF and DMRG-CASSCF) and write to file 'output/N2PES_bond_dim.npz'
bond_dims = np.array([10, 20, 30, 40, 50, 60])
bond_dim_PES(R, nactorb, nactelec, basis, bond_dims, molecule, mo_indices)

# Compute dissociation energy data (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF, NEVPT2) and write to file
eq_energies(basis, nactorb, nactelec, bond_dim, molecule, mo_indices) # Energy of molecule at equilibrium 'output/eq_energies_N2.txt'
isolated_atoms_energies(basis, bond_dim) # 2 x Energy of isolated atom, also solves FCI, 'output/isolated_atom_energies_2N.txt'