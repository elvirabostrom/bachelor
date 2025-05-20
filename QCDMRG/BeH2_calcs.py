from utils import *

"""
Stretching bonds in BeH2

Uncomment to run
"""

# Settings
basis = 'ccpvdz'
R = np.linspace(0, 4, 100)
bond_dim = 20

molecule = 'BeH2'

# Inspect orbital energies and occupations, prints energy and occupation number of all MO's in terminal
# Used for determining active mo indices
inspect_orbs(molecule, basis, 1.4)

# Set active space
nactorb = nactelec = 2
mo_indices = [3, 4] # OBS: sort_mo by default take the 1-based orbital indices!!

# Compute PES using different methods (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF) and write to file 'output/BeH2_PES.txt'
# Simulataneously writes to a separate file 'output/energy_at_max_separation_BeH2.txt' the energies at maximal chosen separation
compute_PES(basis, nactorb, nactelec, bond_dim, R, molecule, mo_indices)

# # Compute PES with varying bond dim (only CASSCF and DMRG-CASSCF) and write to file 'output/BeH2PES_bond_dim.npz'
# bond_dims = np.array([2, 5, 10, 20])
# bond_dim_PES(R, nactorb, nactelec, basis, bond_dims, molecule, mo_indices)