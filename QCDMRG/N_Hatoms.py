from utils import *


"""
Chain of H-atoms with a distance R between them (can choose circle shape)

Uncomment to run
"""

# Settings
basis = 'ccpvdz'
R = np.linspace(0.74, 5, 100)

molecule = 'Hchain' # either 'Hchain' or 'Hcircle'

# Compute PES using different methods (RHF, RKS, CCSD, CASSCF, DMRG-CASSCF) and write to file 'output/Hchain_PES.txt'
# Simulataneously writes to a separate file 'output/energy_at_max_separation_Hchain.txt' the energies at maximal chosen separation
bond_dim = 200
nactorb = nactelec = 6
compute_PES(basis, nactorb, nactelec, bond_dim, R, molecule, N = 6)

# N increase over PES using CASSCF and DMRG-CASSCF, output file 'output/N_increase_PES_Hchain.npz'
natoms = np.arange(2, 14, 2)
bond_dim = 200
N_increase(basis, bond_dim, molecule, natoms, R)

# N increase over PES using DMRG-CASSCF only, output file 'output/N_increase_PES_DMRGHchain.npz'
natoms = np.arange(14, 20, 2)
R = np.array([0.74, 1.5, 2.0, 3]) # some chosen R values
bond_dim = 500
N_increase_DMRG(basis, bond_dim, molecule, natoms, R)

# Increasing D at equilibrium geometry, active spaces (10,10) and (12,12), output file 'output/Hchain_N_H_bond_dim.npz'
natoms = [10, 12]
bond_dims = [50, 100, 200, 300, 500, 700, 1000]
D_increase_N_H(basis, bond_dims, molecule, natoms)

# Get occupation numbers from CASSCF 1RDM at chosen bond lengths, output file 'output/occupation_numbersHchain.txt'
R = np.array([0.74, 1.5, 2.0, 3, 5])
N = 6
get_natural_occupation_numbers(basis, R, molecule, N)


