# QC-DMRG computations on systems of high multi-reference character

## Description
This project compares DMRG-CASSCF with other electronic structure methods to demonstrate its strengths in large strongly correlated systems. The project is divided into four systems of interest which can be run independently: the nitrogen molecule, active space expansion, beryllium hydride and N H-atoms in a chain. 

## Prerequisites
PySCF, the pyscf extension dmrgscf and block2 are required for this project. Installation can be done by following instructions provided here: https://block2.readthedocs.io/en/latest/user/dmrg-scf.html. NumPy and Matplotlib are also required. 

## Running project
In each of the following files, different data-gathering simulations are listed and can be run one at a time. 
- `N2_calcs.py`: Compute PES and key energetic data in dissociation. Electronic structure methods applied are RHF, RKS, CCSD, CASSCF, DMRG-CASSCF, FCI, NEVPT2. 
- `AS_exp.py`: Compute equilibrium energies for nitrogen molecule in expanding active space. Electronic structure methods applied are CASSCF and DMRG-CASSCF. 
- `BeH2_calcs.py`: Compute PES. Electronic structure methods applied are RHF, RKS, CCSD, CASSCF, DMRG-CASSCF. 
- `N_Hatoms.py`: Compute PES with expanding active space (N,N) and growing chain, increase D at equilibrium geometry in active spaces (10,10) and (12,12). Electronic structure methods applied are RHF, RKS, CCSD, CASSCF, DMRG-CASSCF. 

## Other
- `utils.py`: Contains most of the code for all simulations. 

## Plotting
- `N2.py`: Plot results from nitrogen molecule computations.
- `N2_AS.py`: Plot results from active space computations.
- `BeH2.py`: Plot results from beryllium hydride computations.
- `N_H.py`: Plot results from N H-atoms chain computations.
