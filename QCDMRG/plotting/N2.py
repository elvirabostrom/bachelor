import matplotlib.pyplot as plt 
import numpy as np 

"""
Nitrogen dimer
"""

#
# PES
#

R = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 0]
RHF = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 1]
RKS = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 2]
CCSD = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 3]
CASSCF = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 4]
DMRG = np.loadtxt('output/N2_PES.txt', skiprows = 1)[:, 5]

plt.figure(figsize = (5,4))
markers_on = [10, 20, 30, 40, 50, 60, 70, 80, 90]
plt.plot(R, RHF, label = 'RHF', color = 'grey', marker = 'o', markevery=markers_on, linewidth=0.7)
plt.plot(R, RKS, label = 'RKS', color = 'grey', marker = 's', markevery=markers_on, linewidth=0.7)
plt.plot(R, CCSD, label = 'CCSD', color = 'grey', marker = 'v', markevery=markers_on, linewidth=0.7)
plt.plot(R, CASSCF, label = 'CASSCF', color = 'r', linewidth=0.8)
plt.plot(R, DMRG, label = 'DMRG-CASSCF', color = 'b', linewidth=0.8, linestyle = '--')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.grid(True)
plt.xlim(0.8, 2.9)
plt.tight_layout()
plt.savefig('output/PES_N2_test_phase.pdf', bbox_inches='tight')
plt.show()



#
# bond dim PES
#

data = np.load('output/N2PES_bond_dim.npz')

CASSCF = data['energy_CASSCF']
R = data['bond_lengths']
D = data['bond_dims']
DMRG = data['energy_DMRG'] # 50 * 6 array

errors = np.array([np.log10(np.abs(CASSCF - DMRG[:, i])) for i in range(len(D))])

#colormap
plt.figure(figsize=(5, 4))
plt.imshow(np.log10(np.abs(CASSCF[:, np.newaxis] - DMRG)).T, aspect='auto', 
           extent=[R.min(), R.max(), D.min(), D.max()],
           origin='lower', cmap='GnBu')
plt.colorbar(label='Energy error (Hartree)')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Bond Dimension (D)')
plt.xlim(0.8, 2.9)
plt.tight_layout()
plt.savefig('output/colormap_CASSCF_DMRG_error_N2.pdf', bbox_inches='tight')
plt.show()

#D lines
colors = ['darkturquoise', 'firebrick', 'yellowgreen', 'orange', 'violet', 'darkslategrey']
plt.figure(figsize=(5, 4))
for i, d in enumerate(D):
	plt.plot(R, errors[i], label=f'D={d}', color = colors[i % len(colors)], linewidth = 0.7)
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy error (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.xlim(0.8, 2.9)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/Dlines_CASSCF_DMRG_error_N2.pdf', bbox_inches='tight')
plt.show()




#
#  Energy data
#

large_sep = np.loadtxt('output/energy_at_max_separation_N2.txt', skiprows = 1)
#print(large_sep)
"""
[-108.01449115 -108.90173877 -108.52442388 -108.77723779 -108.77723779]
"""

isolated_atoms = np.loadtxt('output/isolated_atom_energies_2N.txt', skiprows = 1)
#print(isolated_atoms)
"""
[-108.77682847 -109.17562922 -108.95859813 -108.77682847 -108.77682847 -108.92393272 -108.96023011]
"""

eq_energies = np.loadtxt('output/eq_energies_N2.txt', skiprows = 1)
#print(eq_energies)
"""
[-108.95306278 -109.53337515 -109.26764579 -109.09052853 -109.09052853 -109.24688891]
"""

diff = np.abs(isolated_atoms[:-2] - large_sep)
#print(diff)
"""
[7.62337328e-01 2.73890453e-01 4.34174248e-01 4.09312285e-04 4.09313859e-04]
"""

D_e = isolated_atoms[:-1] - eq_energies
#print(D_e)
"""
[0.1762343  0.35774592 0.30904767 0.31370005 0.31370006 0.32295619]
"""

