import matplotlib.pyplot as plt 
import numpy as np 

"""
Beryllium dihydride
"""


#
# PES
#

R = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 0]
RHF = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 1]
RKS = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 2]
CCSD = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 3]
CASSCF = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 4]
DMRG = np.loadtxt('output/BeH2_PES.txt', skiprows = 1)[:, 5]

plt.figure(figsize = (5,4))
markers_on = [10, 20, 30, 40, 50, 60, 70, 80, 90]
plt.plot(R, RHF, label = 'RHF', color = 'grey', marker = 'o', markevery=markers_on, linewidth=0.7)
plt.plot(R, RKS, label = 'RKS', color = 'grey', marker = 's', markevery=markers_on, linewidth=0.7)
plt.plot(R, CCSD, label = 'CCSD', color = 'grey', marker = 'v', markevery=markers_on, linewidth=0.7)
plt.plot(R, CASSCF, label = 'CASSCF', color = 'r', linewidth=0.7)
plt.plot(R, DMRG, label = 'DMRG-CASSCF', color = 'b', linewidth=0.7, linestyle = '--')
plt.xlabel('x (Å)')
plt.ylabel('Energy (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.grid(True)
plt.xlim(0, 4)
plt.tight_layout()
plt.savefig('output/PES_BeH2.pdf', bbox_inches='tight')
plt.show()


#
# Scan of DMRG-CASSCF and CASSCF
#

R_scan = R[40 : 80]
DMRG_scan = DMRG[40 : 80]
CASSCF_scan = CASSCF[40 : 80]

plt.figure(figsize = (3.5,2.5))
plt.plot(R_scan, CASSCF_scan, label = 'CASSCF', color = 'r', linewidth=0.7)
plt.plot(R_scan, DMRG_scan, label = 'DMRG-CASSCF', color = 'b', linewidth=0.7)
plt.xlabel('x (Å)')
plt.ylabel('Energy (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 2, frameon = False)
plt.grid(True)
plt.xlim(1.70, 3.2)
plt.tight_layout()
plt.savefig('output/PES_BeH2_SCAN.pdf', bbox_inches='tight')
plt.show()


#
# Bond dim PES
#

data = np.load('output/BeH2PES_bond_dim.npz')

CASSCF = data['energy_CASSCF']
angles = data['bond_lengths']
D = data['bond_dims']
DMRG = data['energy_DMRG'] 
t_CASSCF = data['time_CASSCF']
t_DMRG = data['time_DMRG']


#D lines
errors = np.array([np.log10(np.abs(CASSCF - DMRG[:,i]) + 1e-10) for i in range(len(D))])
colors = ['darkturquoise', 'firebrick', 'yellowgreen', 'orange', 'violet', 'darkslategrey']
plt.figure(figsize=(5, 4))
for i, d in enumerate(D):
	plt.plot(angles, errors[i], label=f'D={d}', color = colors[i % len(colors)], linewidth = 0.7)
plt.xlabel('x (Å)')
plt.ylabel('Energy error (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 2, frameon = False)
plt.xlim(0, 4)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/Dlines_CASSCF_DMRG_error_BeH2.pdf', bbox_inches='tight')
plt.show()
