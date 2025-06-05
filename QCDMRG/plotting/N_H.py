import matplotlib.pyplot as plt 
import numpy as np 

"""
Chain/circle of N H-atoms

"""


#
# PES all methods
#

# choose shape
shape = 'Hchain'

filename = 'output/' + shape + '_PES.txt'

R = np.loadtxt(filename, skiprows = 1)[:, 0]
RHF = np.loadtxt(filename, skiprows = 1)[:, 1]
RKS = np.loadtxt(filename, skiprows = 1)[:, 2]
CCSD = np.loadtxt(filename, skiprows = 1)[:, 3]
CASSCF = np.loadtxt(filename, skiprows = 1)[:, 4]
DMRG = np.loadtxt(filename, skiprows = 1)[:, 5]

plt.figure(figsize = (5,4))
markers_on = [10, 20, 30, 40]
plt.plot(R, RHF, label = 'RHF', color = 'grey', marker = 'o', markevery=markers_on, linewidth=0.7)
plt.plot(R, RKS, label = 'RKS', color = 'grey', marker = 's', markevery=markers_on, linewidth=0.7)
plt.plot(R, CCSD, label = 'CCSD', color = 'grey', marker = 'v', markevery=markers_on, linewidth=0.7)
plt.plot(R, CASSCF, label = 'CASSCF', color = 'r', linewidth=0.7)
plt.plot(R, DMRG, label = 'DMRG-CASSCF', color = 'b', linewidth=0.7)
plt.xlabel('Distance Between atoms (Å)')
plt.ylabel('Energy (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.grid(True)
plt.xlim(0.74, 3)
plt.tight_layout()
plt.savefig('output/N_H_PES_' + shape + '.pdf', bbox_inches='tight')
plt.show()



#
# Increase N
#

filename = 'output/N_increase_PES_' + shape + '.npz'
data = np.load(filename)

CASSCF = data['energy_CASSCF']
R = data['bond_lengths']
N = data['natoms'] 
DMRG = data['energy_DMRG'] # N * R 
t_CASSCF = data['time_CASSCF']
t_DMRG = data['time_DMRG']


# R, error for different N
errors = np.log10(np.abs(CASSCF - DMRG) + 1e-10)
colorse = ['darkturquoise', 'firebrick', 'yellowgreen', 'orange', 'violet', 'darkslategrey']

plt.figure(figsize=(4, 3.5))
for i, d in enumerate(N):
	plt.plot(R, errors[i, :], label=f'N={d}', color = colorse[i % len(colorse)], linewidth = 0.7)
plt.xlabel('Bond length (Å)')
plt.ylabel('log$_{10}$ Energy error (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.xlim(0.74, 5)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ERROR_N_H_' + shape + '.pdf', bbox_inches='tight')
plt.show()


# N, t (for selected R)
selected_indices = [0, int(len(R) - 80), len(R) - 1]
color1 = ['b', 'r', 'k']
markers = ['o', 's', '^']

plt.figure(figsize=(4, 3.5))
for i, idx in enumerate(selected_indices): 
	if i==1:
		markers_on = np.array([1, 3, 5])
	else:
		markers_on = [0, 2, 4]
	r_val = R[idx]
	print(r_val)
	plt.plot(N, np.log10(t_DMRG[:, idx]), label=f'R={r_val:.2f}, DMRG', color = color1[i % len(color1)], linestyle = '--', linewidth=0.7)
	plt.plot(N, np.log10(t_CASSCF[:, idx]), label=f'R={r_val:.2f}, CASSCF', color = color1[i % len(color1)], linewidth=0.7)
plt.xlabel('System size (N)')
plt.ylabel('log$_{10}$ CPU-time (s)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 2, frameon = False)
plt.xlim(2, 12)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/CPU_time_N_H_' + shape + '.pdf', bbox_inches='tight')
plt.show()


# PES curves for N >= n 
colors1 = ['skyblue', 'b', 'navy', 'aquamarine', 'lightseagreen', 'darkcyan']
colors2 = ['coral', 'r', 'firebrick', 'gold', 'orange', 'peru']
def plot_PES(n):
	plt.figure(figsize = (6,4))
	color_idx = 0
	for i in range(n, N[-1] + 1, 2):
		idx = np.argmin(np.abs(N - i))
		plt.plot(R, DMRG[idx, :], label=f'N={i}, DMRG', color=colors1[color_idx], linewidth=0.7)
		plt.plot(R, CASSCF[idx, :], label=f'N={i}, CASSCF', color=colors2[color_idx], linewidth=0.7)
		color_idx += 1
	plt.xlabel('Distance Between atoms (Å)')
	plt.ylabel('Energy (Hartree)')
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
	plt.grid(True)
	plt.xlim(0.74, 5)
	plt.tight_layout()
	plt.savefig('output/N_H_PES_' + shape + '_' + str(N) + '.pdf', bbox_inches='tight')
	plt.show()

plot_PES(8)


#
# Higher N for DMRG only
#

shape = 'Hchain'
filename = 'output/N_increase_PES_DMRG' + shape + '.npz'
data = np.load(filename)
R = data['bond_lengths']
N = data['natoms'] 
DMRG = data['energy_DMRG'] # N * R 
t_DMRG = data['time_DMRG']

for i, n in enumerate(N):
	print('N:',n, np.log10(t_DMRG[i, :]))

"""
N: 14 [3.18421295 2.91935759 2.91970784 2.96395618]
N: 16 [3.30158951 3.06829012 3.32736413 3.10408128]
N: 18 [3.54124877 3.42765765 3.53588293 4.30988723]
"""


#
# increasing D for (10,10) and (12,12) AS (eq distance)
#

filename = 'output/' + shape + '_N_H_bond_dim.npz'
data = np.load(filename)
D = data['bond_dims']
N = data['natoms']
DMRG = data['energy_DMRG'] # len(N) * len(D)
t_DMRG = data['time_DMRG'] # len(N) * len(D)
CASSCF = data['energy_CASSCF'] # len(N) 
t_CASSCF = data['time_CASSCF'] # len(N)

error_N10 = np.log10(np.abs([CASSCF[0] - DMRG[0, i] for i in range(len(D))]))
error_N12 = np.log10(np.abs([CASSCF[1] - DMRG[1, i] for i in range(len(D))]))

plt.figure(figsize = (4,3))
plt.plot(D, error_N10, label ='N=10', linewidth=0.7, color = 'b', marker='o')
plt.plot(D, error_N12, label ='N=12', linewidth=0.7, color = 'b', marker='s')
plt.xlabel('Bond dimension (D)')
plt.ylabel('log$_{10}$ Energy error (Hartree)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 2, frameon = False)
plt.xlim(50, 1000)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/' + shape + '_N_H_bond_dim_error.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize = (4,3.5))
#plt.plot(D, np.log10(np.ones(len(D)) * t_CASSCF[0]), label='N=10, CASSCF', linewidth=0.7, color = 'r')
plt.plot(D, np.log10(np.ones(len(D)) * t_CASSCF[1]), label='N=12, CASSCF', linewidth=0.7, color = 'r')
plt.plot(D, np.log10(t_DMRG[0,:]), label='N=10, DMRG-CASSCF', linewidth=0.7, color = 'b', marker='o')
plt.plot(D, np.log10(t_DMRG[1,:]), label='N=12, DMRG-CASSCF', linewidth=0.7, color = 'b', marker='s')
plt.xlabel('Bond dimension (D)')
plt.ylabel('log$_{10}$ CPU-time (s)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 1, frameon = False)
plt.xlim(50, 1000)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/' + shape + '_N_H_bond_dim_time.pdf', bbox_inches='tight')
plt.show()
