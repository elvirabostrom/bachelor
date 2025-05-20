import matplotlib.pyplot as plt 
import numpy as np 

"""
Active space expansion for equilibrium N2

"""

AS = np.loadtxt('output/expand_AS.txt', skiprows = 1)[:, 0]
CASSCF = np.loadtxt('output/expand_AS.txt', skiprows = 1)[:, 1]
DMRG = np.loadtxt('output/expand_AS.txt', skiprows = 1)[:, 2]
t_casscf = np.loadtxt('output/expand_AS.txt', skiprows = 1)[:, 3]
t_dmrg = np.loadtxt('output/expand_AS.txt', skiprows = 1)[:, 4]

error =  np.abs(CASSCF - DMRG)

# time
plt.figure(figsize = (3.1, 2.4))
plt.plot(AS, t_dmrg, label = 'DMRG-CASSCF', color = 'b', linewidth = 0.9)
plt.plot(AS, t_casscf, label = 'CASSCF', color = 'r', linewidth = 0.9)
plt.xlabel('Active space size (orbitals)')
plt.ylabel('CPU-time (s)')
plt.legend(frameon = False)
plt.xlim(6, 14)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/AS_exp_t.pdf', bbox_inches='tight')
plt.show()

# energy
plt.figure(figsize = (3.2, 2.4))
plt.plot(AS, DMRG, label = 'DMRG-CASSCF', color = 'b', linewidth = 0.9)
plt.plot(AS, DMRG, label = 'CASSCF', color = 'r', linewidth = 0.9, linestyle = '--')
plt.xlabel('Active space size (orbitals)')
plt.ylabel('Energy (Hartree)')
plt.grid(True)
plt.legend(frameon = False)
plt.xlim(6, 14)
plt.tight_layout()
plt.savefig('output/AS_exp_E.pdf', bbox_inches='tight')
plt.show()


# error
plt.figure(figsize = (3.1, 2.5))
plt.plot(AS, error, color = 'grey', linewidth = 0.9)
plt.xlabel('Active space size (orbitals)')
plt.ylabel('Energy error (Hartree)')
plt.legend(frameon = False)
plt.grid(True)
plt.xlim(6, 14)
plt.tight_layout()
plt.savefig('output/AS_exp_E_error.pdf', bbox_inches='tight')
plt.show()