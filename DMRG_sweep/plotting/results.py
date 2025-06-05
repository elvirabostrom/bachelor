import matplotlib.pyplot as plt 
import numpy as np 

#
# ALS
#

ALS_2_N = np.loadtxt('output/ALS_N_t_for_bond_dim_2.txt')[:, 1]
ALS_2_t = np.loadtxt('output/ALS_N_t_for_bond_dim_2.txt')[:, 0]
ALS_2_E = np.loadtxt('output/ALS_N_t_for_bond_dim_2.txt')[:, 2]
ALS_3_N = np.loadtxt('output/ALS_N_t_for_bond_dim_3.txt')[:, 1]
ALS_3_t = np.loadtxt('output/ALS_N_t_for_bond_dim_3.txt')[:, 0]
ALS_3_E = np.loadtxt('output/ALS_N_t_for_bond_dim_3.txt')[:, 2]
ALS_4_N = np.loadtxt('output/ALS_N_t_for_bond_dim_4.txt')[:, 1]
ALS_4_t = np.loadtxt('output/ALS_N_t_for_bond_dim_4.txt')[:, 0]
ALS_4_E = np.loadtxt('output/ALS_N_t_for_bond_dim_4.txt')[:, 2]
ALS_5_N = np.loadtxt('output/ALS_N_t_for_bond_dim_5.txt')[:, 1]
ALS_5_t = np.loadtxt('output/ALS_N_t_for_bond_dim_5.txt')[:, 0]
ALS_5_E = np.loadtxt('output/ALS_N_t_for_bond_dim_5.txt')[:, 2]


#
# power method
#

pm_2_N = np.loadtxt('output/pm_N_t_for_bond_dim_2.txt')[:, 1]
pm_2_t = np.loadtxt('output/pm_N_t_for_bond_dim_2.txt')[:, 0]
pm_2_E = np.loadtxt('output/pm_N_t_for_bond_dim_2.txt')[:, 2]
pm_3_N = np.loadtxt('output/pm_N_t_for_bond_dim_3.txt')[:, 1]
pm_3_t = np.loadtxt('output/pm_N_t_for_bond_dim_3.txt')[:, 0]
pm_3_E = np.loadtxt('output/pm_N_t_for_bond_dim_3.txt')[:, 2]
pm_4_N = np.loadtxt('output/pm_N_t_for_bond_dim_4.txt')[:, 1]
pm_4_t = np.loadtxt('output/pm_N_t_for_bond_dim_4.txt')[:, 0]
pm_4_E = np.loadtxt('output/pm_N_t_for_bond_dim_4.txt')[:, 2]
pm_5_N = np.loadtxt('output/pm_N_t_for_bond_dim_5.txt')[:, 1]
pm_5_t = np.loadtxt('output/pm_N_t_for_bond_dim_5.txt')[:, 0]
pm_5_E = np.loadtxt('output/pm_N_t_for_bond_dim_5.txt')[:, 2]


#
# FDM
#

FDM_2_N = np.loadtxt('output/FDM_N_t_for_bond_dim_2.txt')[:, 1]
FDM_2_t = np.loadtxt('output/FDM_N_t_for_bond_dim_2.txt')[:, 0]
FDM_2_E = np.loadtxt('output/FDM_N_t_for_bond_dim_2.txt')[:, 2]
FDM_3_N = np.loadtxt('output/FDM_N_t_for_bond_dim_3.txt')[:, 1]
FDM_3_t = np.loadtxt('output/FDM_N_t_for_bond_dim_3.txt')[:, 0]
FDM_3_E = np.loadtxt('output/FDM_N_t_for_bond_dim_3.txt')[:, 2]
FDM_4_N = np.loadtxt('output/FDM_N_t_for_bond_dim_4.txt')[:, 1]
FDM_4_t = np.loadtxt('output/FDM_N_t_for_bond_dim_4.txt')[:, 0]
FDM_4_E = np.loadtxt('output/FDM_N_t_for_bond_dim_4.txt')[:, 2]
FDM_5_N = np.loadtxt('output/FDM_N_t_for_bond_dim_5.txt')[:, 1]
FDM_5_t = np.loadtxt('output/FDM_N_t_for_bond_dim_5.txt')[:, 0]
FDM_5_E = np.loadtxt('output/FDM_N_t_for_bond_dim_5.txt')[:, 2]


#
# PLOTS
#

color = 'grey'
# ALS: compare run time for varying D, N
plt.figure(figsize = (5,3))
plt.plot(ALS_2_N, ALS_2_t, label = 'D = 2', color = color, marker = '^', linewidth = 0.8)
plt.plot(ALS_3_N, ALS_3_t, label = 'D = 3', color = color, marker = 's', linewidth = 0.8)
plt.plot(ALS_4_N, ALS_4_t, label = 'D = 4', color = color, marker = 'v', linewidth = 0.8)
plt.plot(ALS_5_N, ALS_5_t, label = 'D = 5', color = color, marker = 'o', linewidth = 0.8)
plt.yscale('log')
plt.xlabel('Grid dimension (N)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon = False)
plt.ylabel('CPU time (s)')
plt.xlim(50, 200)
plt.ylim(0, 200)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ALS_compare_runtime_varying_D_and_N.pdf', bbox_inches='tight')
plt.show()

# ALS: compare results for varying D, N
plt.figure(figsize = (5,3))
plt.plot(ALS_2_N, ALS_2_E, label = 'D = 2', color = 'forestgreen', linewidth = 0.8)
plt.plot(ALS_3_N, ALS_3_E, label = 'D = 3', color = 'b', linewidth = 0.8)
plt.plot(ALS_4_N, ALS_4_E, label = 'D = 4', color = 'r', linewidth = 0.8)
plt.plot(ALS_5_N, ALS_5_E, label = 'D = 5', color = 'k', linewidth = 0.8)
plt.xlabel('Grid dimension (N)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon = False)
plt.ylabel(f'Energy')
plt.xlim(50, 200)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ALS_compare_E_varying_D_and_N.pdf', bbox_inches='tight')
plt.show()



# POWER METHOD: compare run time for varying D, N
plt.figure(figsize = (4,3))
plt.plot(pm_2_N, pm_2_t, label = 'D = 2', color = color, marker = '^', linewidth = 0.8)
plt.plot(pm_3_N, pm_3_t, label = 'D = 3', color = color, marker = 's', linewidth = 0.8)
plt.plot(pm_4_N, pm_4_t, label = 'D = 4', color = color, marker = 'v', linewidth = 0.8)
plt.plot(pm_5_N, pm_5_t, label = 'D = 5', color = color, marker = 'o', linewidth = 0.8)
plt.yscale('log')
plt.xlabel('Grid dimension (N)')
plt.ylabel('CPU time (s)')
plt.ylim(0, 200)
plt.xlim(50, 200)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/pm_compare_runtime_varying_D_and_N.pdf')
plt.show()

# POWER METHOD: compare results for varying D, N
plt.figure(figsize = (4,3))
plt.plot(pm_2_N, pm_2_E, label = 'D = 2', color = 'forestgreen', linewidth = 0.8)
plt.plot(pm_3_N, pm_3_E, label = 'D = 3', color = 'b', linewidth = 0.8)
plt.plot(pm_4_N, pm_4_E, label = 'D = 4', color = 'r', linewidth = 0.8)
plt.plot(pm_5_N, pm_5_E, label = 'D = 5', color = 'k', linewidth = 0.8)
plt.xlabel('Grid dimension (N)')
plt.ylabel(f'Energy')
plt.grid(True)
plt.xlim(50, 200)
plt.tight_layout()
plt.savefig('output/pm_compare_E_varying_D_and_N.pdf')
plt.show()

#scan of energies
plt.figure(figsize = (5,3))
plt.plot(ALS_3_N[1:], ALS_3_E[1:], label = 'D = 3', color = color, marker = 's', linewidth = 0.8)
plt.plot(ALS_4_N[1:], ALS_4_E[1:], label = 'D = 4', color = color, marker = 'v', linewidth = 0.8)
plt.plot(ALS_5_N[1:], ALS_5_E[1:], label = 'D = 5', color = color, marker = 'o', linewidth = 0.8)
plt.xlabel('Grid dimension (N)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon = False)
plt.grid(True)
plt.ylabel(f'Energy')
plt.xlim(100, 200)
plt.tight_layout()
plt.savefig('output/ALS_compare_E_scan.pdf')
plt.show()

plt.figure(figsize = (4,3))
plt.plot(pm_3_N[1:], pm_3_E[1:], label = 'D = 3', color = color, marker = 's', linewidth = 0.8)
plt.plot(pm_4_N[1:], pm_4_E[1:], label = 'D = 4', color = color, marker = 'v', linewidth = 0.8)
plt.plot(pm_5_N[1:], pm_5_E[1:], label = 'D = 5', color = color, marker = 'o', linewidth = 0.8)
plt.xlabel('Grid dimension (N)')
plt.ylabel(f'Energy')
plt.xlim(100, 200)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/pm_compare_E_scan.pdf')
plt.show()


# FDM VS ALS
energy_diff_2 = np.abs(np.array(FDM_2_E) - np.array(ALS_2_E))
energy_diff_3 = np.abs(np.array(FDM_3_E) - np.array(ALS_3_E))
energy_diff_4 = np.abs(np.array(FDM_4_E) - np.array(ALS_4_E))
energy_diff_5 = np.abs(np.array(FDM_5_E) - np.array(ALS_5_E))
plt.figure(figsize = (4,3))
#plt.plot(FDM_2_N, energy_diff_2, label = 'D = 2')
plt.plot(FDM_3_N, energy_diff_3, label = 'D = 3', color = color, marker = 's', linewidth = 0.8)
plt.plot(FDM_4_N, energy_diff_4, label = 'D = 4', color = color, marker = 'v', linewidth = 0.8)
plt.plot(FDM_5_N, energy_diff_5, label = 'D = 5', color = color, marker = 'o', linewidth = 0.8)
plt.xlabel('Grid dimension (N)')
plt.ylabel('Energy error')
plt.yscale('log')
plt.xlim(50, 200)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()
plt.savefig('output/FDM_vs_ALS.pdf')
plt.show()


# time increasement factor with D 
time_diff1_ALS = np.abs(np.array(ALS_3_t) / np.array(ALS_2_t))
time_diff2_ALS = np.abs(np.array(ALS_4_t) / np.array(ALS_3_t))
time_diff3_ALS = np.abs(np.array(ALS_5_t) / np.array(ALS_4_t))
print(time_diff1_ALS)
print(time_diff2_ALS)
print(time_diff3_ALS)
time_diff1_pm = np.abs(np.array(pm_3_t) / np.array(pm_2_t))
time_diff2_pm = np.abs(np.array(pm_4_t) / np.array(pm_3_t))
time_diff3_pm = np.abs(np.array(pm_5_t) / np.array(pm_4_t))
print(time_diff1_pm)
print(time_diff2_pm)
print(time_diff3_pm)


