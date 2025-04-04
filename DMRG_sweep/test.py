import numpy as np 
from scipy import sparse
from scipy import special
import utils

# General setup
N = 100
L = 10
h = 2 * L / (N - 1)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)

# Precalculate
x_sq = 0.5 * x**2
y_sq = 0.5 * y**2
z_sq = 0.5 * z**2

# Harmonic oscillator potensial tensor train decomp
def V_HO_TT():
	V = np.zeros((N, N, N))
	for i in range(len(x)):
		for j in range(len(y)):
			for k in range(len(z)):
				V[i ,j, k] = x_sq[i] + y_sq[j] + z_sq[k]
	# np.save("output/test_harmonic.npy", V)
	TT = utils.tensor_SVD(N, V, 4)

	return TT

# Generate ground state
def HO_ground_state():
	Psi = np.zeros((N, N, N))
	psi_x = np.exp(-x_sq)
	psi_y = np.exp(-y_sq)
	psi_z = np.exp(-z_sq)

	for i in range(len(psi_x)):
		for j in range(len(psi_y)):
			for k in range(len(psi_z)):
				Psi[i ,j, k] = psi_x[i] * psi_y[j] * psi_z[k]
	#Psi_norm = Psi / np.linalg.norm(Psi)

	# np.save("output/test_harmonic_GS.npy", np.conj(Psi) * Psi)
	MPS = utils.tensor_SVD(N, Psi, 2)

	return MPS


# Initializations
V = V_HO_TT() 
MPS = HO_ground_state() 
T = utils.get_kinetic(N, h)

# test
T_MPO = utils.get_kinetic_MPO(h, N)

# Compute expectation values
expectation_value = utils.expectation_values(N, T, V, MPS)

print(f'<E> = {expectation_value}')


# Solverrs
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.evp as evp


V = TT(utils.MPS_to_MPO(V))
T = utils.get_kinetic_MPO(h, N)
H = V + T

MPO = TT([site[:, :, np.newaxis, :] for site in MPS])

# Solve alternating linear scheme (single site)
eigval, eigtens, it = evp.als(H, initial_guess = MPO, sigma = -1)
print(eigval)

# Solve power method
pm = evp.power_method(H, initial_guess = MPO, sigma = -1, repeats = 10)
print(pm)

