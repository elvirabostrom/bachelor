import numpy as np 
from scipy import sparse
from scipy import special
import utils

# General setup
N = 200
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
	TT = utils.tensor_SVD(N, V, 2)

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
T = utils.get_kinetic(N) * (1 / (2 * h**2)) 

# Compute expectation values
expectation_values = utils.expectation_values(N, T, V, MPS)

print(f'<T> = {expectation_values[0]}')
print(f'<V> = {expectation_values[1]}')
print(f'<E> = {expectation_values[2]}')

