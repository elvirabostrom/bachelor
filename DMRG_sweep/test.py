import numpy as np 
from scipy import sparse
from scipy import special
import utils

# General setup
N = 10
L = 2
h = 2 * L / (N - 1)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)

# Precalculate
x = 0.5 * x**2
y = 0.5 * y**2
z = 0.5 * z**2

# Harmonic oscillator potensial tensor train decomp
def V_HO_TT():
	V = np.zeros((N, N, N))
	for i in range(len(x)):
		for j in range(len(y)):
			for k in range(len(z)):
				V[i ,j, k] = x[i] + y[j] + z[k]
	# np.save("output/test_harmonic.npy", V)
	TT = utils.tensor_SVD(N, V, 10)
	return TT

# Generate ground state
def HO_ground_state():
	Psi = np.zeros((N, N, N))
	psi_x = np.exp(-x)
	psi_y = np.exp(-y)
	psi_z = np.exp(-z)
	for i in range(len(psi_x)):
		for j in range(len(psi_y)):
			for k in range(len(psi_z)):
				Psi[i ,j, k] = psi_x[i] * psi_y[j] * psi_z[k]
	# np.save("output/test_harmonic_GS.npy", np.conj(Psi) * Psi)
	MPS = utils.tensor_SVD(N, Psi, 10)
	return MPS


# Initializations
V = V_HO_TT()

MPS = HO_ground_state()

T = utils.get_kinetic(N)


# foreløpig test (går ellers rett i expectation_value())
T_MPS = utils.kinetic_psi(T, MPS)
V_MPS = utils.potential_psi(V, MPS) # putt disse inni expec values senere

# Computate expectation value
E = utils.expectation_value(T, V, MPS) 
print(E)