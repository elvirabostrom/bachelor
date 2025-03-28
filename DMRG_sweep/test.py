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
	TT = utils.tensor_SVD(N, V, 9)

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

	# TEST KINETISK FORVENTNINGSVERDI
	"""
	T = utils.get_kinetic(N) * (1 / (2 * h**2))
	T_psi_x = np.tensordot(T, Psi_norm, axes=([1], [0]))
	T_psi_y = np.tensordot(T, Psi_norm, axes=([1], [1]))
	T_psi_z = np.tensordot(T, Psi_norm, axes=([1], [2]))
	E_kin_x = np.sum(np.conj(Psi_norm) * T_psi_x)
	E_kin_y = np.sum(np.conj(Psi_norm) * T_psi_y)
	E_kin_z = np.sum(np.conj(Psi_norm) * T_psi_z)
	E_kin_total = (E_kin_x + E_kin_y + E_kin_z).real
	print("Kinetisk energi (direkte beregning):", E_kin_total)
	"""

	# np.save("output/test_harmonic_GS.npy", np.conj(Psi) * Psi)
	MPS = utils.tensor_SVD(N, Psi, 10)

	return MPS


# Initializations
V = V_HO_TT() 
MPS = HO_ground_state() 
T = utils.get_kinetic(N) * (1 / (2 * h**2)) 

MPS_conj, norm = utils.get_bra_state(MPS)
print("Norm av MPS:", norm)


E_kinetic = utils.kinetic_psi(T, MPS)


# Computate expectation value
E_kinetic, E_potential, E = utils.expectation_value(N, T, V, MPS) 
print(E_kinetic)
print(E_potential)
print(E)


#test
"""
contraction = np.einsum('ijk, kmn->ijmn', MPS[0], MPS[1])
contraction = np.einsum('ijmn,nwe->ijmwe', contraction, MPS[2])
contraction = contraction.reshape(50, 50, 50)
print("Norm av Psi etter SVD:", np.linalg.norm(contraction))
print("Max Psi etter SVD:", np.max(np.abs(contraction)))

print('tridiagonal matrise før skalering: ', utils.get_kinetic(N))
"""
