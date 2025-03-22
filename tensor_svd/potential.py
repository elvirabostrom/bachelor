import numpy as np 
from scipy import sparse
from scipy import special

"""
# Load Verfgau tensor from cpp
N = 10 # set equal to dimensions used as input in potential.cpp
dtype = np.float64
data = np.fromfile("output/potential_test.bin", dtype=dtype)
tensor = data.reshape((N, N, N))
"""

# general setup
N = 20
L = 2
h = 2 * L / (N - 1)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)

# generate potential
def generate_Verfgau_tensor(mu):
	V = np.zeros((N, N, N))
	c = 0.923 + 1.568 * mu
	a = 0.2411 + 1.405 * mu

	for i in range(len(x)):
		for j in range(len(y)):
			for k in range(len(z)):
				if i == j == k == 0:
					r_ijk = mu  
				else:
					r_ijk = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
				erf_r = special.erf(r_ijk)
				V[i, j, k] = erf_r / r_ijk * c * np.exp(- a * r_ijk)

	return V


# Perform SVD
def tensor_SVD(tensor, tol):
	TT = []
	rank2_tensor = np.reshape(tensor, (N, N * N))
	U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	# Truncate singular values
	for i, value in enumerate(s):
		truncated_dim = i
		if value < tol:
			break

	s = s[:truncated_dim]
	U = U[:, :truncated_dim] # keep trunctated_dim columns
	V = V[:truncated_dim] # keep trunctated_dim rows

	# Save U
	TT.append(U)

	# Left canonical form
	remaining_tensor = np.diag(s) @ V

	# For all sites that are not on either egde
	for site in range(tensor.ndim - 2):
	    row, col = remaining_tensor.shape
	    rank2_tensor = np.reshape(remaining_tensor, (row * N, N))
	    U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	    # Truncate singular values
	    for i, value in enumerate(s):
	    	truncated_dim = i
	    	if value < tol:
	    		break

	    s = s[:truncated_dim]
	    U = U[:, :truncated_dim] # keep trunctated_dim columns
	    V = V[:truncated_dim] # keep trunctated_dim rows

	    rows, cols = TT[site].shape
	    U = np.reshape(U, (cols, N, truncated_dim))
	    TT.append(U)

	    remaining_tensor = np.diag(s) @ V

	# Append last tensor
	TT.append(remaining_tensor)

	return TT


"""
# test
for i, tensor in enumerate(TT):
	print(f'Shape of tensor {i + 1}: {tensor.shape} \n')
# contracted tensor
contraction = np.tensordot(TT[0], TT[1], axes = 1)
contraction = np.tensordot(contraction, TT[2], axes = 1)
print(f"Shape of contracted TT: {contraction.shape}")
# alternativt
contraction = np.einsum('ij, jkl->ikl', TT[0], TT[1])
contraction = np.einsum('ikl,lm->ikm', contraction, TT[2])
print(f"Shape of contracted TT: {contraction.shape}")
"""
#Diff = tensor - contraction
#print(np.linalg.norm(Diff))


# kinetic energy
def get_kinetic_operator():
	# return N x N matrix operating on physical dimensions
	diagonals = [np.ones(N) * 2, np.ones(N - 1) * - 1, np.ones(N - 1) * - 1]
	T = sparse.diags(diagonals, [0, -1, 1]).toarray()

	return T

"""
# MPS anzats
M = 120 # bond dim, sufficient for 1d ground state
site1 = np.random.random((N, M)) # vet ikke hva slags fordeling jeg bør velge fra, hvis det har noe å si
site2 = np.random.random((N, M))
site3 = np.random.random((N, M))
"""

def kinetic_psi(T, MPS):
	# compute T |psi> (contract) and return new MPS
	# use T on each site
	MPS_new = []
	for i, site in enumerate(MPS):
		MPS_new[i] = np.einsum('ijk, jj->ijk', T, site)
	return MPS_new

def potential_psi(V, MPS):
	# compute V |psi> and return new MPS
	# hadamand product (numpy.multiply(x1, x2, /, out=ndarray))
	MPS_new = []
	for i, site in enumerate(MPS):
		MPS_new[i] = np.multiply(site, V[i], out = ndarray)
	return MPS_new

def expectation_value(T, V, MPS):
	# compute H |psi> = V |psi> + T |psi> = new MPS
	# inner product between <psi| and new MPS  (contraction over two MPS's - site by site)
	# normalize <psi||psi> (some test to check if it is actually normalized???)
	# return a scalar
	H_MPS = np.add(potential_psi(V, MPS), kinetic_psi(T, MPS), out = ndarray)
	# Get normalization
	conj_MPS = []
	for site in MPS:
		conj_MPS.append(np.conj(site))
	norm = 0
	for i, site in enumerate(MPS):
		norm += np.einsum('ijk, ijk', conj_MPS[i], MPS[i])

	# Compute expectation value
	M = np.einsum('pjl, ijk->kl', conj_MPS[0], H_MPS[0])

	for i in range(len(H_MPS) - 1):
		B = np.einsum('lrq, kl->krq', conj_MPS[i + 1], M)
		M = np.einsum('krs, krm->sm', H_MPS[i + 1], B)

	B = np.einsum('kr, rsp->ksp', conj_MPS[-1], M)
	E = np.einsum('ksp, ksp', B, H_MPS[-1]) / norm
	return E


# Verfgau exact
potential = generate_Verfgau_tensor(1.5)
np.save("output/tensor_test.npy", potential)
# TT of potential
tol = 1e-15
V = tensor_SVD(potential, tol)
# Kinetic energy
T = get_kinetic_operator()




