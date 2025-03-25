import numpy as np 
from scipy import sparse
from scipy import special

# Generate potential 
def get_Verfgau(x, y, z, N, mu):
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
def tensor_SVD(N, tensor, bond_dim):
	TT = []
	rank2_tensor = np.reshape(tensor, (N, N * N))
	U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	"""
	# Truncate singular values
	for i, value in enumerate(s):
		bond_dim = i
		if value < tol:
			break
	"""

	s = s[:bond_dim]
	U = U[:, :bond_dim] # keep bond_dim columns
	V = V[:bond_dim] # keep bond_dim rows

	# Save U
	U = U[np.newaxis, :, :]
	TT.append(U)

	# Left canonical form
	remaining_tensor = np.diag(s) @ V

	# For all sites that are not on either egde
	for site in range(tensor.ndim - 2):
	    row, col = remaining_tensor.shape
	    rank2_tensor = np.reshape(remaining_tensor, (row * N, N))
	    U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	    """
	    # Truncate singular values
	    for i, value in enumerate(s):
	        bond_dim = i
	        if value < tol:
			    break
	    """

	    s = s[:bond_dim]
	    U = U[:, :bond_dim] # keep trunctated_dim columns
	    V = V[:bond_dim] # keep trunctated_dim rows

	    U = np.reshape(U, (bond_dim, N, bond_dim))
	    TT.append(U)

	    remaining_tensor = np.diag(s) @ V

	# Append last tensor
	TT.append(remaining_tensor[:, :, np.newaxis])

	return TT



# Generate kinetic energy operator
def get_kinetic(N):
	# return N x N matrix operating on physical dimensions
	diagonals = [np.ones(N) * 2, np.ones(N - 1) * - 1, np.ones(N - 1) * - 1]
	T = sparse.diags(diagonals, [0, -1, 1]).toarray()

	return T



# Compute kinetic energy operator acting on MPS
def kinetic_psi(T, MPS):
	# compute T |psi> (contract?) and return new MPS (tridiagonal simplifies?)
	# use T on each site
	MPS_new = []
	for i, site in enumerate(MPS):
		MPS_new.append(np.einsum('jj, ijk->ijk', T, site))
	return MPS_new


# Make MPO 
def MPS_to_MPO(MPS):
	MPO = []
	for site in MPS:
		dim1, N, dim2 = site.shape
		identity = np.eye(N)

		new_site = site[:, :, np.newaxis, :] * identity[np.newaxis, :, :, np.newaxis]
		new_site = new_site.reshape(dim1, N, N, dim2)
		MPO.append(new_site)
	return MPO


# Compute potential energy operator acting on MPS
def potential_psi(V, MPS):
	# compute V |psi> and return new MPS
	# hadamard product
	# MPS_new = [MPS[i][np.newaxis, :, :] * V[i][:, :, np.newaxis] for i in range(len(MPS))]
	# MPS_new = MPS_new.reshape(N, DPsi, DV)

	V_MPO = MPS_to_MPO(V)
	MPS_new = []
	for i, site in enumerate(MPS):
		MPS_new.append(np.einsum('ippa, ipa->ipa', V_MPO[i], site))

	return MPS_new


# Compute expectation value for current state
def expectation_value(T, V, MPS):
	# compute H |psi> = V |psi> + T |psi> = new MPS
	# inner product between <psi| and new MPS  (contraction over two MPS's - site by site)
	# normalize <psi||psi> (some test???)
	# return a scalar
	H_MPS = [potential_psi(V, MPS)[i] + kinetic_psi(T, MPS)[i] for i in range(len(MPS))]
	# Get normalization
	conj_MPS = [np.transpose(site, (2, 1, 0)) for site in MPS]
	norm = 0
	for i, site in enumerate(MPS):
		norm += np.einsum('ijk, ijk', conj_MPS[i], MPS[i])

	# Compute expectation value (make loop)
	A = np.einsum('inm, ino->mo', conj_MPS[0], H_MPS[0])
	B = np.einsum('mnm, mo->onm', conj_MPS[1], A)
	A = np.einsum('onm, ono->mo', B, H_MPS[1]) 
	B = np.einsum('mni, mo->oni', conj_MPS[2], A)
	E = np.einsum('oni, oni', B, H_MPS[2])

	return E / norm


