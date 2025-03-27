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

# Perform SVD to make TT
def tensor_SVD(N, tensor, bond_dim):
	TT = []
	rank2_tensor = np.reshape(tensor, (N, N * N))
	U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	s = s[:bond_dim]
	U = U[:, :bond_dim] # keep bond_dim columns
	V = V[:bond_dim] # keep bond_dim rows

	# Append first tensor
	U = U[np.newaxis, :, :]
	TT.append(U)

	# Left canonical form
	remaining_tensor = np.diag(s) @ V

	# For all sites that are not on either egde
	for site in range(tensor.ndim - 2):
	    row, col = remaining_tensor.shape
	    rank2_tensor = np.reshape(remaining_tensor, (row * N, N))
	    U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	    s = s[:bond_dim]
	    U = U[:, :bond_dim] # keep trunctated_dim columns
	    V = V[:bond_dim] # keep trunctated_dim rows

	    U = np.reshape(U, (bond_dim, N, bond_dim))
	    TT.append(U)

	    remaining_tensor = np.diag(s) @ V

	# Append last tensor
	TT.append(remaining_tensor[:, :, np.newaxis])

	return TT

# Compute transpose (only real numbers) of MPS and the norm
def get_bra_state(MPS):
	conj_MPS = [np.transpose(site, (2, 1, 0)) for site in MPS]
	norm = 0
	for i, site in enumerate(MPS):
		norm += np.einsum('ijk, ijk', conj_MPS[i], MPS[i])

	return conj_MPS, norm

# Make MPO - add one additional physical index on each site
def MPS_to_MPO(MPS):
	MPO = []
	for site in MPS:
		dim1, N, dim2 = site.shape
		identity = np.eye(N)

		new_site = site[:, :, np.newaxis, :] * identity[np.newaxis, :, :, np.newaxis]
		new_site = new_site.reshape(dim1, N, N, dim2)
		MPO.append(new_site)

	return MPO



# Generate kinetic energy operator
def get_kinetic(N):
	# return N x N matrix operating on physical dimensions
	diagonals = [np.ones(N) * 2, np.ones(N - 1) * - 1, np.ones(N - 1) * - 1]
	T = sparse.diags(diagonals, [0, -1, 1]).toarray()

	return T

# Compute kinetic energy operator acting on MPS
def kinetic_psi(T, MPS):
	# Compute T |psi>
	# bond dim will not grow here
	MPS_new = [np.einsum('jl, ilk->ijk', T, site) for site in MPS]

	# Compute expectation value
	MPS_conj, norm = get_bra_state(MPS)
	E_kinetic = np.einsum('ipj, jqk, krl, xpy, yqz, zrw ->ilxw', MPS_new[0], MPS_new[1], MPS_new[2], MPS_conj[0], MPS_conj[1], MPS_conj[2], optimize = True)
	E_kinetic = E_kinetic[0][0][0][0] / norm

	return E_kinetic

# Compute potential energy operator acting on MPS
def potential_psi(N, V, MPS):
	# Compute V |psi>
	V_MPO = MPS_to_MPO(V)
	MPS_new = []
	for i, site in enumerate(MPS):
		contracted_MPS = np.einsum('ipqa, jqb->ijpab', V_MPO[i], site)
		new_bond_dim1 = list(V_MPO[i].shape)[0] * list(site.shape)[0]
		new_bond_dim2 = list(V_MPO[i].shape)[-1] * list(site.shape)[-1]
		MPS_new.append(contracted_MPS.reshape(new_bond_dim1, N, new_bond_dim2)) #truncate this?? it will be very big

	# Compute expectation value
	MPS_conj, norm = get_bra_state(MPS)
	E_potential = np.einsum('ipj, jqk, krl, xpy, yqz, zrw ->ilxw', MPS_new[0], MPS_new[1], MPS_new[2], MPS_conj[0], MPS_conj[1], MPS_conj[2], optimize = True)
	E_potential = E_potential[0][0][0][0] / norm

	return E_potential

# Compute expectation value for current state
def expectation_value(N, T, V, MPS):
	E_kinetic = kinetic_psi(T, MPS)
	E_potential = potential_psi(N, V, MPS)
	E = E_kinetic + E_potential

	return E_kinetic, E_potential, E


