import numpy as np 
from scipy import sparse
from scipy import special
#from scikit_tt.tensor_train import TT

# Generate potential exact
def get_Verfgau(x, y, z, N, mu):
	V = np.zeros((N, N, N))
	c = 0.923 + 1.568 * mu
	a = 0.2411 + 1.405 * mu

	for i in range(len(x)):
		for j in range(len(y)):
			for k in range(len(z)):
				r_ijk = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
				if r_ijk  == 0:
					V[i, j, k] = - (mu + c) 
				else:
					erf_r = special.erf(r_ijk * mu)
					V[i, j, k] = - (erf_r / r_ijk + c * np.exp(- (a**2 * r_ijk**2)))

	return V

# Potential for TT cross
def Verfgau_for_tt_cross(r, mu = 1.5):
	c = 0.923 + 1.568 * mu
	a = 0.2411 + 1.405 * mu
	r_ijk = np.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
	erf_r = special.erf(r_ijk * mu)
	V = - (erf_r / r_ijk + c * np.exp(- (a**2 * r_ijk**2)))
	V[(r_ijk == 0)] = - (mu + c)
	
	return V
	

# Perform SVD to make TT
def tensor_SVD(N, tensor, bond_dim):
	Ttrain = []
	rank2_tensor = np.reshape(tensor, (N, N * N))
	U, s, V = np.linalg.svd(rank2_tensor, full_matrices = False, compute_uv=True) 

	s = s[:bond_dim]
	U = U[:, :bond_dim] # keep bond_dim columns
	V = V[:bond_dim] # keep bond_dim rows

	# Append first tensor
	U = U[np.newaxis, :, :]
	Ttrain.append(U)

	# Right canonical form
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
	    Ttrain.append(U)

	    remaining_tensor = np.diag(s) @ V

	# Append last tensor
	Ttrain.append(remaining_tensor[:, :, np.newaxis])

	return Ttrain

# copied from scikit_tt
def add_MPOs(A, B):
	order = len(A)
	A_ranks = [list(A[0].shape)[0], list(A[1].shape)[0], list(A[1].shape)[3], list(A[2].shape)[3]]
	B_ranks = [list(B[0].shape)[0], list(B[1].shape)[0], list(B[1].shape)[3], list(B[2].shape)[3]]
	ranks = [1] + [A_ranks[i] + B_ranks[i] for i in range(1, order)] + [1]
	MPO_SUM = []

	for i in range(order):
		MPO_SUM.append(np.zeros([ranks[i], list(A[i].shape)[1], list(A[i].shape)[2], ranks[i + 1]]))
		MPO_SUM[i][0:A_ranks[i], :, :, 0:A_ranks[i + 1]] = A[i]
		r1 = ranks[i] - B_ranks[i]
		r2 = ranks[i]
		r3 = ranks[i + 1] - B_ranks[i + 1]
		r4 = ranks[i + 1]
		MPO_SUM[i][r1:r2, :, :, r3:r4] = B[i]

	return MPO_SUM

# Compute conjugate MPS and norm
def get_bra_state(MPS):
	conj_MPS = [np.conj(site) for site in MPS]
	norm = np.einsum('ijk, kpl, lmn, qje, epu, umt->inqt', MPS[0], MPS[1], MPS[2], conj_MPS[0], conj_MPS[1], conj_MPS[2], optimize = True)
	norm = norm[0, 0, 0, 0]

	return conj_MPS, norm

# Make diagonal MPO - add one additional physical index on each site
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
def get_kinetic(N, h):
	diagonals = [np.ones(N) * 30, np.ones(N - 1) * - 16, np.ones(N - 1) * - 16, np.ones(N - 2), np.ones(N - 2)]
	T = sparse.diags(diagonals, [0, -1, 1, -2, 2]).toarray()

	return T * (1 / (24 * h**2)) 

# Kinetic energy MPO
def get_kinetic_MPO(h, N):
	identity = np.eye(N)
	T = get_kinetic(N, h)

	# T_x = TT([T[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis]])
	# T_y = TT([identity[np.newaxis, :, :, np.newaxis], T[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis]]) 
	# T_z = TT([identity[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis], T[np.newaxis, :, :, np.newaxis]])
	# T_MPO = T_x + T_y + T_z
	T_x = [T[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis]]
	T_y = [identity[np.newaxis, :, :, np.newaxis], T[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis]]
	T_z = [identity[np.newaxis, :, :, np.newaxis], identity[np.newaxis, :, :, np.newaxis], T[np.newaxis, :, :, np.newaxis]]
	T_ = add_MPOs(T_x, T_y)
	T_MPO = add_MPOs(T_, T_z)

	return T_MPO

# Compute kinetic expectectation value
def kinetic_psi(T, MPS):
	# Compute T |psi>
	Tx_MPS = np.einsum('jl,ilk->ijk', T, MPS[0])  
	Ty_MPS = np.einsum('jl,ilk->ijk', T, MPS[1])  
	Tz_MPS = np.einsum('jl,ilk->ijk', T, MPS[2])  

	# Compute expectation value
	conj_MPS, norm = get_bra_state(MPS)
	E_kinetic = ( np.einsum('ijk, kpl, lmn, qje, epu, umt->inqt', Tx_MPS, MPS[1], MPS[2], conj_MPS[0], conj_MPS[1], conj_MPS[2], optimize = True)
    + np.einsum('ijk, kpl, lmn, qje, epu, umt->inqt', MPS[0], Ty_MPS, MPS[2], conj_MPS[0], conj_MPS[1], conj_MPS[2], optimize = True)
    + np.einsum('ijk, kpl, lmn, qje, epu, umt->inqt', MPS[0], MPS[1], Tz_MPS, conj_MPS[0], conj_MPS[1], conj_MPS[2], optimize = True) )
	E_kinetic = E_kinetic[0, 0, 0, 0] / norm

	return E_kinetic

# Compute potential expectectation value
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
	E_potential = np.einsum('ijk, kpl, lmn, qje, epu, umt->inqt', MPS_new[0], MPS_new[1], MPS_new[2], MPS_conj[0], MPS_conj[1], MPS_conj[2], optimize = True)
	E_potential = E_potential[0, 0, 0, 0] / norm

	return E_potential

# Compute expectation value for current state
def expectation_value(N, T, V, MPS):
	E_kinetic = kinetic_psi(T, MPS)
	E_potential = potential_psi(N, V, MPS)
	E = E_kinetic + E_potential

	return E


