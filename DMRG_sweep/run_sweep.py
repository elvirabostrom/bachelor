import numpy as np 
from scipy import sparse
from scipy import special
import utils
from ttml.tt_cross import estimator_to_tt_cross
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.evp as evp
from time import time

def construct_H(N, L = 15):
    # General setup
    h = 2 * L / (N - 1)
    x = np.linspace(-L, L, N)

    # TT cross for potential
    xg = np.append(x, np.inf)
    dims = [xg] * 3
    TT_cross = estimator_to_tt_cross(utils.Verfgau_for_tt_cross, dims, max_rank = 15, tol = 1e-9, method = 'regular')
    TT_ = [site[:,:-1,:] for site in TT_cross]

    # Verfgau = utils.get_Verfgau(x, x, x, N, 1.5)
    # np.save("output/test_Verfgau_exact.npy", Verfgau)
    # contraction1 = np.einsum('ijk, kpm->ijpm', TT_[0], TT_[1])
    # contraction1 = np.einsum('ijpm, mqi->jpq', contraction1, TT_[2])
    # np.save("output/tt_potential.npy", contraction1)

    # Make potential energy MPO
    V = TT(utils.MPS_to_MPO(TT_))

    # Make kinetic energy MPO
    T = TT(utils.get_kinetic_MPO(h, N))

    # Hamiltonian MPO
    H = V + T

    return H

def solve_ALS(MPO_state, H):
	# Solve alternating linear scheme (single site)
	tic = time()
	eigval, eigtens, it = evp.als(H, initial_guess = MPO_state, sigma = -1)
	toc = time() - tic
	print(f'Time for ALS:  {toc:.2f}')
	return eigval, toc

def solve_pm(MPO_state, H):
	# Solve power method
	tic = time()
	pm = evp.power_method(H, initial_guess = MPO_state, sigma = -1, repeats = 10)
	toc = time() - tic
	print(f'Time for pm:  {toc:.2f}')
	return pm, toc


# Solve FDM
from scipy.sparse import kron, diags
from scipy.sparse.linalg import LinearOperator, eigsh
def solve_FDM(N, L = 15)

    h = 2 * L / (N - 1)
    x1 = np.linspace(-L, L, N)
    x, y, z = np.meshgrid(x1, x1, x1)
    T1 = utils.get_kinetic(N, h)
    T = kron(kron(T1, np.eye(N)), np.eye(N)) + kron(kron(np.eye(N), T1), np.eye(N)) + kron(kron(np.eye(N), np.eye(N)), T1)

    r = np.zeros((N**3, 3))
    r[:, 0] = x.flatten()
    r[:, 1] = y.flatten()
    r[:, 2] = z.flatten()
    V1 = utils.Verfgau_for_tt_cross(r)
    V = diags([V1], [0], shape=(N**3, N**3))

    H = T + V
    tic = time()
    E, Psi = eigsh(H, k=1, which='SA', tol=1e-6)
    toc = time() - tic
    print(f'Time for FDM:  {toc:.2f}')
    return E, toc


bond_dims = [2, 3, 4, 5]
num_points = [50, 100, 150, 200]

for D in bond_dims:
	print('solving for D =', D)
	f = open('output/ALS_N_t_for_bond_dim_' + str(D) + '.txt', 'w')
	g = open('output/pm_N_t_for_bond_dim_' + str(D) + '.txt', 'w')
	t = open('output/FDM_N_t_for_bond_dim_' + str(D) + '.txt', 'w')
	for N in num_points:
		# Initial guess MPO
		MPS = [np.random.rand(1, N, D), np.random.rand(D, N, D), np.random.rand(D, N, 1)]
		MPO = [site[:, :, np.newaxis, :] for site in MPS]
		MPO_state = TT(MPO)
		# hamiltonian
		H = construct_H(N)
		# solve and write to file
		eigval_pm, toc_pm = solve_pm(MPO_state, H)
		g.write(str(toc_pm) + ' ' + str(N) + ' ' + str(list(eigval_pm)[0]) + '\n')
		eigval, toc = solve_ALS(MPO_state, H)
		f.write(str(toc) + ' ' + str(N) + ' ' + str(eigval) + '\n')
		eigval_fdm, toc_fdm = solve_FDM(N)
		t.write(str(toc_fdm) + ' ' + str(N) + ' ' + str(eigval_fdm) + '\n')
	f.close()
	g.close()
	t.close()




# # shape TT
# for i, tensor in enumerate(V):
# print(f'Shape of tensor {i + 1}: {tensor.shape} \n')
