import numpy as np 
from scipy import sparse
from scipy import special
import utils
from ttml.tt_cross import estimator_to_tt_cross
from scikit_tt.tensor_train import TT
#from scipy.sparse.linalg import LinearOperator
import scikit_tt.solvers.evp as evp

# General setup
N = 20
L = 5
h = 2 * L / (N - 1)
x = np.linspace(-L, L, N)

# TT cross for potential
xg = np.append(x[:-1], np.inf)
dims = [xg] * 3
TT_ = estimator_to_tt_cross(utils.Verfgau_for_tt_cross, dims, max_rank = 15, tol = 1e-9, method = 'regular')

# Make potential energy MPO
V = TT(utils.MPS_to_MPO(TT_))

# Make kinetic energy MPO
T = utils.get_kinetic_MPO(h, N)

# Hamiltonian MPO
H = V + T

# Initial guess MPO
M = 10 # bond dim, ≈ 100 is sufficient for 1d ground state (senere)
MPS = [np.random.rand(1, N, M), np.random.rand(M, N, M), np.random.rand(M, N, 1)]
MPO = [site[:, :, np.newaxis, :] for site in MPS]
MPO_state = TT(MPO)

# Solve alternating linear scheme (single site)
eigval, eigtens, it = evp.als(H, initial_guess = MPO_state, sigma = -1)
print(eigval)

# Solve power method
pm = evp.power_method(H, initial_guess = MPO_state, sigma = -1, repeats = 10)
print(pm)



# # finne forventningsverdi til MPS uten MPS formen
# Psi = np.einsum('ijk, kpo, omn->ijpmn', MPS[0], MPS[1], MPS[2]).reshape(N, N, N)
# u = Psi.ravel() #Psi på flat form
# v = utils.get_Verfgau(x, y, z, N, 1.5).ravel() #Potensiale på flat form

# H = T + T + T + V

# def mv(v):
# 	# returnerer A @ v

# A = LinearOperator((dimensjon, dimensjon), matvec = mv(u)) #matrise-vektor produktet, som vi skal finne egenverdien til
# A.matvec(u)
# #finn egenverdi
# eigval, eigvec = sparse.linalg.eigsh(A, k = 1, which = 'SA') #finner egenverdi
# Psi = u.reshape(N, N, N)


# # exact
# Verfgau = utils.get_Verfgau(x, y, z, N, 1.5)
# np.save("output/test_Verfgau_exact.npy", Verfgau)
# # SVD decomp of exact
# Verfgau_TT = utils.tensor_SVD(N, Verfgau, 20)
# #contract tensor trains
# contraction1 = np.einsum('ijk, kpm->ijpm', TT[0], TT[1])
# contraction1 = np.einsum('ijpm, mqi->jpq', contraction1, TT[2])
# print(f"Shape of contracted TT: {contraction1.shape}")
# np.save("output/tt_potential.npy", contraction1)

# contraction = np.einsum('ijk, kpm->ijpm', Verfgau_TT[0], Verfgau_TT[1])
# contraction = np.einsum('ijpm, mqi->jpq', contraction, Verfgau_TT[2])
# print(f"Shape of contracted Verfgau: {contraction.shape}")

# # test with exact
# diff_TT = Verfgau - contraction1
# print('Verfgau - TT: ', np.linalg.norm(diff_TT))

# diff_Verfgau = Verfgau - contraction
# print('Verfgau - Verfgau_TT', np.linalg.norm(diff_Verfgau))




# # compatibility test tensor train datatype

# expectation_values = utils.expectation_values(N, T, TT, MPS)

# print(f'<T> = {expectation_values[0]}')
# print(f'<V> = {expectation_values[1]}')
# print(f'<E> = {expectation_values[2]}')



# # test
# for i, tensor in enumerate(V):
# print(f'Shape of tensor {i + 1}: {tensor.shape} \n')
