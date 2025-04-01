import numpy as np 
from scipy import sparse
from scipy import special
import utils
from ttml.tt_cross import estimator_to_tt_cross

# General setup
N = 100
L = 10
h = 2 * L / (N - 1)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)

# exact
Verfgau = utils.get_Verfgau(x, y, z, N, 1.5)
np.save("output/test_Verfgau_exact.npy", Verfgau)

# TT cross
xg = np.append(x[:-1], np.inf)
dims = [xg] * 3
TT = estimator_to_tt_cross(utils.Verfgau_for_tt_cross, dims, max_rank = 20, tol = 1e-9, method = 'regular') #max_rank > 5 endrer lite

# SVD decomp of exact
Verfgau_TT = utils.tensor_SVD(N, Verfgau, 20)


#contract tensor trains
contraction1 = np.einsum('ijk, kpm->ijpm', TT[0], TT[1])
contraction1 = np.einsum('ijpm, mqi->jpq', contraction1, TT[2])
print(f"Shape of contracted TT: {contraction1.shape}")
np.save("output/tt_potential.npy", contraction1)

contraction = np.einsum('ijk, kpm->ijpm', Verfgau_TT[0], Verfgau_TT[1])
contraction = np.einsum('ijpm, mqi->jpq', contraction, Verfgau_TT[2])
print(f"Shape of contracted Verfgau: {contraction.shape}")

# test with exact
diff_TT = Verfgau - contraction1
print('Verfgau - TT: ', np.linalg.norm(diff_TT))

diff_Verfgau = Verfgau - contraction
print('Verfgau - Verfgau_TT', np.linalg.norm(diff_Verfgau))


"""
# compatibility test tensor train datatype
T = utils.get_kinetic(N)
M = 10 # bond dim, sufficient for 1d ground state
site1 = np.random.random((1, N, M)) # vet ikke hva slags fordeling jeg bør velge fra, hvis det har noe å si
site2 = np.random.random((M, N, M))
site3 = np.random.random((M, N, 1))
MPS = [site1, site2, site3]

expectation_values = utils.expectation_values(N, T, TT, MPS)

print(f'<T> = {expectation_values[0]}')
print(f'<V> = {expectation_values[1]}')
print(f'<E> = {expectation_values[2]}')
"""


"""
# test
for i, tensor in enumerate(V):
	print(f'Shape of tensor {i + 1}: {tensor.shape} \n')

# contracted tensor
contraction = np.tensordot(V[0], V[1], axes = 1)
contraction = np.tensordot(contraction, V[2], axes = 1)
print(f"Shape of contracted TT: {contraction.shape}")
# alternativt
contraction = np.einsum('ij, jkl->ikl', V[0], V[1])
contraction = np.einsum('ikl,lm->ikm', contraction, V[2])
print(f"Shape of contracted TT: {contraction.shape}")
"""
#Diff = tensor - contraction
#print(np.linalg.norm(Diff))




