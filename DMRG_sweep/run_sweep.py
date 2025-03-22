import utils.py

# General setup
N = 20
L = 2
h = 2 * L / (N - 1)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)

# Verfgau exact
potential = get_Verfgau(N, 1.5)
np.save("output/tensor_test.npy", potential)

# TT of potential
tol = 1e-15
V = tensor_SVD(N, potential, tol)

# Kinetic energy
T = get_kinetic(N)



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



"""
# MPS anzats
M = 120 # bond dim, sufficient for 1d ground state
site1 = np.random.random((N, M)) # vet ikke hva slags fordeling jeg bør velge fra, hvis det har noe å si
site2 = np.random.random((N, M))
site3 = np.random.random((N, M))
"""




