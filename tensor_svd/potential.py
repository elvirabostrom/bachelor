import numpy as np 

# Load Verfgau tensor
N = 10 # set equal to dimensions used as input in potential.cpp
dtype = np.float64
data = np.fromfile("output/potential_test.bin", dtype=dtype)
tensor = data.reshape((N, N, N))

TT = []
tol = 1e-15

# Perform SVD and truncation of singular values
def tensor_SVD(tensor, tol, outer):
	U, s, V = np.linalg.svd(tensor, full_matrices = False, compute_uv=True) 

	# Truncate singular values
	for i, value in enumerate(s):
		if value < tol:
			truncated_dim = i
			break

	s = s[:truncated_dim]
	U = U[:, :truncated_dim] # keep trunctated_dim columns
	V = V[:truncated_dim] # keep trunctated_dim rows

	# Save U
	if outer:
		TT.append(U)
	else:
		U = np.reshape(U, (int(len(tensor) / N), N, truncated_dim))
		TT.append(U)

	# Transfer singular values
	return np.diag(s) @ V


# SVD of original tensor
rank2_tensor = np.reshape(tensor, (N, N * N))
remaining_tensor = tensor_SVD(rank2_tensor, tol, True)

# SVD of remaining tensor
row, col = remaining_tensor.shape
rank2_tensor = np.reshape(remaining_tensor, (row * N, N))
remaining_tensor = tensor_SVD(rank2_tensor, tol, False)

# Append last tensor to TT/MPS
TT.append(remaining_tensor)

for i, tensor in enumerate(TT):
	print(f'Shape of tensor {i + 1}: {tensor.shape} \n')


