import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# PLot 2D slices of rank3 tensor along Z axis - slices in XY plane
def plot_tensor_slices(tensor):
    num_slices = tensor.shape[0]  # Number of slices along Z

    fig, axes = plt.subplots(1, num_slices, figsize = (15, 5))
    vmin = tensor.min()
    vmax = tensor.max()

    im_list = []
    for i in range(num_slices):
        im = axes[i].imshow(tensor[:, :, i], cmap = 'viridis', aspect = 'auto', vmin = vmin, vmax = vmax)
        axes[i].set_title(f"$z_{i}$")
        axes[i].axis("off")
        im_list.append(im)
    cbar = fig.colorbar(im_list[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Intensity")
    plt.tight_layout()
    plt.savefig('output/potential_slices_in_z.pdf')
    plt.show()

# 3D plot 
def plot_3d_tensor(tensor):
    fig = go.Figure(data = go.Volume(
        x = np.repeat(np.arange(tensor.shape[0]), tensor.shape[1] * tensor.shape[2]),
        y = np.tile(np.repeat(np.arange(tensor.shape[1]), tensor.shape[2]), tensor.shape[0]),
        z = np.tile(np.arange(tensor.shape[2]), tensor.shape[0] * tensor.shape[1]),
        value = tensor.flatten(),
        opacity = 0.3,  
        isomin = tensor.min(),
        isomax = tensor.max(),
        surface_count = 10
    ))
    fig.show()

# visualize tt-cross potential
TT_cross = np.load("output/tt_potential.npy")
#plot_tensor_slices(TT_cross)
#plot_3d_tensor(TT_cross)


# compare TT-cross with exact
Verfgau = np.load("output/test_Verfgau_exact.npy")
N = 100
L = 15
h = 2 * L / (N - 1)

# errors
diff_TT = Verfgau - TT_cross
print('Verfgau - TT: ', np.linalg.norm(diff_TT) * np.sqrt(h**3))
print('rel error: ', np.linalg.norm(diff_TT) / np.linalg.norm(Verfgau))
print('største feil: ', np.max(np.abs(diff_TT)))

# annen error (men samme som linalg.norm?)
diff = np.sqrt(np.sum(diff_TT**2) * h**3)
print('diff: ', diff)

# hist
plt.figure(figsize = (4,3))
plt.hist(diff_TT.flatten(), bins=50, density = True)
plt.yscale('log')
plt.xlabel('Error')
plt.ylabel('Number of points')
plt.tight_layout()
plt.savefig('output/potential_error_distribution.pdf')
plt.show()

# worst positions
# diff_flat = diff_TT.flatten()
# worst_indices = np.argsort(diff_flat)[:10]  # 10 minste verdier (mest negative, mest avvik)
# shape = diff_TT.shape
# worst_positions = np.unravel_index(worst_indices, shape)
# print('x, y, z arrays med høyest avvik: ', worst_positions)

# error in z-plane
plt.figure(figsize = (4,3))
plt.imshow(np.log10(np.abs(1e-16 + diff_TT[:,:,N//2])))
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.tight_layout()
plt.savefig('output/potential_error_zplane.pdf')
plt.show()

#tomography
error = np.abs(diff_TT)**2
tom = np.sum(error, axis = 2) * h
plt.figure(figsize = (4,3))
plt.imshow(np.log10(tom + 1e-16))
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar(label = ' ', format = '{x:.2f}')
plt.tight_layout()
plt.savefig('output/potential_error_density.pdf')
plt.show()

error = np.abs(diff_TT)**2
tom = np.sum(error, axis = 0) * h
plt.figure(figsize = (4,3))
plt.imshow(np.log10(tom + 1e-16))
plt.xlabel('z')
plt.ylabel('y')
plt.colorbar(label = ' ', format = '{x:.2f}')
plt.tight_layout()
plt.savefig('output/potential_error_density_x.pdf')
plt.show()

error = np.abs(diff_TT)**2
tom = np.sum(error, axis = 1) * h
plt.figure(figsize = (4,3))
plt.imshow(np.log10(tom + 1e-16))
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar(label = ' ', format = '{x:.2f}')
plt.tight_layout()
plt.savefig('output/potential_error_density_y.pdf')
plt.show()

# # 3d vizualisation of largest errors
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.where(np.abs(diff_TT) > 0.0001)  # terskel
# sc = ax.scatter(x, y, z, c=diff_TT[x, y, z], cmap='coolwarm', alpha=0.5)
# plt.colorbar(sc, label='avvik')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.title('større avvik')
# plt.show()

