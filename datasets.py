import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


with open("download_data.pickle", "rb") as f:
    u, v, w = pickle.load(f)
# Visualisation
plt.plot(v[:, 10, 10])
plt.show()

plt.contourf(u[1, :, :], cmap="hsv")
plt.quiver(u[10, :, :], v[10, :, :], scale=200)
plt.show()

# Handling the masked array problem by
# 1) Filling the masked arrays with nan
# 2) removing first,second and last row to get it by same grid dimension
# 2.1) Need to get the grid dimension on 2^n form, e.g. 128 x 128
u_nan = np.ma.filled(u.astype(float), np.nan)
u_nomask = u_nan[:, 4:-4, 4:-3]

v_nan = np.ma.filled(v.astype(float), np.nan)
v_nomask = v_nan[:, 4:-4, 4:-3]

w_nan = np.ma.filled(w.astype(float), np.nan)
w_nomask = w_nan[:, 4:-4, 4:-3]

# Creating coarse simulation by skipping every alternate grid
u_coarse = u_nomask[:, ::2, ::2]
v_coarse = v_nomask[:, ::2, ::2]
w_coarse = w_nomask[:, ::2, ::2]

# Transforming the HR data into tensor form
u_tensor = torch.from_numpy(u_nomask)
v_tensor = torch.from_numpy(v_nomask)
w_tensor = torch.from_numpy(w_nomask)
u_tensor = u_tensor.reshape(745, 1, 128, 128)
v_tensor = v_tensor.reshape(745, 1, 128, 128)
w_tensor = w_tensor.reshape(745, 1, 128, 128)

# Transforming the LR data into tensor form
u_tensor_lr = torch.from_numpy(u_coarse)
v_tensor_lr = torch.from_numpy(v_coarse)
w_tensor_lr = torch.from_numpy(w_coarse)
u_tensor_lr = u_tensor_lr.reshape(745, 1, 64, 64)
v_tensor_lr = v_tensor_lr.reshape(745, 1, 64, 64)
w_tensor_lr = w_tensor_lr.reshape(745, 1, 64, 64)

# Concatenating the tensors together like three RGB channels
HR_data = torch.cat((u_tensor, v_tensor, w_tensor), dim=1)
print(HR_data.shape)  # output = dim ( 745, 3, 128, 128)
LR_data = torch.cat((u_tensor_lr, v_tensor_lr, w_tensor_lr), dim=1)
print(LR_data.shape)  # output = dim ( 745, 3, 64, 64)

# Hyperparamters
batchSize = 64
num_epochs = 2
nz = HR_data.shape[0]  # size of generator input
ngf = 64  # size of feature maps in generator
ndf = 128  # size of feature maps in discriminator

# Normalizing
HR_data_norm = (HR_data - HR_data.mean()) / HR_data.std()
LR_data_norm = (LR_data - LR_data.mean()) / LR_data.std()

# Creating training set
dataset_train = torch.utils.data.TensorDataset(LR_data_norm, HR_data_norm)
trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batchSize, shuffle=True, num_workers=0
)

# dataset_test = torch.utils.data.TensorDataset(LR_data)
# testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize,
# shuffle=False, num_workers=2)
