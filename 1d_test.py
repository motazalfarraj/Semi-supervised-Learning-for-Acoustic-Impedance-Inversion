import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import *
from time import time
from model import *
from os.path import isfile
import torch.optim as optim
import random
#import seaborn as sns
import segyio

from torch.nn.functional import  mse_loss
#%%data
def get_params():
    params = dict()
    params['dt'] = 8e-3
    params['dz'] = 10
    params['data_path'] = "Data/SEAM_I_2D_Model/"
    params['density_file'] = "SEAM_Den_Elastic_N23900.sgy"
    params['vp_file'] = "SEAM_Vp_Elastic_N23900.sgy"
    params['vs_file'] = "SEAM_Vs_Elastic_N23900.sgy"
    return params

params = get_params()
file = segyio.open(params['data_path'] + params['density_file'], ignore_geometry=True)
rho = np.array([file.trace[i] for i in range(file.tracecount)])

file = segyio.open(params['data_path'] + params['vp_file'], ignore_geometry=True)
vp = np.array([file.trace[i] for i in range(file.tracecount)])

AI = vp*rho


file = segyio.open("Data/SEAM_Interpretation_Challenge_1_Depth/SEAM_Interpretation_Challenge_1_2DGathers_Depth.sgy")
seismic = segyio.cube(file)

AI_ilines = np.arange(1001,8001+1,4)
s_ilines = file.ilines

common_item = np.intersect1d(AI_ilines, s_ilines)
seismic = seismic.squeeze()

inds = np.array([i in common_item for i in AI_ilines])
AI = AI[inds, ::2].T
vp = vp[inds, ::2].T

inds = np.array([i in common_item for i in s_ilines])
seismic = seismic[inds]

AI = AI[40:-40]
seismic = seismic[...,40:-40]
vp = vp[40:-40]
#%
water = 1534.6996
salt = 9699.2

mask = np.zeros_like(AI)
mask[AI==water]= 1

tmp = np.zeros_like(AI)
tmp[np.where(AI==salt)]=1
tmp = np.cumsum(abs(np.diff(tmp, axis=0))[::-1], axis=0)[::-1]

mask[:-1][tmp==0] = 1
mask[-1] = 1
#%%
# seismic_mean = np.mean(seismic[(1-mask).astype(bool)], axis=(0,-1), keepdims=True)
# seismic_std = np.std(seismic(1-mask).astype(bool), axis=(0,-1), keepdims=True)
#
# seismic = (seismic-seismic_mean)/seismic_std
seismic = np.mean(seismic,axis=1).T

AI = depth_2_time(vp,AI,params['dt'],params['dz'])
seismic = depth_2_time(vp,seismic,params['dt'],params['dz'])
mask = depth_2_time(vp,mask,params['dt'],params['dz']).astype(int)

mask = np.expand_dims(mask.T,axis=1)
seismic = np.expand_dims(seismic.T,axis=1)
AI = np.expand_dims(AI.T,axis=1)

seismic = np.expand_dims(seismic.mean(axis=1),1)
#%% Preping data
from copy import deepcopy
AI_mean = np.mean(AI[(1-mask).astype(bool)])
AI_std = np.std(AI[(1-mask).astype(bool)])

seismic_mean = np.mean(seismic[(1-mask).astype(bool)])
seismic_std = np.std(seismic[(1-mask).astype(bool)])

seismic = normalize(seismic, seismic_mean, seismic_std)
AI = normalize(AI, AI_mean, AI_std)

AI[(mask).astype(bool)]=-999
mask_seismic = deepcopy(seismic)
mask_seismic[(mask).astype(bool)]=-999

AI = torch.tensor(AI).float()
seismic = torch.tensor(seismic).float()
mask_seismic = torch.tensor(mask_seismic).float()


#%%
if torch.cuda.is_available():
    AI = AI.cuda()
    seismic = seismic.cuda()
    mask_seismic = mask_seismic.cuda()



#%%# Preparing the  model
import random
random_seed = 30
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
forward_model = ForwardModel(num_channels=1)

if torch.cuda.is_available():
    forward_model.cuda()

train_inds = np.linspace(0,AI.shape[0]-1,20).astype(int)
#criterion = nn.MSELoss()

def criterion(pred, target, ignore=-999):
    error = mse_loss(pred, target, reduction="none")
    error[target==ignore] = 0
    return error.mean()

optimizer_f = optim.Adam(forward_model.parameters(), lr=0.001, amsgrad=True)

#%%
forward_model.train()
for iter in range(2000):
    optimizer_f.zero_grad()
    seismic_inv = forward_model(AI[train_inds])
    loss = criterion(seismic_inv, mask_seismic[train_inds])
    loss.backward()
    print(loss.item())
    optimizer_f.step()

forward_model.eval()

#%%
# for param in forward_model.parameters():
#     param.requires_grad = False
#%%
max_iter = 100
inverse_model = InverseModel(num_angles=1)

if torch.cuda.is_available():
    inverse_model = inverse_model.cuda()

optimizer = optim.Adam(inverse_model.parameters(), lr = 0.01, weight_decay=0, amsgrad=True)
#% Training
AI_loss = []
seismic_loss = []
validation_loss = []
torch.cuda.empty_cache()
batch_size=50
start_time = time()
a = 0.2
b = 1
#%%
inverse_model.train()
count = 0
inds = np.arange(0,seismic.shape[0])
np.random.shuffle(inds)
for iter in range(max_iter):
    for batch in range(0,seismic.shape[0], batch_size):
        forward_model.zero_grad()
        inverse_model.zero_grad()

        seismic_batch = seismic[inds[batch:batch+batch_size]]
        AI_inv = inverse_model(seismic_batch)
        seismic_inv = forward_model(AI_inv)
        loss1 = criterion(seismic_inv, mask_seismic[inds[batch:batch+batch_size]])

        AI_inv = inverse_model(seismic[train_inds])
        loss2 = criterion(AI_inv, AI[train_inds])

        loss = a*loss1+b*loss2
        loss.backward()
        optimizer.step()


        del seismic_batch, AI_inv

        torch.cuda.empty_cache()
        print("iter: {:4}/{} | Training loss: {:0.4f} | AI loss: {:0.4f}".format(iter + 1,max_iter,
                                                                                 loss.item(), loss2.item()))
        torch.cuda.empty_cache()

    if iter%5==0:
        inverse_model.eval()
        AI_inv = inverse_model(seismic)
        val_loss = criterion(AI_inv, AI)

        plt.imshow((AI_inv.detach().cpu().numpy()*(1-mask))[:, 0].T, cmap="rainbow")
        plt.title("Predicted AI | {:0.4f}".format(val_loss))
        plt.colorbar()
        plt.show()
        inverse_model.train()
#%% Predicting AI

inverse_model.eval()
forward_model.eval()

with torch.no_grad():
    AI_inv = inverse_model(seismic)
    seismic_inv = forward_model(AI_inv)
#%%
plt.imshow((AI_inv.detach().cpu().numpy()*(1-mask))[:,0].T, cmap="rainbow")
plt.title("Predicted AI")
plt.colorbar()
plt.show()


plt.imshow(((AI.cpu().numpy())*(1-mask))[:,0].T,cmap="rainbow")
plt.title("True AI")
plt.colorbar()
plt.show()

plt.imshow((abs(AI-AI_inv).cpu().numpy()*(1-mask))[:,0].T, cmap='gray')
plt.title("Difference")
plt.colorbar()
plt.show()

plt.imshow(((seismic.detach().cpu().numpy())*(1-mask))[:,0].T)
plt.title("True Seismic")
plt.colorbar()
plt.show()

plt.imshow(((seismic_inv.detach().cpu().numpy())*(1-mask))[:,0].T)
plt.title("Synth Seismic")
plt.colorbar()
plt.show()

#%%
plt.plot(validation_loss)
plt.plot(AI_loss)
plt.plot(seismic_loss)
plt.legend(["Validation loss","AI loss", "seismic loss"])
plt.show()

