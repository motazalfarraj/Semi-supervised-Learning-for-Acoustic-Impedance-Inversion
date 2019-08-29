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
    params['dt'] = 1e-3
    params['dz'] = 1.25
    params['data_path'] = "marmousi/"
    params['density_file'] = "MODEL_DENSITY_1.25m.segy"
    params['vp_file'] = "MODEL_P-WAVE_VELOCITY_1.25m.segy"
    return params

params = get_params()
file = segyio.open(params['data_path'] + params['density_file'])
rho = segyio.cube(file).squeeze().T
rho = rho[:, ::5]

file = segyio.open(params['data_path'] + params['vp_file'])
vp = segyio.cube(file).squeeze().T
vp = vp[:, ::5]
AI = vp*rho

seismic = segyio.cube("marmousi/SYNTHETIC.segy").squeeze().T


AI = depth_2_time(vp,AI,params['dt'],params['dz'])
seismic = depth_2_time(vp[::4],seismic,params['dt'],params['dz'])



AI = AI[4*180:4*650]
seismic = seismic[180:650]


#%% Preping data

AI = np.expand_dims(AI.T,1)
seismic = np.expand_dims(seismic.T,1)

AI_mean = np.mean(AI)
AI_std = np.std(AI)

seismic_mean = np.mean(seismic)
seismic_std = np.std(seismic)

seismic = normalize(seismic, seismic_mean, seismic_std)
AI = normalize(AI, AI_mean, AI_std)


AI = torch.tensor(AI).float()
seismic = torch.tensor(seismic).float()

#%%
if torch.cuda.is_available():
    AI = AI.cuda()
    seismic = seismic.cuda()

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
train_inds = np.linspace(0,AI.shape[0]-1,20).astype(int)
criterion = nn.MSELoss()

max_iter = 100
inverse_model = InverseModel(num_angles=1, vertical_scale=4)
forward_model = ForwardModel(num_channels=1)

if torch.cuda.is_available():
    inverse_model = inverse_model.cuda()
    forward_model = forward_model.cuda()

optimizer = optim.Adam(list(inverse_model.parameters())+list(forward_model.parameters()),
                       amsgrad=True,lr=0.005)
#% Training
AI_loss = []
seismic_loss = []
validation_loss = []
torch.cuda.empty_cache()
batch_size=50
start_time = time()
a = 0.2
b = 1
c = 1
#%%
inverse_model.train()
forward_model.train()
count = 0
inds = np.arange(0,seismic.shape[0])
np.random.shuffle(inds)
for iter in range(max_iter):
    for batch in range(0,seismic.shape[0], batch_size):
        forward_model.zero_grad()
        inverse_model.zero_grad()

        seismic_batch = seismic[inds[batch:batch+batch_size]]

        AI_inv_1 = inverse_model(seismic_batch)
        seismic_inv_1 = forward_model(AI_inv_1)
        loss1 = criterion(seismic_inv_1, seismic_batch)

        AI_inv_2 = inverse_model(seismic[train_inds])
        loss2 = criterion(AI_inv_2, AI[train_inds])

        seismic_inv_3 = forward_model(AI[train_inds])
        loss3 = criterion(seismic_inv_3, seismic[train_inds])

        loss = a*loss1+b*loss2+c*loss3
        loss.backward()
        optimizer.step()


        torch.cuda.empty_cache()
        print("iter: {:4}/{} | Training loss: {:0.4f} | AI loss: {:0.4f}".format(iter + 1,max_iter,
                                                                                 loss.item(), loss2.item()))
        torch.cuda.empty_cache()

    with torch.no_grad():
        inverse_model.eval()
        forward_model.eval()
        AI_inv = inverse_model(seismic)
        val_loss = criterion(AI_inv, AI)
        r2 = r2_coeff(AI_inv, AI)
        plt.imshow((AI_inv.detach().cpu().numpy())[:, 0].T, cmap="rainbow")
        plt.title("Predicted AI | {:0.4f} | {:0.4f}".format(val_loss, r2.item()))
        plt.colorbar()
        plt.show()
        forward_model.train()
        inverse_model.train()
#%% Predicting AI

inverse_model.eval()
forward_model.eval()

with torch.no_grad():
    AI_inv = inverse_model(seismic)
    seismic_inv = forward_model(AI_inv)


#%%
AI = AI*AI_std+AI_mean
AI_inv = AI_inv*AI_std+AI_mean

#%%
np.save('AI.npy', AI.detach().cpu().numpy())
np.save('AI_inv.npy', AI_inv.detach().cpu().numpy())
#%%
plt.imshow((AI_inv.detach().cpu().numpy())[:,0].T, cmap="rainbow", vmin=AI.min(), vmax=AI.max())
plt.title("Predicted AI")
plt.colorbar()
plt.show()


plt.imshow(((AI.cpu().numpy()))[:,0].T,cmap="rainbow",vmin=AI.min(), vmax=AI.max())
plt.title("True AI")
plt.colorbar()
plt.show()

plt.imshow((abs(AI-AI_inv).cpu().numpy())[:,0].T, cmap='gray')
plt.title("Difference")
plt.colorbar()
plt.show()

plt.imshow(((seismic.detach().cpu().numpy()))[:,0].T)
plt.title("True Seismic")
plt.colorbar()
plt.show()

plt.imshow(((seismic_inv.detach().cpu().numpy()))[:,0].T)
plt.title("Synth Seismic")
plt.colorbar()
plt.show()

#%%
plt.plot(validation_loss)
plt.plot(AI_loss)
plt.plot(seismic_loss)
plt.legend(["Validation loss","AI loss", "seismic loss"])
plt.show()

#%%

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

