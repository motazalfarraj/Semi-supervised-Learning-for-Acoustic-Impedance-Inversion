import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import *
from time import time
from models_2d import *
from os.path import isfile
import torch.optim as optim
import random
#import seaborn as sns
import segyio
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
rho = np.array([file.trace[i] for i in range(file.tracecount)]).T

file = segyio.open(params['data_path'] + params['vp_file'], ignore_geometry=True)
vp = np.array([file.trace[i] for i in range(file.tracecount)]).T

vp = np.expand_dims(vp, axis=1)
rho = np.expand_dims(rho, axis=1)

AI = vp*rho
AI = AI.T

AI_ilines = np.arange(1001,8001+1,4)
s_ilines = np.arange(1499,7505+1,6)

common_item = np.intersect1d(AI_ilines, s_ilines)

file = segyio.open("Data/SEAM_Interpretation_Challenge_1_Depth/SEAM_Interpretation_Challenge_1_Depth.sgy")
seismic = segyio.cube(file)
seismic = seismic.squeeze()

inds = np.array([i in common_item for i in AI_ilines])
AI = AI[inds]

inds = np.array([i in common_item for i in s_ilines])
seismic = seismic[:,inds]

seismic = seismic[...,50:700]
AI = AI[...,100:1400][...,::2]

seismic = np.expand_dims(np.swapaxes(seismic, axis1=-2,axis2=-1), axis=1)
AI = np.expand_dims(np.swapaxes(np.swapaxes(AI, axis1=-2,axis2=-1),axis1=-1,axis2=0), axis=1)
#%% Preping data

AI_mean = np.mean(AI, keepdims=True)
AI_std = np.std(AI,keepdims=True)

seismic_mean = np.mean(seismic,keepdims=True)
seismic_std = np.std(seismic,keepdims=True)

AI = torch.tensor(AI).float()
AI_mean = torch.tensor(AI_mean).float()
AI_std = torch.tensor(AI_std).float()

seismic = torch.tensor(seismic).float()

seismic_mean = torch.tensor(seismic_mean).float()
seismic_std = torch.tensor(seismic_std).float()

seismic = normalize(seismic, seismic_mean, seismic_std)
AI = normalize(AI, AI_mean, AI_std)


#%%
if torch.cuda.is_available():
    # AI = AI.cuda()
    # seismic = seismic.cuda()
    seismic_mean = seismic_mean.cuda()
    seismic_std = seismic_std.cuda()
    AI_mean = AI_mean.cuda()
    AI_std = AI_std.cuda()

#%%
seismic_ref = seismic[[717]]
seismic = seismic[::5]
#seismic = seismic.view(-1,seismic.shape[-1]).unsqueeze(1)

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


inverse_model = InverseModel(num_inputs=1, num_channels=[4,8,10,16,20], kernel_size=(7,3))
forward_model = ForwardModel(num_channels=1)

if torch.cuda.is_available():
    inverse_model = inverse_model.cuda()
    forward_model.cuda()

max_iter = 500
optimizer = optim.Adam(list(inverse_model.parameters())+ list(forward_model.parameters()),
                       weight_decay=1e-4, amsgrad=True)
criterion = nn.MSELoss()
#%% Training
AI_loss = []
seismic_loss = []
validation_loss = []
torch.cuda.empty_cache()
batch_size=2
train_inds = np.linspace(0,AI.shape[-1]-1,20).astype(int)
start_time = time()

inverse_model.train()
forward_model.train()
a = 0.2
b = 1
c = 1
count = 0
inds = np.arange(0,seismic.shape[0])
np.random.shuffle(inds)
for iter in range(max_iter):
    for batch in range(0,seismic.shape[0], batch_size):
        optimizer.zero_grad()
        seismic_batch = seismic[inds[batch:batch+batch_size]]

        if torch.cuda.is_available():
            seismic_batch = seismic_batch.cuda()
            AI = AI.cuda()
            seismic_ref = seismic_ref.cuda()

        AI_inv = inverse_model(seismic_batch)
        seismic_inv = forward_model(AI_inv)
        loss1 = criterion(seismic_inv, seismic_batch)

        AI_ref_inv = inverse_model(seismic_ref)
        loss2 = criterion(AI_ref_inv, AI)

        seismic_inv = forward_model(AI)
        loss3 = criterion(seismic_inv, seismic_ref)

        loss = a*loss1+b*loss2 + c*loss3

        loss.backward()
        optimizer.step()

        del seismic_batch, AI_inv, AI_ref_inv

        torch.cuda.empty_cache()

        print("iter: {:4}/{} | Training loss: {:0.4f} | AI loss: {:0.4f}".format(iter + 1,max_iter,
                                                                                 loss.item(), loss2.item()))

        torch.cuda.empty_cache()
        if batch+batch_size>seismic.shape[0]:
            np.random.shuffle(inds)

        torch.cuda.empty_cache()

    #%% Predicting AI

inverse_model.eval()
forward_model.eval()

with torch.no_grad():
    AI_inv = inverse_model(seismic_ref)
    seismic_inv = forward_model(AI_inv)
#%%
plt.imshow(AI_inv.detach().cpu().numpy()[0,0], cmap="rainbow")
plt.title("Predicted AI")
plt.colorbar()
plt.show()


plt.imshow(AI.cpu().numpy()[0,0],cmap="rainbow")
plt.title("True AI")
plt.colorbar()
plt.show()

plt.imshow(abs(AI-AI_inv)[0,0].cpu().numpy(), cmap='gray')
plt.title("Difference")
plt.colorbar()
plt.show()


plt.imshow((seismic_ref.detach().cpu().numpy())[0,0])
plt.title("True Seismic")
plt.colorbar()
plt.show()

plt.imshow((seismic_inv.detach().cpu().numpy())[0,0])
plt.title("Synth Seismic")
plt.colorbar()
plt.show()

#%%
plt.plot(validation_loss)
plt.plot(AI_loss)
plt.plot(seismic_loss)
plt.legend(["Validation loss","AI loss", "seismic loss"])
plt.show()

# #%%
# from skimage.measure import compare_ssim as ssim
#
# x = AI.cpu().numpy().T
# y = AI_inv.cpu().numpy().T
# y = (y-y.min())/(y.max()-y.min())
# x = (x-x.min())/(x.max()-x.min())
#
# y = np.swapaxes(y,axis1=-1,axis2=1)
# x = np.swapaxes(x,axis1=-1,axis2=1)
# z = ssim(x,y,gaussian_weights=True, sigma=1.5, multichannel=True)
#
# print(z)
# #%%
# np.save("AI_loss.npy", AI_loss)
# np.save("seismic_loss.npy", seismic_loss)
# np.save("validation_loss.npy", validation_loss)
# np.save("elapsed_time.npy", elapsed_time)
#
