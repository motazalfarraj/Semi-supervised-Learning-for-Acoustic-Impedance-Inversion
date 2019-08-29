import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

AI = np.load('AI.npy')
AI_inv = np.load('AI_inv.npy')


#%% Saving Figures (Slow)
dt = 1e-3
dx = 1.25*5

vmin = min([AI.min(), AI_inv.min()])
vmax = max([AI.max(), AI_inv.max()])

def plot(img, dt, dx, cmap='rainbow', cbar_label=r'AI ($m/s\times g/cm^3$)', vmin=None, vmax=None):
    time = np.arange(0.6, dt * img.shape[-1] + 0.6, dt)
    x = np.linspace(0, 17000, img.shape[0])
    Y, X = np.mgrid[slice(time.min(), time.max() + dt, dt),
                    slice(0, x.max() + dx, dx)]

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    if (vmin is None or vmax is None):
        plt.pcolormesh(X, Y, img.T, cmap=cmap)
    else:
        plt.pcolormesh(X, Y, img.T, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar()
    plt.ylabel("Time  (s)", fontsize=30)
    plt.xlabel("Distance (m)", fontsize=30, labelpad=15)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position("top")
    plt.gca().set_xticks(np.arange(0, 17000 + 1, 1700 * 2))
    plt.tick_params(axis='both', which='major', labelsize=30)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(cbar_label, rotation=270, fontsize=30, labelpad=40)
    return fig

fig = plot(AI[:,0], dt=dt, dx=dx, vmin=vmin, vmax=vmax)
fig.savefig('AI.png', bbox_inches='tight')
fig = plot(AI_inv[:, 0], dt=dt, dx=dx,vmin=vmin, vmax=vmax)
fig.savefig('AI_inv.png', bbox_inches='tight')
fig = plot(abs(AI_inv[:, 0]-AI[:, 0]), dt=dt, dx=dx, cmap='gray', cbar_label='Absolute Difference')
fig.savefig('AI_diff.png', bbox_inches='tight')

#%%
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")
fig = plt.figure()
np.random.seed(30)
inds = np.random.choice(AI.shape[0],30)
x = np.reshape(AI[inds, 0], -1)
y = np.reshape(AI_inv[inds, 0], -1)

std = AI[:,0].std()

max = np.max([AI[:,0].max(),AI_inv[:,0].max()])
min = np.min([AI[:, 0].min(), AI_inv[:, 0].min()])

d = {'True AI': x, 'Estimated AI': y}
df = pd.DataFrame(data=d)

fig = plt.figure(figsize=(15, 15))
g = sns.jointplot("Estimated AI", "True AI", data=df, kind="reg",
                  xlim=(min, max), ylim=(min, max), color="k", height=15, scatter_kws={'s':10},label='big')


plt.plot([min, max + std], [min - std, max], "r--", zorder=0)
plt.plot([min, max + std], [min + std, max + 2 * std], "r--")
plt.fill_between([min, max + std], [min - std, max],
                 [min + std, max + 2 * std],
                 facecolor='r', alpha=0.1)

plt.xlabel(r"Estimated AI", fontsize=30)
plt.ylabel(r"True AI", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


plt.savefig('Scatter.png',bbox_inches='tight')


plt.show()

#%%
#
# time_elapsed= np.load('time_elapsed.npy')
# n = np.arange(0,max_iter)
# t = np.linspace(0, time_elapsed,len(n))
#
# if True:
#     plt.close('all')
#     fig, ax = plt.subplots(1,1,figsize=(10,5))
#     ax.plot(n,training_loss, linewidth=3.0, color='r')
#     ax.set_aspect(aspect=90)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     plt.xlabel('Iteration number', fontsize=20)
#     plt.ylabel('Training Loss', fontsize=20)
#     plt.xlim(n[0]-2,n[-1]+1)
#     plt.ylim(0,np.max(training_loss))
#     plt.tight_layout()
#     plt.show()
#
# fig.savefig('Figures/LearningCurve.png',bbox_inches='tight')
#%% Generating results and plots

plt.close('all')

x_loc = np.array([3400,6800,10200,13600])
inds = (AI.shape[0]*(x_loc/17000)).astype(int)

x = AI[inds].squeeze()
y = AI_inv[inds].squeeze()
time = np.arange(0.6, dt * AI.shape[-1] + 0.6, dt)
ang = np.arange(0,30+1,10)
fig, ax = plt.subplots(1,x.shape[0], figsize=(10,12),sharey=True)

max = np.max([y.max(), x.max()])*1.2
min = np.min([y.min(), x.min()])*0.8

for i in range(len(inds)):
    p1 = ax[i].plot(x[i],time, 'k')
    p2 = ax[i].plot(y[i],time, 'r')
    ax[i].set_xlabel(r'AI($m/s \times g/cm^3$)'+'\n'+r'$distance={}$'.format(x_loc[i]), fontsize=15)
    if i==0:
        ax[i].set_ylabel('Time (s)', fontsize=20)
        ax[i].yaxis.set_tick_params(labelsize=20)

    ax[i].set_ylim(time[0],time[-1])
    ax[i].set_xlim(min,max)
    ax[i].invert_yaxis()
    ax[i].xaxis.set_tick_params(labelsize=15)


fig.legend([p1[0],p2[0]],["True AI","Estiamted AI"], loc="upper center", fontsize=20,bbox_to_anchor=(0.5, 1.07))
plt.show()

fig.savefig('AI_traces.png'.format(x_loc),bbox_inches='tight')


#%%
AI_corr = correlation(AI_inv,AI, return_mean=False)
AI_r2 = r2_coeff(AI_inv,AI, return_mean=False)
AI_ssim = calc_ssim(AI_inv,AI, return_mean=False)
x = np.reshape(AI_r2,-1)
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(10,5), sharex=True)
sns.despine(left=True)
sns.distplot(x, hist=False, color="g", kde_kws={"shade": True})
plt.xlabel(r"Coefficient of determination ($r^2$)", fontsize=20)
plt.setp(axes, yticks=[])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.xlim([0.5, 1.05])
plt.savefig("dist_r2.png", bbox_inches='tight', dpi=600)
plt.show()

x = np.reshape(AI_corr,-1)
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(10,5), sharex=True)
sns.despine(left=True)
sns.distplot(x, hist=False, color="b", kde_kws={"shade": True})
plt.xlabel("Pearson Correlation Coefficient (PCC)", fontsize=20)
plt.setp(axes, yticks=[])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.xlim([0.5, 1.05])
plt.savefig("dist_corr.png", bbox_inches='tight', dpi=600)
plt.show()

x = np.reshape(AI_ssim[::4,:,::4],-1)
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(10,5), sharex=True)
sns.despine(left=True)
sns.distplot(x, hist=False, color="r", kde_kws={"shade": True})
plt.xlabel("SSIM score", fontsize=20)
plt.setp(axes, yticks=[])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.xlim([0.5, 1.05])
plt.savefig("dist_ssim.png", bbox_inches='tight', dpi=600)
plt.show()
