import segyio
import numpy as np
import torch
from bruges.filters.wavelets import ormsby
#from models_old import forward_model
from skimage.measure import compare_ssim as ssim
#%% # default parameters

def get_params():
    params = dict()
    params['dt'] = 8e-3
    params['dz'] = 10
    params['incident_angles'] = np.arange(0, 30 + 1, 10)  # in degrees
    params['vertical_scale'] = 2
    params['data_path'] = "Data/SEAM_I_2D_Model/"
    params['density_file'] = "SEAM_Den_Elastic_N23900.sgy"
    params['vp_file'] = "SEAM_Vp_Elastic_N23900.sgy"
    params['vs_file'] = "SEAM_Vs_Elastic_N23900.sgy"
    return params

#%% Metrics
def correlation(x,y, return_mean=True):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    num_traces = x.shape[0]
    num_angles = x.shape[1]
    corr = np.zeros((num_traces,num_angles))
    for i in range(num_traces):
        x_mean = np.mean(x[i], axis=-1, keepdims=True)
        y_mean = np.mean(y[i], axis=-1, keepdims=True)
        x_std = np.std(x[i], axis=-1, keepdims=True)
        y_std = np.std(y[i], axis=-1, keepdims=True)
        corr [[i],:]= (np.mean((x[i]-x_mean)*(y[i]-y_mean), axis=-1,keepdims=True)/(x_std*y_std)).T

    if return_mean:
        return np.mean(corr, axis=0)
    else:
        return corr

def r2_coeff(x,y,return_mean=True):
    #x: predicted
    #y: target
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    num_traces = x.shape[0]
    num_angles = x.shape[1]
    r2 = np.zeros((num_traces,num_angles))

    for i in range(num_traces):
        S_tot = np.sum((y[i]-np.mean(y[i], axis=-1, keepdims=True))**2, axis=-1, keepdims=True)
        S_res = np.sum((x[i] - y[i])**2, axis=-1, keepdims=True)
        r2[[i],:] = (1-S_res/S_tot).T

    if return_mean:
        return np.mean(r2, axis=0)
    else:
        return r2

def calc_ssim(x,y, return_mean=True):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    num_angles = x.shape[1]
    ssim_score = np.zeros(num_angles)
    if not return_mean:
        ssim_map = np.zeros_like(x)

    for i in range(num_angles):
        x_tmp = x[:,i]
        y_tmp = y[:,i]

        max_val = np.max([np.max(y_tmp),np.max(x_tmp)])
        min_val = np.min([np.min(y_tmp),np.min(x_tmp)])

        y_tmp = (y_tmp-min_val)/(max_val-min_val)
        x_tmp = (x_tmp-min_val)/(max_val-min_val)
        if return_mean:
            ssim_score[i] = ssim(x_tmp,y_tmp,gaussian_weights=True, sigma=1.5)
        else:
            _,ssim_map[:,i] = ssim(x_tmp, y_tmp, gaussian_weights=True, sigma=1.5, full=True)
    return ssim_map


#%%
from matplotlib.colors import LinearSegmentedColormap
def RedGreenBlue(n_bins=256):
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    n_bins = 256
    cmap_name = 'RedGreenBlue'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


#%% Fatiando toolbox adapted to work in Python 3

"""
Zero-offset convolutional seismic modeling

Give a depth model and obtain a seismic zero-offset convolutional gather. You
can give the wavelet, if you already have, or use one of the existing, from
which we advise ricker wavelet (rickerwave function).

* :func:`~fatiando.seismic.conv.convolutional_model`: given the reflectivity
  series and wavelet, it returns the convolutional seismic gather.
* :func:`~fatiando.seismic.conv.reflectivity`: calculates the reflectivity
  series from the velocity model (and density model if present).
* :func:`~fatiando.seismic.conv.depth_2_time`: convert depth property model to
  the model in time.
* :func:`~fatiando.seismic.conv.rickerwave`: calculates a ricker wavelet.

References
----------

Yilmaz, Oz,
Ch.2 Deconvolution. In: YILMAZ, Oz. Seismic Data Analysis: Processing,
Inversion, and Interpretation of Seismic Data. Tulsa: Seg, 2001. Cap. 2.
p. 159-270. Available at: <http://dx.doi.org/10.1190/1.9781560801580.ch2>


"""
import numpy as np
from scipy import interpolate  # linear interpolation of velocity/density

def depth_2_time(vel, model, dt, dz):
    """
    Convert depth property model to time model.

    Parameters:

    * vel : 2D-array
        Velocity values in the depth domain.
    * model : 2D-array
        Model values in the depth domain.
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram, in
        general a value of 2.e-3 is used.
    * dz : float
        Length of square grid cells.
    * rho : 2D-array (optional)
        Density values for all the model, in depth domain.

    Returns:

    * model_t : 2D-array
        Property model in time domain.

    """
    err_message = "Velocity and model matrix must have the same dimension."
    assert vel.shape == model.shape, err_message
    # downsampled time rate to make a better interpolation
    n_samples, n_traces = [vel.shape[0], vel.shape[1]]
    dt_dwn = dt/10.
    if dt_dwn > dz/np.max(vel):
        dt_dwn = (dz/np.max(vel))/10.
    TWT = np.zeros((n_samples, n_traces))
    TWT[0, :] = 2*dz/vel[0, :]
    for j in range(1, n_samples):
        TWT[j, :] = TWT[j-1]+2*dz/vel[j, :]
    TMAX = max(TWT[-1, :])
    TMIN = min(TWT[0, :])
    TWT_rs = np.zeros(int(np.ceil(TMAX/dt_dwn)))
    for j in range(1, len(TWT_rs)):
        TWT_rs[j] = TWT_rs[j-1]+dt_dwn
    resmpl = int(dt/dt_dwn)
    model_t = _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces)
    return model_t


def _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces):
    """
    Resamples the input data to adjust it after time conversion with the chosen
    time sample rate, dt.

    Returns:

    * vel_l : 2D-array
        Resampled input data.

    """

    vel = np.ones((int(np.ceil(TMAX/dt_dwn)), n_traces))
    for j in range(0, n_traces):
        kk = int(np.ceil(TWT[0, j]/dt_dwn))
        lim = int(np.ceil(TWT[-1, j]/dt_dwn))-1
    # necessary do before resampling to have values in all points of time model
        tck = interpolate.interp1d(TWT[:, j], model[:, j])
        vel[kk:lim, j] = tck(TWT_rs[kk:lim])
    # the model is extended in time because of depth time conversion
        vel[lim:, j] = vel[lim-1, j]
    # because of time conversion, the values between 0 e kk need to be filed
        vel[0:kk, j] = model[0, j]
    # resampling from dt_dwn to dt
    vel_l = np.zeros((int(np.ceil(TMAX/dt)), n_traces))
    resmpl = int(dt/dt_dwn)
    vel_l[0, :] = vel[0, :]
    for j in range(0, n_traces):
        for jj in range(1, int(np.ceil(TMAX/dt))):
            vel_l[jj, j] = vel[resmpl*jj, j]
    return vel_l

#%% Make data
def make_data(params = get_params()):
    file = segyio.open(params['data_path']+params['density_file'],ignore_geometry=True)
    rho = np.array([file.trace[i] for i in range(file.tracecount)])

    file = segyio.open(params['data_path']+params['vp_file'], ignore_geometry=True)
    vp = np.array([file.trace[i] for i in range(file.tracecount)])

    file = segyio.open(params['data_path']+params['vs_file'],ignore_geometry=True)
    vs = np.array([file.trace[i] for i in range(file.tracecount)])

    rho_time = depth_2_time(vp,rho,params['dt'],params['dz'])
    vp_time = depth_2_time(vp,vp,params['dt'],params['dz'])
    vs_time = depth_2_time(vp,vs,params['dt'],params['dz'])

    vp_time = np.expand_dims(vp_time,axis=0)
    vs_time = np.expand_dims(vs_time,axis=0)
    rho_time = np.expand_dims(rho_time,axis=0)

    angles = params['incident_angles']*np.pi/180
    angles = np.expand_dims(np.expand_dims(angles,axis=-1),axis=-1)

    vp_0= np.mean(vp_time,axis=1,keepdims=True)
    vs_0= np.mean(vs_time,axis=1,keepdims=True)
    rho_0= np.mean(rho_time,axis=1,keepdims=True)

    K = (vs_time/vp_time)**2
    A = (vp_time/vp_0)**(1+(np.tan(angles))**2)
    B = (vs_time/vs_0)**(-8*K*(np.sin(angles))**2)
    C = (rho_time/rho_0)**(1-4*K*(np.sin(angles)**2))
    EI = vp_0*rho_0*A*B*C
    EI = EI[:,700:-400,:]

    EI = np.rollaxis(EI,axis=-1)

    EI = EI[..., :int(np.floor(EI.shape[-1] / (params['vertical_scale'])) * (params['vertical_scale']))]

    EI = torch.tensor(EI).float()

    wavelet = segyio.cube("Data/SEAM_I_2D_Data/Wavelet/SEAM_wavelet-g-zph_8ms.sgy")
    wavelet = np.squeeze(wavelet)
    wavelet = torch.from_numpy(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    forward = forward_model(wavelet)

    seismic_clean = forward(EI)[...,::params['vertical_scale']]

    P_seismic = torch.mean(seismic_clean ** 2)
    P_noise = P_seismic / 10 ** (params['SNR'] / 10)
    noise = torch.randn_like(seismic_clean)*torch.sqrt(P_noise)
    seismic_noisy = seismic_clean + noise

    data = dict()
    data['seismic_clean'] = seismic_clean.numpy()
    data['seismic_noisy'] = seismic_noisy.numpy()
    data['EI'] = EI.numpy()
    data['params'] = params

    return data
#%%
from torch.utils.data import Dataset, DataLoader


class SeismicData(Dataset):
    def __init__(self, seismic):
        self.seismic = seismic
    def __len__(self):
        return self.seismic.shape[0]

    def __getitem__(self, idx):
        seismic_trace = self.seismic[idx]

        return seismic_trace

#%%
def compare_dic(a,b):
    assert (isinstance(a,dict) and isinstance(b,dict)), 'inputs must be dictionaries'

    if len(set(a.keys())^set(b.keys()))!=0:
        return 0

    for key in a.keys():
        tmp = (a[key] == b[key])
        try:
            tmp = tmp.all()
        except:
            pass
        if not tmp:
            return 0

    return 1


def normalize(x, n1,n2, mode="mean"):
    if mode=="mean":
        return (x-n1)/n2
    else:
        return 2*((x-n1)/(n2-n1)-0.5)

def unnormalize(x, n1,n2, mode='mean'):
    if mode=="mean":
        return x*n2+n1
    else:
        return (x/2+0.5)*(n2-n1)+n1

#%%
import numpy as np
import matplotlib.pyplot as plt



def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input
    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, fill=True,verbose=False):
    '''Wiggle plot of a sesimic data section
    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)
    Use the column major order for array as in Fortran to optimal performance.
    The following color abbreviations are supported:
    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========
    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        if fill:
          ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                           where=trace_zi >= 0,
                           facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()


#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)