import segyio
import numpy as np
from functions import *


# Reading p-wave velocity and density from segy
p_velocity = np.squeeze(segyio.cube("data/MODEL_P-WAVE_VELOCITY_1.25m.segy"))[::5].T # sample every 5 traces to match the seismic
density = np.squeeze(segyio.cube("data/MODEL_DENSITY_1.25m.segy"))[::5].T

#computing acoustic_impedance as the product
acoustic_impedance = p_velocity*density

# Reading synthetic seismic data
seismic = np.squeeze(segyio.cube("data/SYNTHETIC.segy")).T

#Convert to time
acoustic_impedance = depth_2_time(p_velocity,acoustic_impedance,1e-3, 1.25)
seismic = depth_2_time(p_velocity[::4],seismic,1e-3, 1.25)

# Cutting out empty parts of the model
acoustic_impedance = acoustic_impedance[4*180:4*650]
seismic = seismic[180:650]


np.save("data/seismic.npy", seismic)
np.save("data/acoustic_impedance.npy", acoustic_impedance)
