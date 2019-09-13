import numpy as np

data = np.load("data.npy", allow_pickle=True).item()

seismic = data["seismic"]
acoustic_impedance = data["acoustic_impedance"]

print(acoustic_impedance.shape)
