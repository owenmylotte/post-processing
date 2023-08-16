import numpy as np
import matplotlib.pyplot as plt  # for plotting
import matplotlib.font_manager
import matplotlib.ticker as ticker
import tikzplotlib
from chrestData import ChrestData
from chrest.xdmfGenerator import XdmfGenerator
import scipy.io

file_path = '/Users/owen/Downloads/radHFLowG.mat'

mat = scipy.io.loadmat(file_path)
experimental_data = mat['totIntensity']
experimental_data *= 1.E6  # Convert to watts per meter squared.
# print(experimental_data)

# %% Define a variable to hold the flux averages.
flux_averages = np.zeros(3)  # This will be indexed based on the following definitions
experimental_index = 0
original_compute_index = 1
dns_index = 2
magic_index = 10  # TODO: Update this with the index of the slice which the boundary sits on.

# %% Take experimental data and plot contour.
# Compute the average of the experimental data computed from the tcp
flux_averages[experimental_index] = np.average(experimental_data)  # Take the average of the experimental data.

# %% Take the heat flux in from ablate data.
# Define the slab flux data name as whatever is saved in the data file.
slab_flux_name = "flux"

data_boundary = ChrestData(
    "/Users/owen/paraffinPpms/_3dSlabBurner_2023-08-03_10ppm/slab boundary_monitor" + "/" + "slab boundary_monitor.00097.chrest/slab boundary_monitor.00097.chrest.00000.hdf5")
field_things = data_boundary.get_field(slab_flux_name)
# Check that the slab flux exists, get the field out of the object.
if field_things is None:
    raise Exception(slab_flux_name + " does not exist.")
flux = np.asarray(field_things[0])

# Take the average of the slab flux where the value is not zero.
# (Take the slice of the flux that is on the top face. Maybe just manually find it.)
flux_averages[dns_index] = np.average(flux)
print(flux_averages)

# %% Plot heat flux contour on top plot.
# TODO: Mesh grid stuff with the slice of the contour on the magic index.
# plt.plot(flux[:, :, magic_index])  # TODO: Update this to plot a proper contour.

# %% Compute the error on the heat flux between the experimental and current model.
error = (flux_averages[0] - flux_averages[2]) / flux_averages[0]
print(error)
