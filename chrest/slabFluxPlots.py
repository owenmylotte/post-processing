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
experimental_data *= 1.E3  # Convert to watts per meter squared.
print(experimental_data)

# %% Define a variable to hold the flux averages.
flux_averages = np.zeros(3)  # This will be indexed based on the following definitions
experimental_index = 0
original_compute_index = 1
dns_index = 2
magic_index = 10  # TODO: Update this with the index of the slice which the boundary sits on.

# %% Take experimental data and plot contour.
# TODO: Read in the experimental data from the matlab file in whatever encoding it's currently saved in.
# Compute the average of the experimental data computed from the tcp
flux_averages[experimental_index] = np.average(experimental_data)  # TODO: Take the average of the experimental data.

# %% Take the heat flux in from ablate data.
# Define the slab flux data name as whatever is saved in the data file.
slab_flux_name = "flux"  # TODO: Update this with the actual slab flux name.

data_boundary = ChrestData(
    "/Users/owen/paraffinPpms/_3dSlabBurner_2023-08-03_10ppm/slab boundary_monitor" + "/" + "slab boundary_monitor.00097.chrest/slab boundary_monitor.00097.chrest.00000.hdf5")
flux = data_boundary.get_field(slab_flux_name)

# THIS CAN BE USED AS REFERENCE FOR IMPLEMENTING THE CHREST FORMAT FUNCTIONS.
# def get_optical_thickness(self, dns_data):
#     # Calculate the optical thickness of the frame
#     # First get the absorption for each cell in the dns
#     dns_temperature, _, _ = dns_data.get_field(self.dns_temperature_name)
#     if self.dns_soot is None:
#         self.get_dns_soot(dns_data)
#     kappa = (3.72 * self.dns_soot * self.C_0 * dns_temperature) / self.C_2  # Soot mean absorption
#     # Then sum the absorption through all cells in the ray line
#     axis_values = [1, 2]
#     optical_thickness_attributes = ['front_dns_optical_thickness', 'top_dns_optical_thickness']
#
#     for axis, attribute in zip(axis_values, optical_thickness_attributes):
#         dns_sum_soot = kappa.sum(axis=axis, keepdims=True)
#         setattr(self, attribute, dns_sum_soot * dns_data.delta[2 - axis])

# Check that the slab flux exists, get the field out of the object.
if flux is None:
    raise Exception(slab_flux_name + " does not exist.")

# Take the average of the slab flux where the value is not zero.
# (Take the slice of the flux that is on the top face. Maybe just manually find it.)
flux_averages[dns_index] = np.average(flux[np.where(flux != 0)])  # Is this the correct way to do this?

# %% Take previous model data and plot contour.
# TODO: Read in the experimental data from the MATLAB file in whatever encoding it's currently saved in.
# Compute the average of the experimental data computed from the tcp
flux_averages[original_compute_index] = 0  # TODO: Average the original model flux and compare it to the current model.

# %% Plot heat flux contour on top plot.
# TODO: Mesh grid stuff with the slice of the contour on the magic index.
plt.plot(flux[:, :, magic_index])  # TODO: Update this to plot a proper contour.

# %% Compute the error on the heat flux between the experimental, previous model, and current model.
# TODO: Error on between each of the entries in the fluxes.
error = np.zeros(len(flux_averages - 1))
print(error)
