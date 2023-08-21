import numpy as np
import matplotlib.pyplot as plt  # for plotting
import matplotlib.font_manager
import matplotlib.ticker as ticker
import tikzplotlib
from chrestData import ChrestData
from chrest.xdmfGenerator import XdmfGenerator
import scipy.io
from scipy.ndimage import gaussian_filter

# file_path = '/Users/owen/Downloads/radHFLowG.mat'
file_path = '/home/owen/Downloads/radHFLowG.mat'

mat = scipy.io.loadmat(file_path)
experimental_data = mat['totIntensity']
experimental_data *= 1.E6  # Convert to watts per meter squared.

# %% Define a variable to hold the flux averages.
flux_averages = np.zeros(4)  # This will be indexed based on the following definitions
experimental_index = 0
dns_index = 1  # 10, 30, and 60 ppm respectively will be stored on and after.
magic_index = 30  # TODO: Update this with the index of the slice which the boundary sits on.

# %% Take experimental data and plot contour.
# Compute the average of the experimental data computed from the tcp
flux_averages[experimental_index] = np.average(experimental_data)  # Take the average of the experimental data.

# %% Take the heat flux in from ablate data.
# Define the slab flux data name as whatever is saved in the data file.
slab_flux_name = "flux"

# Perform the processing step for each set of experimental data.
file_names = ["10ppm", "30ppm", "60ppm"]  # ppm of the separate cases.

flux = np.zeros([3, 50, 50, 200])

for i in range(3):
    data_boundary = ChrestData(
        # "/Users/owen/paraffinPpms/_3dSlabBurner_2023-08-03_10ppm/slab boundary_monitor" + "/" +
        # "slab boundary_monitor.00097.chrest/slab boundary_monitor.00097.chrest.00000.hdf5")
        "/home/owen/paraffinSlabThing/" + file_names[i] + "/slab boundary_monitor.hdf5")
    field_things = data_boundary.get_field(slab_flux_name)
    # Check that the slab flux exists, get the field out of the object.
    if field_things is None:
        raise Exception(slab_flux_name + " does not exist.")
    flux[i, :, :, :] = np.asarray(field_things[0])

    # Take the average of the slab flux where the value is not zero.
    # (Take the slice of the flux that is on the top face. Maybe just manually find it.)
    flux_averages[dns_index + i] = np.average(flux[i, 17:33, magic_index, 45:185])

print(flux_averages)

# Set the contour plot dimensions which the data will be sliced from.
slab_width = 0.004750 * 2  # width of the slab in meters.
slab_start = np.array([0.0232, ((0.0254/2) - (slab_width/2))])
slab_end = np.array([0.0933, slab_start[1] + slab_width])
box_dims = np.array([0.15, 0.0254])
cells = np.array([200, 50])
start_index = (np.round(cells * slab_start / box_dims)).astype(int)
end_index = (np.round(cells * slab_end / box_dims)).astype(int)

# %% Plot heat flux contour on top plot.
fig, axarr = plt.subplots(4, 1, figsize=(8, 15))

# Define the x and y range assuming it's a regular grid
# y, x = np.mgrid[0:1:50j, 0:1:200j]
y, x = np.mgrid[0:16, 0:140]

# Define the x and y range for experimental_data
y_exp, x_exp = np.mgrid[0:1:9j, 0:1:80j]

# Specify the contour levels
contour_levels = np.linspace(0, 3, 31)

for i in range(3):
    # c = axarr[i].contour(x, y, flux[i, start_index[1]:end_index[1], magic_index, start_index[0]:end_index[0]],
    #                      levels=contour_levels, colors='black')  # Contour plot with specified levels.
    c = axarr[i].contour(x, y, 1.E-6 * gaussian_filter(flux[i, 17:33, magic_index, 45:185], sigma=1),
                         levels=contour_levels, colors='black')  # Contour plot with specified levels.
    # c = axarr[i].contour(x, y, 1.E-6 * gaussian_filter(flux[i, :, magic_index, :], sigma=1),
    #                      levels=contour_levels, colors='black')  # Contour plot with specified levels.
    axarr[i].clabel(c, inline=1, fontsize=10)
    # axarr[i].set_title(f"Heat Flux Contour for {file_names[i]}")
    axarr[i].set_xlabel("X-axis label")
    axarr[i].set_ylabel("Y-axis label")

# Contour plot for experimental_data
experimental_data *= 1.E-6
contour_levels_exp = np.linspace(np.min(experimental_data), np.max(experimental_data), 10)  # Adjust if needed.
c = axarr[3].contour(x_exp, y_exp, experimental_data, levels=contour_levels_exp, colors='black')
axarr[3].clabel(c, inline=1, fontsize=10)
axarr[3].set_xlabel("X-axis label for Experimental Data")
axarr[3].set_ylabel("Y-axis label for Experimental Data")

plt.tight_layout()
plt.show()

# %% Compute the error on the heat flux between the experimental and current model.

# flux_averages[2] = flux[2, 17:33, magic_index, 45:185]

error = np.zeros(4)
for i in range(1, len(flux_averages)):
    error[i] = (flux_averages[i] - flux_averages[0]) / flux_averages[0]

print(error)
