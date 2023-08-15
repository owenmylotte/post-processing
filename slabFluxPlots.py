import numpy as np
import matplotlib.pyplot as plt  # for plotting
import matplotlib.font_manager
import matplotlib.ticker as ticker
import tikzplotlib
from chrest.chrestData import ChrestData

# Take experimental data and plot contour.

data_flux = ChrestData(input['base_path'] + "/" + input['dns'])

# Take the heat flux in from ablate data.
# Plot heat flux contour on top plot.
# Calculate average heat flux.

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

# Take the slice of the flux that is on the top face. Maybe just manually find it.



# Take previous model data and plot contour.


# Compute the error on the heat flux between the experimental, previous model, and current model.
