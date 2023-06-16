import argparse
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import pandas as pd
import matplotlib.gridspec as gridspec
from chrest.chrestData import ChrestData
import os
import yaml

plt.rcParams["font.family"] = "Noto Serif CJK JP"


class VTcpData:

    def __init__(self, front_files=None, top_files=None, fields=None, tcp_axis=None, base_path=None, write_path=None,
                 dns_temperature_name=None, dns_soot_name=None, save=None):
        h = 6.62607004e-34
        c = 299792458
        k = 1.380622e-23
        self.C_2 = h * c / k
        self.C_1 = 2 * np.pi * h * c * c
        self.C_0 = 7.0  # Constants for the absorptivity calculations

        self.base_path = base_path
        self.write_path = write_path
        self.tcp_soot = None
        self.save = save
        self.soot_error = None
        self.front_temperature_error = None

        self.dns_soot = None
        self.front_dns_optical_thickness = None
        self.top_dns_optical_thickness = None
        self.dns_maximum_temperature = None
        self.dns_maximum_soot = None
        self.dns_temperature_name = dns_temperature_name
        self.dns_soot_name = dns_soot_name

        self.prf = None
        self.rhoC = 2000  # [kg / m^3]
        self.field_size = len(fields)
        if tcp_axis == "x":
            self.tcp_axis = 0
        if tcp_axis == "y":
            self.tcp_axis = 1
        if tcp_axis == "z":
            self.tcp_axis = 2

        def load_vtcp_data(vtcp_files, field_size):
            vtcp = ChrestData(base_path + "/" + vtcp_files)
            data = np.array([])

            for f in range(field_size):
                data_tmp, _, _ = vtcp.get_field(fields[f])
                data = np.vstack((data, np.expand_dims(data_tmp, axis=0))) if f else np.expand_dims(data_tmp, axis=0)

            return data

        self.start_point = ChrestData(base_path + "/" + front_files).start_point
        self.end_point = ChrestData(base_path + "/" + front_files).end_point
        self.times = np.zeros(self.field_size)
        self.names = np.zeros(self.field_size)
        self.front_tcp_temperature = None
        self.front_temperature_error = None
        self.top_tcp_temperature = None
        self.top_temperature_error = None

        self.front_data = load_vtcp_data(front_files, self.field_size)
        self.top_data = load_vtcp_data(top_files, self.field_size)

        self.set_limits()  # Sets the time step range of the processing

    def get_tcp_temperature(self):
        # Calculate the two-color-pyrometry temperature of the frame
        # First, get the intensity ratio between the red and green channels (0 and 1)
        # Then, use the ratio to get the temperature
        # Finally, plot the temperature
        c = 3.e8  # Speed of light
        h = 6.626e-34  # Planck's constant
        k = 1.3806e-23  # Boltzmann Constant

        # Planck's first and second constant
        c1 = 2. * np.pi * h * c * c
        c2 = h * c / k

        lambdaR = 650e-9
        lambdaG = 532e-9
        threshold_fraction = 0.05  # Threshold for the absolute intensity

        for data_type in ['front', 'top']:
            if data_type == 'front':
                data = self.front_data
            else:
                data = self.top_data

            ratio = data[1, :, :, :, :] / data[0, :, :, :, :]
            ratio = np.nan_to_num(ratio)
            tcp_temperature = np.zeros_like(ratio, dtype=np.dtype(float))

            for n in range(np.shape(data)[1]):
                for i in range(np.shape(data)[2]):
                    for j in range(np.shape(data)[3]):
                        for k in range(np.shape(data)[4]):
                            if data[0, n, i, j, k] < threshold_fraction * np.max(data[0, n, :, :, :]) \
                                    or data[1, n, i, j, k] < threshold_fraction * np.max(data[1, n, :, :, :]):
                                tcp_temperature[n, i, j, k] = 0
                            elif ratio[n, i, j, k] != 0:
                                tcp_temperature[n, i, j, k] = (c2 * ((1. / lambdaR) - (1. / lambdaG))) / (
                                        np.log(ratio[n, i, j, k]) + np.log((lambdaG / lambdaR) ** 5))

                            if tcp_temperature[n, i, j, k] < 300:
                                tcp_temperature[n, i, j, k] = 300

            # Assign the computed temperatures to the corresponding class variable
            if data_type == 'front':
                self.front_tcp_temperature = tcp_temperature
            else:
                self.top_tcp_temperature = tcp_temperature

    # Get the size of a single mesh.
    # Iterate through the time steps
    # Iterate through each time step and place a point on the plot

    def get_optical_thickness(self, dns_data):
        # Calculate the optical thickness of the frame
        # First get the absorption for each cell in the dns
        dns_temperature, _, _ = dns_data.get_field(self.dns_temperature_name)
        if self.dns_soot is None:
            self.get_dns_soot(dns_data)
        kappa = (3.72 * self.dns_soot * self.C_0 * dns_temperature) / self.C_2  # Soot mean absorption
        # Then sum the absorption through all cells in the ray line
        axis_values = [1, 2]
        optical_thickness_attributes = ['front_dns_optical_thickness', 'top_dns_optical_thickness']

        for axis, attribute in zip(axis_values, optical_thickness_attributes):
            dns_sum_soot = kappa.sum(axis=axis, keepdims=True)
            setattr(self, attribute, dns_sum_soot * dns_data.delta[2 - axis])

    # / (
    # self.end_point[2 - self.tcp_axis] - self.start_point[2 - self.tcp_axis])

    def get_tcp_soot(self):
        lambda_r = 650.e-9

        for data_type in ['front', 'top']:
            axis = 0
            if data_type == 'front':
                axis = 0
            else:
                axis = 1
            path_length = self.end_point[2 - axis] - self.start_point[
                2 - axis]  # TODO: Make sure that the axis selection works correctly.
            tcp_temperature = getattr(self, f"{data_type}_tcp_temperature")
            data = getattr(self, f"{data_type}_data")
            tcp_soot = np.zeros_like(tcp_temperature)

            threshold_condition = (tcp_temperature > 400.0)
            tcp_soot = np.where(
                threshold_condition,
                (-lambda_r * 1.e9 / (self.C_0 * path_length)) * np.log(
                    1 - (data[0, :, :, :, :] * (lambda_r ** 5) * np.exp(
                        self.C_2 / (lambda_r * tcp_temperature))) / self.C_1),
                tcp_soot
            )

            # Assign the computed soot values to the corresponding class variable
            setattr(self, f"{data_type}_tcp_soot", tcp_soot)

    def get_dns_soot(self, dns_data):
        dns_density_yi, _, _ = dns_data.get_field(self.dns_soot_name)
        self.dns_soot = dns_density_yi / self.rhoC

    def plot_temperature_step(self, n, name, data_type='front'):
        # Check if the data_type argument is valid
        if data_type not in ['front', 'top']:
            raise ValueError("data_type must be either 'front' or 'top'.")

        # Get the tcp_temperature if it hasn't been computed already
        tcp_temperature = getattr(self, f"{data_type}_tcp_temperature")
        if tcp_temperature is None:
            self.get_tcp_temperature()  # Calculate the TCP temperature of the given boundary intensities

        tcp_temperature_frame = tcp_temperature[n, :, :, :]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # plot the temperature as a slice in the z direction
        im = ax.imshow(tcp_temperature_frame[0, :, :],
                       interpolation='none', cmap="inferno",
                       origin='lower', extent=[self.start_point[0], self.end_point[0],
                                               self.start_point[1], self.end_point[1]],
                       vmax=4500, vmin=300)
        fig.colorbar(im, shrink=0.5, pad=0.05)
        # ax.clabel(CS, inline=True, fontsize=10)
        # ax.set_title(f'CHREST Format vTCP ({data_type}) - n = {n}')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(r"Temperature $[K]$")  # Add label for the temperature
        # if self.save:
        # plt.savefig(str(name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inchees='tight')
        plt.show()

    def set_limits(self):
        if 'end' in input:
            self.end = input['end']
        else:
            self.end = len(self.front_data[:, 0]) - 1  # Set the end time to the last step by default
        if 'start' in input:
            self.start = input['start']
        else:
            self.start = 0  # Set the start time step to the first by default

    # def rgb_transform(self, delta_t):
    #     self.prf = np.loadtxt("PRF_Color.csv", delimiter=',', skiprows=0)
    #
    #     # Get the correct exposure for the camera by getting the maximum intensity for each channel and shifting to 255
    #     exposure_fraction = 1.0
    #     brightness_max = np.array([-100.0, -100.0, -100.0])
    #     for fieldIndex in range(self.field_size):
    #         for timeStep in range(np.shape(self.front_data)[1]):
    #             for pointIndex in range(np.shape(self.front_data)[2]):
    #                 brightness_transformed = np.log(np.pi * self.front_data[fieldIndex, timeStep, pointIndex] * delta_t)
    #                 if brightness_transformed > brightness_max[fieldIndex]:
    #                     brightness_max[fieldIndex] = brightness_transformed
    #     for fieldIndex in range(self.field_size):
    #         prf_row_max = int(255.0 * exposure_fraction)
    #         shift_constant = self.prf[prf_row_max, fieldIndex] - brightness_max[fieldIndex]
    #
    #     for fieldIndex in range(self.field_size):
    #         for timeStep in range(np.shape(self.front_data)[1]):
    #             for pointIndex in range(np.shape(self.front_data)[2]):
    #                 brightness = 0
    #                 brightness_transformed = np.log(np.pi * self.front_data[fieldIndex, timeStep, pointIndex] * delta_t)
    #                 brightness_transformed += shift_constant
    #
    #                 if np.isinf(brightness_transformed):
    #                     brightness_transformed = 0
    #                 for brightnessIndex in range(np.shape(self.prf)[0]):
    #                     if self.prf[brightnessIndex, fieldIndex] > brightness_transformed:
    #                         brightness = brightnessIndex / 255
    #                         break
    #                 self.rgb[timeStep, pointIndex, fieldIndex] = brightness  # pixel brightness based on camera prf

    # def plot_rgb_step(self, n, name):
    #     rframe = np.vstack(
    #         (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 0]))
    #     rframe = np.transpose(rframe)
    #     r = pd.DataFrame(rframe, columns=['x', 'y', 'r'])
    #     R = r.pivot_table(index='x', columns='y', values=['r']).T.values
    #
    #     gframe = np.vstack(
    #         (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 1]))
    #     gframe = np.transpose(gframe)
    #     g = pd.DataFrame(gframe, columns=['x', 'y', 'g'])
    #     G = g.pivot_table(index='x', columns='y', values=['g']).T.values
    #
    #     bframe = np.vstack(
    #         (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 2]))
    #     bframe = np.transpose(bframe)
    #     b = pd.DataFrame(bframe, columns=['x', 'y', 'b'])
    #     B = b.pivot_table(index='x', columns='y', values=['b']).T.values
    #
    #     X_unique = np.sort(r.x.unique())
    #     Y_unique = np.sort(r.y.unique())
    #     X, Y = np.meshgrid(X_unique, Y_unique)
    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     CS = ax.imshow(np.rot90(np.array([R.data, G.data, B.data]).T, axes=(0, 1)), interpolation='lanczos',
    #                    extent=[rframe[:, 0].min(), rframe[:, 0].max(), rframe[:, 1].min(), rframe[:, 1].max()],
    #                    vmax=abs(R).max(), vmin=-abs(R).max())
    #     ax.set_xlabel("x [m]")
    #     ax.set_ylabel("y [m]")
    #     if self.save:
    #         plt.savefig(self.write_path + "/" + (name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inches='tight')
    #     plt.show()

    def get_uncertainty_field(self, dns_data):
        for orientation in ['front', 'top']:

            axis = 0
            if orientation == 'front':
                axis = 0
            else:
                axis = 1

            tcp_temperature = getattr(self, f"{orientation}_tcp_temperature", None)
            tcp_soot = getattr(self, f"{orientation}_tcp_soot", None)

            # Calculate the TCP temperature and soot of the given boundary intensities
            if tcp_temperature is None:
                self.get_tcp_temperature()
            if tcp_soot is None:
                self.get_tcp_soot()

            tcp_temperature = getattr(self, f"{orientation}_tcp_temperature", None)
            tcp_soot = getattr(self, f"{orientation}_tcp_soot", None)

            # Get the appropriate attribute names
            dns_maximum_temperature_attr = f"{orientation}_dns_maximum_temperature"
            dns_maximum_soot_attr = f"{orientation}_dns_maximum_soot"
            temperature_error_attr = f"{orientation}_temperature_error"
            soot_error_attr = f"{orientation}_soot_error"

            # Fetch the DNS data
            dns_temperature, _, _ = dns_data.get_field(self.dns_temperature_name)
            dns_soot, _, _ = dns_data.get_field(self.dns_soot_name)

            # Compute maximum values along the ray lines
            dns_maximum_temperature = dns_temperature.max(axis=(axis + 1), keepdims=True)
            dns_maximum_soot = dns_soot.max(axis=(axis + 1), keepdims=True)

            # Compute error values
            temperature_error = np.abs(dns_maximum_temperature - tcp_temperature)
            soot_error = np.abs(dns_maximum_soot - tcp_soot)

            # Assign the computed values to the appropriate attributes
            setattr(self, dns_maximum_temperature_attr, dns_maximum_temperature)
            setattr(self, dns_maximum_soot_attr, dns_maximum_soot)
            setattr(self, temperature_error_attr, temperature_error)
            setattr(self, soot_error_attr, soot_error)

    def plot_optical_thickness(self, n, orientation='front'):
        if orientation not in ['front', 'top']:
            raise ValueError("Orientation must be 'front' or 'top'")

        optical_thickness = getattr(self, f"{orientation}_dns_optical_thickness")
        optical_thickness_frame = optical_thickness[n, :, :, :]

        if orientation == 'front':
            x_span = 0
            y_span = 1
        else:
            x_span = 0
            y_span = 2

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # plot the optical thickness as a slice in the z direction
        im = ax.imshow(np.squeeze(optical_thickness_frame),
                       interpolation='none', cmap="inferno",
                       origin='lower', extent=[self.start_point[x_span], self.end_point[x_span],
                                               self.start_point[y_span], self.end_point[y_span]],
                       vmax=optical_thickness_frame.max(), vmin=optical_thickness_frame.min())
        fig.colorbar(im, shrink=0.5, pad=0.05, label=r"Optical Thickness")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.tight_layout()
        if self.save:
            plt.savefig(self.write_path + "/opticalThickness_" + orientation + "." + str(n).zfill(3) + ".png", dpi=1000,
                        bbox_inches='tight')
        plt.show()

    def plot_uncertainty_field(self, n, orientation='front'):
        if orientation not in ['front', 'top']:
            raise ValueError("Orientation must be 'front' or 'top'")

        # Get the appropriate data based on the orientation
        tcp_temperature = getattr(self, f"{orientation}_tcp_temperature")
        dns_maximum_temperature = getattr(self, f"{orientation}_dns_maximum_temperature")
        temperature_error = getattr(self, f"{orientation}_temperature_error")
        tcp_soot = getattr(self, f"{orientation}_tcp_soot")
        dns_maximum_soot = getattr(self, f"{orientation}_dns_maximum_soot")
        soot_error = getattr(self, f"{orientation}_soot_error")

        tcp_temperature_frame = tcp_temperature[n, :, :, :]
        dns_temperature_frame = dns_maximum_temperature[n, :, :, :]
        temperature_error_frame = temperature_error[n, :, :, :]

        tcp_soot_frame = tcp_soot[n, :, :, :]
        dns_soot_frame = dns_maximum_soot[n, :, :, :]
        soot_error_frame = soot_error[n, :, :, :]


        if orientation == 'front':
            x_span = 0
            y_span = 1
        else:
            x_span = 0
            y_span = 2

        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(3, 4, width_ratios=[20, 1, 20, 1], height_ratios=[1, 1, 1])

        # Plot dns_temperature
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np.squeeze(dns_temperature_frame), interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=4500, vmin=300)
        ax1.set_title("DNS Temperature")
        ax1.set_ylabel("y [m]")

        # Plot tcp_temperature
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.imshow(np.squeeze(tcp_temperature_frame[:, :, :]),
                         interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=4500, vmin=300)
        ax2.set_title("TCP Temperature")
        ax2.set_ylabel("y [m]")

        # Plot temperature uncertainty
        ax3 = fig.add_subplot(gs[2, 0])
        im3 = ax3.imshow(np.squeeze(temperature_error_frame), interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=1000, vmin=0)
        ax3.set_title("Temperature Error Field")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")

        # Add colorbar
        cbar_ax1 = fig.add_subplot(gs[0:2, 1])
        cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label="Temperature [K]")
        cbar_ax1.yaxis.set_ticks_position('right')
        cbar_ax1.yaxis.set_label_position('right')

        cbar_ax2 = fig.add_subplot(gs[2, 1])
        cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label="Error [K]")
        cbar_ax2.yaxis.set_ticks_position('right')
        cbar_ax2.yaxis.set_label_position('right')

        # Plot dns_soot
        ax4 = fig.add_subplot(gs[0, 2])
        im4 = ax4.imshow(np.squeeze(dns_soot_frame), interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=tcp_soot_frame.max(), vmin=tcp_soot_frame.min())
        ax4.set_title("DNS Soot")
        ax4.set_ylabel("y [m]")

        # Plot tcp_soot
        ax5 = fig.add_subplot(gs[1, 2])
        im5 = ax5.imshow(np.squeeze(tcp_soot_frame),
                         interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=tcp_soot_frame.max(), vmin=tcp_soot_frame.min())
        ax5.set_title("TCP Soot")
        ax5.set_ylabel("y [m]")

        # Plot soot uncertainty
        ax6 = fig.add_subplot(gs[2, 2])
        im6 = ax6.imshow(np.squeeze(soot_error_frame), interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[x_span], self.end_point[x_span], self.start_point[y_span], self.end_point[y_span]],
                         vmax=soot_error_frame.max(), vmin=soot_error_frame.min())
        ax6.set_title("Soot Error Field")
        ax6.set_xlabel("x [m]")
        ax6.set_ylabel("y [m]")

        # Add colorbar
        cbar_ax4 = fig.add_subplot(gs[0:2, 3])
        cbar3 = fig.colorbar(im4, cax=cbar_ax4, orientation='vertical', label="Soot Volume Fraction")
        cbar_ax4.yaxis.set_ticks_position('right')
        cbar_ax4.yaxis.set_label_position('right')

        cbar_ax5 = fig.add_subplot(gs[2, 3])
        cbar4 = fig.colorbar(im6, cax=cbar_ax5, orientation='vertical', label="Soot Error")
        cbar_ax5.yaxis.set_ticks_position('right')
        cbar_ax5.yaxis.set_label_position('right')

        plt.tight_layout()
        if self.save:
            plt.savefig(self.write_path + "/uncertaintyField_" + orientation + "." + str(n).zfill(3) + ".png", dpi=1000,
                        bbox_inches='tight')
        plt.show()

    def plot_line_of_sight(self, n, data_3d, orientation='front'):
        if orientation not in ['front', 'top']:
            raise ValueError("orientation should be either 'front' or 'top'")

        fig, ax = plt.subplots(3, 1, figsize=(10, 15))

        # Create an array for the x-axis.
        x = data_3d.get_coordinates()

        dns_temperature, _, _ = data_3d.get_field(self.dns_temperature_name)

        # Dynamically set the attribute names based on orientation
        tcp_temperature_attr = f"{orientation}_tcp_temperature"
        tcp_soot_attr = f"{orientation}_tcp_soot"
        dns_optical_thickness_attr = f"{orientation}_dns_optical_thickness"

        # Fetch the TCP temperature and soot if not already fetched
        if getattr(self, tcp_temperature_attr) is None:
            self.get_tcp_temperature()
        if getattr(self, tcp_soot_attr) is None:
            self.get_tcp_soot()

        # Fetch the DNS soot if not already fetched
        if self.dns_soot is None:
            self.get_dns_soot(data_3d)

        # Fetch optical thickness if not already fetched
        if getattr(self, dns_optical_thickness_attr) is None:
            self.get_optical_thickness(data_3d)

        # Define data for each subplot
        plots_data = [
            {
                'y': dns_temperature[n, :, :, :],
                'projected_y': getattr(self, tcp_temperature_attr),
                'ylabel': "Temperature [K]",
                'label': f"{orientation} TCP Temperature",
            },
            {
                'y': self.dns_soot[n, :, :, :],
                'projected_y': getattr(self, tcp_soot_attr),
                'ylabel': "Soot Volume Fraction",
                'label': f"{orientation} TCP Soot Volume Fraction",
            },
            {
                'y': (3.72 * self.dns_soot[n, :, :, :] * self.C_0 * dns_temperature[n, :, :, :]) / self.C_2,
                'projected_y': getattr(self, dns_optical_thickness_attr)[n, :, :, :] / (
                        self.end_point[2 - self.tcp_axis] - self.start_point[2 - self.tcp_axis]),
                'ylabel': "Absorption Coefficient",
                'label': f"{orientation} DNS Absorption Coefficient | Mean Optical Thickness: " + str(
                    getattr(self, dns_optical_thickness_attr).mean()),
            }
        ]

        axis = 0
        if orientation == 'front':
            axis = 0
        else:
            axis = 1
        # Iterate over data and axes to plot each subplot
        for i, data in enumerate(plots_data):
            ax[i].scatter(x[:, :, :, 2 - axis], data['y'], color='k', marker='.', label=data['label'])
            ax[i].axhline(y=data['projected_y'].mean(), color='r', linestyle='-',
                          label='Mean ' + data['label'])
            ax[i].set_title(data['label'] + " along the line of sight")
            ax[i].set_ylabel(data['ylabel'])
            ax[i].legend()
            ax[i].set_ylim(bottom=0)

        # Set x-label for the last plot
        ax[2].set_xlabel("Position along the line of sight")

        plt.tight_layout()

        if self.save:
            plt.savefig(self.write_path + "/" + "line_of_sight" + ".png", dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_path', type=str, required=True,
                        help='The path to the YAML input file.')
    args = parser.parse_args()

    # Load configuration from YAML filex
    with open(args.input_path, 'r') as stream:
        try:
            input = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if 'deltaT' not in input:
        input['deltaT'] = 0.004

    if 'write_path' not in input:
        write_path = input['base_path'] + "/figures"
    else:
        write_path = input['write_path']

    if 'save' not in input:
        save = False
    else:
        save = input['save']

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    vTCP = VTcpData(input['front_tcp'], input['top_tcp'], input['fields'], input['tcp_axis'], input['base_path'],
                    write_path,
                    input['dns_temperature_name'], input['dns_soot_name'],
                    save)

    print(len(vTCP.front_data[0, :, 0]))

    data_3d = ChrestData(input['base_path'] + "/" + input['dns'])
    vTCP.get_uncertainty_field(data_3d)
    vTCP.get_optical_thickness(data_3d)

    vTCP.plot_uncertainty_field(50, orientation='front')
    vTCP.plot_uncertainty_field(50, orientation='top')
    vTCP.plot_optical_thickness(50, orientation='front')
    vTCP.plot_optical_thickness(50, orientation='top')
    vTCP.plot_line_of_sight(50, data_3d, orientation='front')
    vTCP.plot_line_of_sight(50, data_3d, orientation='top')

    print('Done')
