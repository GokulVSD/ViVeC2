import numpy as np
import torch

# 9 bin (signed) magnitude-weighted gradient histograms.
# Signed implies we consider 360 degrees, hence 9 bins => 40 degree bins.
# https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
# https://medium.com/@skillcate/histogram-of-oriented-gradients-hog-simplest-intuition-2392995f8010

def get_hog_vector(gs_grid):

    g_mags, g_dirs = get_gradients(gs_grid)

    gradient_bins = get_gradient_bins(g_mags, g_dirs)

    return get_hog_vector_from_bins(gradient_bins)



def get_hog_vector_from_bins(gradient_bins):

    hog_descriptor = []

    for i in range(len(gradient_bins)):
        for j in range(len(gradient_bins[0])):

            bins = gradient_bins[i][j]

            hog_descriptor.extend(bins)

    # Represent as a 900-dim torch vector.
    # The sequential vector representation is as follows:
    #
    # Row-major order of cells (10x10 cells), with each cell having 9 values, one each
    # for binned degrees: 0, 40, 80, 120, 160, 200, 240, 280, 320.
    # 0 is the same as 360.
    # The magnitude of the gradient is distributed according to its direction,
    # with the adjacent bins getting the magnitude weighted according to how
    # close the direction is to the bin.
    #
    # vec: [<cell_1 <0, 40, 80, 120 ... 320>>, <cell_2 ...>...]

    return torch.tensor(hog_descriptor)



def get_gradient_bins(g_mags, g_dirs):

    gradient_bins = []

    for i in range(len(g_mags)):

        gradient_bins.append([])

        for j in range(len(g_mags[0])):

            g_mag = g_mags[i][j]
            g_dir = g_dirs[i][j]

            # We accomplish magnitude-weighted binning via the following strategy:
            #
            # pixel(x, y) has magnitude 72, direction 50. Since the adjacent bins
            # are at 40 and 80 degrees, 72 * |80 - 50|/40 = 54 is added to bin 40,
            # and 72 * |40 - 50|/40 = 18 is added to bin 80.
            # (direction 50 is closer to 40 than to 80, so gets a larger share)
            #
            # Doing this in essense increases the smoothness of the histogram.

            bins = [0] * 9
            # index 0 => 0 deg (and also 360 deg), index 8 => 320

            for row in range(len(g_mag)):
                for col in range(len(g_mag[0])):

                    low_bin_idx = int(np.floor(g_dir[row][col] / 40))
                    high_bin_idx = int(np.ceil(g_dir[row][col] / 40))

                    low_bin_deg = low_bin_idx * 40
                    high_bin_deg = high_bin_idx * 40

                    # Handle case when degree is > 320.
                    high_bin_idx = 0 if high_bin_idx == 9 else high_bin_idx

                    bins[low_bin_idx] += g_mag[row][col] * (abs(high_bin_deg - g_dir[row][col]) / 40)
                    bins[high_bin_idx] += g_mag[row][col] * (abs(low_bin_deg - g_dir[row][col]) / 40)

            gradient_bins[-1].append(bins)

    return gradient_bins



def get_gradients(gs_grid):

    g_mask = np.array([-1, 0, 1])

    g_mags = []
    g_dirs = []


    for i in range(len(gs_grid)):

        g_mags.append([])
        g_dirs.append([])

        for j in range(len(gs_grid[0])):

            cell = gs_grid[i][j][0]

            g_x = []
            g_y = []

            # Prior to calculating dI/dx and dI/dy, we 0 pad the patch so as to maintain
            # resolution. The assumption here is that the contributions to the bins by each
            # of these edge pixels is negligible relative to the rest of the pixels.
            # This is accomplished by using np.correlate with mode 'same'.
            #
            # NOTE: np.correlate is not np.convolve. np.correlate merely moves the kernel
            # over the image (which gives us the gradient).

            for row in cell:
                g_x.append(np.correlate(row, g_mask, mode='same'))

            for column in np.transpose(cell): # Transpose to compute vertical gradient horizontally.
                g_y.append(np.correlate(column, g_mask, mode='same'))

            g_x = np.array(g_x)
            g_y = np.transpose(np.array(g_y)) # Transpose back so that it is vertical.

            # Compute magnitudes: sqrt(g_x^2 + g_y^2)
            g_x2 = np.power(g_x, 2)
            g_y2 = np.power(g_y, 2)
            g_mag = np.sqrt(g_x2 + g_y2)

            # Compute directions: atan(g_y/g_x)
            # We use np.arctan2 to correctly select quadrant.
            g_dir = np.arctan2(g_y, g_x)
            # We do modulo 360 to so as to make -45 deg => 315 deg
            g_dir = np.rad2deg(g_dir) % 360

            g_mags[-1].append(g_mag)
            g_dirs[-1].append(g_dir)

    return g_mags, g_dirs