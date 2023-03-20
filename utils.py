import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns


def get_local_min(array: np.ndarray, i: int, j: int, delta: int = 80) -> int | float:
    """
    :param array: 2d array of values
    :param i: index
    :param j: inedx
    :param delta: is half of border of local square where min in searching
    :return:
    """
    min_i = max(i - delta, 0)
    max_i = min(i + delta + 1, array.shape[0])
    min_j = max(j - delta, 0)
    max_j = min(j + delta + 1, array.shape[1])
    local_area = array[min_i:max_i, min_j:max_j]
    return np.min(local_area[local_area != 0])


def find_nearest_zero_point(arr: np.ndarray, i: int, j: int) -> tuple[int, int] | None:
    """

    :param arr:
    :param i:
    :param j:
    :return:
    """
    zero_points = np.argwhere(arr == 0)
    if len(zero_points) == 0:
        return None
    distances = np.linalg.norm(zero_points - [i, j], axis=1)
    nearest_point_index = np.argmin(distances)
    nearest_point = zero_points[nearest_point_index]
    return nearest_point[0], nearest_point[1]


def get_array_from_path(path_to_tiff: str) -> np.ndarray:
    """
    Reads tif file ana returns numpy ndarray
    Args:
        path_to_tiff:

    Returns: tif array

    """
    tif_image = Image.open(path_to_tiff)
    return np.array(tif_image)


def create_mask(array: np.ndarray) -> np.ndarray:
    """
    Takes array where elements >= 0 is a buildings and <= are background
    And creates a binary mask (1 - buildings, 0 - other)
    Args:
        array:

    Returns:

    """
    new_arr = np.copy(array)
    new_arr[new_arr <= 0] = 0
    new_arr[new_arr >= 1] = 1
    return new_arr


def clear_background(array: np.ndarray, mask_heights: np.ndarray):
    """
    Takes a height array and create height array where mask=0 are zeros
    Args:
        array:
        mask_heights:

    Returns:

    """
    new_array = np.copy(array)
    new_array[mask_heights == 0] = 0
    return new_array


def plot_images(*images, nrows=1, ncols=None, figsize=None):
    """
    Plots an arbitrary number of images in a grid.

    Parameters:
        *images (numpy.ndarray): One or more numpy arrays representing the images to plot.
        nrows (int): Number of rows in the grid. Default is 1.
        ncols (int): Number of columns in the grid. If not provided, it will be calculated based on the number
        of images.
        figsize (tuple): Size of the figure (width, height) in inches. If not provided, the default size will be used.
    """
    # Calculate the number of columns if not provided
    if ncols is None:
        ncols = len(images) // nrows + (len(images) % nrows > 0)

    # Create a new figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Flatten the axes if necessary
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.ravel()
    else:
        axes = axes.flatten()
    # Plot each image
    for i, image in enumerate(images):
        ax = axes[i]
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.colorbar()

    # Adjust the spacing and show the plot
    plt.tight_layout()
    plt.show()


def clear_lidar(path_all_heights: str, path_to_buildings_heights: str, plot=True) -> np.ndarray:
    tif1_array = get_array_from_path(path_all_heights)[:, :3084]
    tif2_array = get_array_from_path(path_to_buildings_heights)

    if tif1_array.shape != tif2_array.shape:
        raise ArithmeticError(f'Both tiffs must be the same shape. Gotten: {tif1_array.shape}, {tif2_array.shape}')

    if plot:
        plot_images(tif1_array, tif2_array)
    plt.imsave('input_all_heights.png', tif1_array)
    plt.imsave('input_buildings_heights.png', tif2_array)
    mask = create_mask(tif2_array)
    # if plot:
    #    plot_images(mask)

    new_tif1 = np.copy(tif1_array)
    new_tif1[mask == 1] = 0

    print('All surface heights: ', np.unique(new_tif1))  #
    # if plot:
    #    plot_images(new_tif1)

    new_tif1_1 = np.copy(tif1_array)
    new_tif1_1[mask == 0] = 0

    print('Buildings heights: ', np.unique(new_tif1_1))  #

    heights = new_tif1_1
    heights[heights < 1] = np.unique(heights)[1]

    print('Heights before work: ', np.unique(heights)[:5], np.unique(heights)[-5:])
    print('Shape: ', heights.shape)

    print(np.count_nonzero(heights == 247.42))

    heights_copy = np.copy(heights)
    for i in range(heights.shape[0]):
        for j in range(heights.shape[1]):
            if mask[i][j] == 1:
                heights[i][j] = heights[i][j] - get_local_min(heights_copy, i, j)
            else:
                heights[i][j] = 0
    # if plot:
    #    plot_images(heights)

    print("Tail and head of unique Heights values: ")
    print(np.unique(heights)[:5], np.unique(heights)[-5:])
    print(np.count_nonzero(heights == 247.42))
    if plot:
        sns.heatmap(heights)
        plt.show()

    plt.imsave('output.png', heights)
    return heights


clear_lidar('test_images/1_232.tif', 'test_images/1_232_Здание.tif')