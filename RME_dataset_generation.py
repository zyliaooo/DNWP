"""
RME data generation
"""
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.io as scio

save_data_path = r"./XXX/XXX"


def create_dataset_element(base_size, end_size, magnitude_min, magnitude_max):
    ndarray = np.random.rand(base_size, base_size, base_size // 4)
    coef = np.random.permutation(np.arange(magnitude_min, magnitude_max, 0.1))[0]
    # element = cv2.resize(array, dsize=(end_size, end_size, end_size), interpolation=cv2.INTER_CUBIC)
    element = torch.from_numpy(ndarray).unsqueeze(0).unsqueeze(0)
    element = F.interpolate(element, scale_factor=end_size // base_size, mode='trilinear', align_corners=True)
    element = element.numpy().squeeze() * coef
    if np.min(element) >= 0:
        min_value = np.min(element)
        element = element - min_value
    else:
        min_value = np.min(element)
        element = element + abs(min_value)
    return element


def make_gaussian(number_of_gaussians, sigma_min, sigma_max, shift_max, magnitude_min, magnitude_max, end_size):
    element = np.zeros([end_size, end_size, end_size // 4])
    value = 3.1415926535897932384626433
    x = np.arange(-value, value, 2 * value / end_size)
    y = np.arange(-value, value, 2 * value / end_size)
    z = np.arange(-value, value, 2 * value / (end_size / 4))
    xx, yy, zz = np.meshgrid(x, y, z)

    for _ in range(number_of_gaussians):
        sigma = np.random.permutation(np.arange(sigma_min, sigma_max, .1))[0]
        shift_x = np.random.permutation(np.arange(0, shift_max, 1))[0]
        shift_y = np.random.permutation(np.arange(0, shift_max, 1))[0]
        shift_z = 0
        magnitude = np.random.permutation(np.arange(magnitude_min, magnitude_max, .1))[0]

        d = np.sqrt((xx - shift_x) ** 2 + (yy - shift_y) ** 2 + (zz - shift_z) ** 2)
        element += 1 / (np.sqrt(2 * pi) * sigma) * np.exp(-(d ** 2 / (2.0 * sigma ** 2)))  # Gaussian density

        element = element / np.max(element)
        element = element * magnitude

    if np.min(element) >= 0:
        min_value = np.min(element)
        element = element - min_value
    else:
        min_value = np.min(element)
        element = element + abs(min_value)

    return element


def to_wrap(image):
    while np.max(image) > pi:
        image = np.where(image > pi, image - 2 * pi, image)
    while np.min(image) <= -pi:
        image = np.where(image <= -pi, image + 2 * pi, image)

    return image


if __name__ == "__main__":
    count = 0
    base_size_max = 17  # data size for
    end_size = 128  # data size for gaussian
    SampleNum = 8000
    all_max = 0

    dataset = np.empty([end_size, end_size, end_size // 4])
    wrap = np.empty([end_size, end_size, end_size // 4])
    alist = []
    for j in range(base_size_max):
        if j > 4 and 128 % j == 0:
            alist.append(j)
    print(f"the number less than base size can be {alist}, length is {len(alist)}")

    while count < SampleNum:
        size = np.random.permutation(alist)[0]
        dataset = create_dataset_element(base_size=size, end_size=128, magnitude_min=4, magnitude_max=20)
        data = to_wrap(dataset)
        wrap = (dataset - data) / (2 * pi)
        max_wrap = np.max(wrap)
        all_max = max(max_wrap, all_max)
        print(f'max_wrap is {max_wrap}, all_max is {all_max}')

        # Add noise
        # scale = np.random.permutation(np.arange(0.01, 0.2, 1))[0]
        # noise = np.random.normal(loc=0, scale=scale, size=[128, 128, 64])
        # data = data + noise

        count += 1
        scio.savemat(save_data_path + f"/M{count}" + '.mat', {'data': data,
                                                              'label': dataset})

    # dataset = np.empty([end_size, end_size, end_size // 4])
    # wrap = np.empty([end_size, end_size, end_size // 4])
    # while count < SampleNum:
    #     num_gauss = np.random.permutation(np.arange(1, 10, 1))[0]  # gauss number
    #     print(f"the gaussian number is {num_gauss}")
    #     dataset = make_gaussian(
    #         number_of_gaussians=num_gauss,
    #         sigma_min=1,
    #         sigma_max=4,
    #         shift_max=4,
    #         magnitude_min=4,
    #         magnitude_max=20,
    #         end_size=end_size)
    #     data = to_wrap(dataset)
    #     wrap = (dataset - data) / (2 * pi)
    #     # print(f"wrap max is {np.max(wrap)} and wrap min is {np.min(wrap)}")
    #     max_wrap = np.max(wrap)
    #     all_max = max(max_wrap, all_max)
    #     print(f'max_wrap is {max_wrap}, all_max is {all_max}')
    #
    #     # Add noise
    #     # scale = np.random.permutation(np.arange(0.01, 0.2, 1))[0]
    #     # noise = np.random.normal(loc=0, scale=scale, size=[128, 128, 32])
    #     # data = data + noise
    #
    #     count += 1
    #     scio.savemat(save_data_path + f"/G{count}" + '.mat', {'data': data,
    #                                                           'label': dataset})

        # plot unwrapped and wrapped data
        # plt.imshow(data[:, :, 16], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(dataset[:, :, 16], cmap='gray')
        # plt.colorbar()
        # plt.show()
