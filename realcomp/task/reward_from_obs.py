import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import cv2
import scipy.ndimage as ndi


def euclidean_distance(x, y):
    if isinstance(x, tuple):
        s = 0
        for i in range(len(x)):
            s += (x[i] - y[i]) ** 2
        return np.sqrt(s)
        
    if len(x.shape) <= 1:
        axis = 0
    else:
        axis = 1

    return np.sqrt(np.sum((x - y) ** 2, axis=axis))


def gray_image_from_obs(observation):
    image = observation["retina"]
    image = Image.fromarray(image)
    image = image.convert('LA')
    image = np.array(image)[:, :, 0] / 255.
    return image

def kernelize(image_1, image_2, size=7.0):

    diff_image = abs(image_1 - image_2)
    smoothed_diff = ndi.filters.gaussian_filter(diff_image, size)
            
    return diff_image, smoothed_diff

def mask_noise(image, threshold=0.01):
    image[image < threshold] = 0
    return image

def subsample(image, x=24, y=32):
    image = Image.fromarray(image)
    image = image.resize((y, x))
    image = np.array(image) / 255.
    return image

def contours(image, value=0.3):
    labeled_image, nb_zones = ndi.measurements.label(image)
    if nb_zones >= 1:
        centers = ndi.measurements.center_of_mass(image, labeled_image, range(1, nb_zones + 1))
                        
    else:
        centers = []

    radius = [0 for i in range(nb_zones)]
    size = [0 for i in range(nb_zones)]

    for label in range(1, nb_zones + 1):
        for i in range(len(labeled_image)):
            for j in range(len(labeled_image[i])):
                px = labeled_image[i][j]
                if px == label:
                    distance = euclidean_distance((i, j), centers[label - 1])
                    size[label - 1] += 1

                    if distance > radius[label - 1]:
                        radius[label - 1] = distance
                    
    return labeled_image, centers, radius, size


def compute_reward(image_1, image_2, threshold_mask=0.1):
    diff, smoothed_diff = kernelize(image_1, image_2)
    smoothed_diff = mask_noise(smoothed_diff, threshold_mask)
    labeled_image, centers, radius, size = contours(smoothed_diff)

    if len(centers) == 0:
        return 0
    elif len(centers) == 1:
        return - size[0]
    elif len(centers) >= 2: #Hope for the best ! If more than 2 zones : can't deal with it atm
        tot_size = size[0] + size[1]
        dist_clusters = euclidean_distance(centers[0], centers[1])
        dist_penalty = dist_clusters / (radius[0] + radius[1])

        return -(tot_size * dist_penalty)
