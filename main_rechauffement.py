import copy
import skimage.util
from imageio import imread
import matplotlib.pyplot as plt
import skimage.transform as skt
import skimage as sk
import skimage.io as skio
import subprocess
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from PIL import Image, ImageDraw
from skimage.transform import resize


def get_point_coor(x, y, H):
    input = np.array(([x], [y], [1]))
    output = np.dot(H, input)
    return int(output[0] / output[2]), int(output[1] / output[2])


def appliqueTransformation(img, H):
    left, up = 0, 0
    right, down = img.shape[1], img.shape[0]
    Hinv = np.linalg.inv(H)

    # after transformation the image size might be different from the original one,
    # we need to find the new size
    height_max = max(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                     get_point_coor(right, down, H)[0])
    width_max = max(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                    get_point_coor(right, down, H)[1])
    height_min = min(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                     get_point_coor(right, down, H)[0])
    width_min = min(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                    get_point_coor(right, down, H)[1])
    print(height_max, width_max, height_min, width_min)

    new_height = abs(height_max) + abs(height_min)
    new_width = abs(width_max) + abs(width_min)

    result = np.zeros((new_height, new_width, img.shape[2]))

    # Loop over each pixel in the output image
    for i in range(new_height):
        for j in range(new_width):
            # Apply inverse perspective transformation to get corresponding pixel in img
            xw, yw, w = Hinv.dot(np.array([i + height_min, j + width_min, 1]))
            x = int(round(xw / w))
            y = int(round(yw / w))

            # Check if the pixel is within the bounds of img
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                # Use bilinear interpolation to get the pixel value from img
                result[i, j] = img[y, x]

    #Correction manuelle des problème de transformation (rotation + effet mirroir)
    result = np.rot90(result, k=3)
    # Get the number of columns
    num_cols = result.shape[1]
    half_cols = num_cols // 2

    for i in range(half_cols):
        result[:, i], result[:, num_cols - i - 1] = result[:, num_cols - i - 1].copy(), result[:, i].copy()

    return result

def appliqueTransformation2(img, H):
    left, up = 0, 0
    right, down = img.shape[1], img.shape[0]
    Hinv = np.linalg.inv(H)

    # after transformation the image size might be different from the original one,
    # we need to find the new size
    height_max = max(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                     get_point_coor(right, down, H)[1])
    width_max = max(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                    get_point_coor(right, down, H)[0])
    height_min = min(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                     get_point_coor(right, down, H)[1])
    width_min = min(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                    get_point_coor(right, down, H)[0])

    minmax = [height_max, width_max, height_min, width_min]
    print(minmax)
    if height_min < 0 and width_min < 0:
        new_height = abs(height_max) + abs(height_min)
        new_width = abs(width_max) + abs(width_min)
    elif height_min < 0 < width_min:
        new_height = abs(height_max) + abs(height_min)
        new_width = abs(width_max) - abs(width_min)
    elif width_min < 0 < height_min:
        new_height = abs(height_max) - abs(height_min)
        new_width = abs(width_max) + abs(width_min)
    elif width_min > 0 < height_min:
        new_height = abs(height_max) - abs(height_min)
        new_width = abs(width_max) - abs(width_min)
    else:
        new_height = abs(height_max)
        new_width = abs(width_max)

    imgTrans = np.zeros((new_height, new_width, img.shape[2]))

    # Loop over each pixel in the output image
    for i in range(new_width):
        for j in range(new_height):
            # Apply inverse perspective transformation to get corresponding pixel in img
            xw, yw, w = Hinv.dot(np.array(([i + width_min], [j + height_min], [1])))
            x = int(round(xw[0] / w[0]))
            y = int(round(yw[0] / w[0]))

            # Check if the pixel is within the bounds of img
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                # Use bilinear interpolation to get the pixel value from img
                imgTrans[j, i] = img[y, x]

    return imgTrans


#Homographie à tester avec la fonction appliqueTransformation()
H1 = np.array([[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]])
H2 = np.array([[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])

#Charger les images
im1 = imread('./images/0-Rechauffement/pouliot.jpg')
im1 = skimage.util.img_as_float(im1)
res1 = copy.deepcopy(im1)

print(im1.shape)
result = appliqueTransformation2(im1, H2)

plt.imshow(result)
plt.axis('off')
plt.title('Tete')
plt.show()

# fname = 'result/premier.jpg'
# img_RGB_ubyte = sk.util.img_as_ubyte(result)
# skio.imsave(fname, img_RGB_ubyte)