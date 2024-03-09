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


def to_mtx(img):
    H, V, C = img.shape
    mtr = np.zeros((V, H, C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]

    return mtr


def to_img(mtr):
    V, H, C = mtr.shape
    img = np.zeros((H, V, C), dtype='int')
    for i in range(mtr.shape[0]):
        img[:, i] = mtr[i]

    return img

def appliqueTransformation(img, H):
    Hinv = np.linalg.inv(H)
    Rows, Cols = img.shape[:2]
    # dest = cv2.warpAffine(img, Hinv, (Rows, Cols))
    dest = np.zeros((Rows, Cols, img.shape[2]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res = np.dot(Hinv, [i, j, 1])
            i2, j2, _ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < Rows:
                if j2 >= 0 and j2 < Cols:
                    dest[i2, j2] = img[i, j]

    return dest

def morph(img, H):
    Hinv = np.linalg.inv(H)
    # Dimensions of the output image
    rows, cols, _ = img.shape

    # Create an output image initialized with zeros
    result = np.zeros_like(img)

    # Loop over each pixel in the output image
    for i in range(rows):
        for j in range(cols):
            # Apply inverse perspective transformation to get corresponding pixel in img
            xw, yw, w = Hinv.dot(np.array([j, i, 1]))
            x = int(round(xw / w))
            y = int(round(yw / w))

            # Check if the pixel is within the bounds of img
            if x >= 0 and x < cols and y >= 0 and y < rows:
                # Use bilinear interpolation to get the pixel value from img
                result[i, j] = img[y, x]

    return result




#Homographie à tester avec la fonction appliqueTransformation()
H1 = np.array([[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]])
H2 = np.array([[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])

#Charger les images
im1 = imread('./tp4/images/0-Rechauffement/pouliot.jpg')
im1 = skimage.util.img_as_float(im1)
res1 = copy.deepcopy(im1)

print(im1.shape)
result = morph(im1, H2)

plt.imshow(result)
plt.axis('off')
plt.title('Tete')
plt.show()

# fname = 'result/premier.jpg'
# img_RGB_ubyte = sk.util.img_as_ubyte(result)
# skio.imsave(fname, img_RGB_ubyte)