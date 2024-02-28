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

# im1 = imread('../tp3/toast.jpg')
# im1 = skimage.util.img_as_float(im1)

# Utiliser pour redimensionner les photos prise en ligne
# im2 = resize(im2, im1.shape[:2], anti_aliasing=True)
# mask = resize(mask, im1.shape[:2], anti_aliasing=True)

# Utiliser pour rotationner les photos prise sur telephone
# im1 = np.transpose(im1)
# im1 = im1[:, ::-1]

# Reduction de la taille de l'image
# im1 = skt.resize(im1, (720, 720), anti_aliasing=True)

# img_RGB_ubyte = sk.util.img_as_ubyte(im1)

# fname = 'toast720.jpg'
# skio.imsave(fname, img_RGB_ubyte)

# plt.imshow(im1)
# plt.axis('off')
# plt.title('Tete')
# plt.show()

def affine_transform(img1_pts, dest_pts, tri):
    A = np.array([[img1_pts[tri[0]][0], img1_pts[tri[0]][1], 1, 0, 0, 0],
                  [0, 0, 0, img1_pts[tri[0]][0], img1_pts[tri[0]][1], 1],
                  [img1_pts[tri[1]][0], img1_pts[tri[1]][1], 1, 0, 0, 0],
                  [0, 0, 0, img1_pts[tri[1]][0], img1_pts[tri[1]][1], 1],
                  [img1_pts[tri[2]][0], img1_pts[tri[2]][1], 1, 0, 0, 0],
                  [0, 0, 0, img1_pts[tri[2]][0], img1_pts[tri[2]][1], 1]])

    b = np.array([dest_pts[0][0], dest_pts[0][1], dest_pts[1][0], dest_pts[1][1], dest_pts[2][0], dest_pts[2][1]])

    x = np.linalg.solve(A, b)

    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [0, 0, 1]])

def getPoints(tri):
    max_x = max(point[0] for point in tri) + 2
    max_y = max(point[1] for point in tri) + 2
    width = round(max_x)
    height = round(max_y)
    mask = Image.new('P', (width, height), 0)
    ImageDraw.Draw(mask).polygon(tuple(map(tuple, tri)), outline=255, fill=255)
    coordArray = np.transpose(np.nonzero(mask))

    return coordArray

def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac, dissolve_frac):
    for triangle in tri:
        target_tri = []
        for sommet in range(3):
            target_tri.append((1 - warp_frac) * img1_pts[triangle[sommet]] + (warp_frac * img2_pts[triangle[sommet]]))

        im1_matrice_affine = affine_transform(img1_pts, target_tri, triangle)
        im1_invmatrice_affine = np.linalg.inv(im1_matrice_affine)
        im2_matrice_affine = affine_transform(img2_pts, target_tri, triangle)
        im2_invmatrice_affine = np.linalg.inv(im2_matrice_affine)

        target_pts = getPoints(target_tri)
        xp, yp = np.transpose(target_pts)

        im1XValues = im1_invmatrice_affine[1, 1] * xp + im1_invmatrice_affine[1, 0] * yp + im1_invmatrice_affine[1, 2]
        im1YValues = im1_invmatrice_affine[0, 1] * xp + im1_invmatrice_affine[0, 0] * yp + im1_invmatrice_affine[0, 2]
        im1YParam = np.arange(min([img1_pts[triangle[0]][0], img1_pts[triangle[1]][0], img1_pts[triangle[2]][0]]),
                               max([img1_pts[triangle[0]][0], img1_pts[triangle[1]][0], img1_pts[triangle[2]][0]]), 1)
        im1XParam = np.arange(min([img1_pts[triangle[0]][1], img1_pts[triangle[1]][1], img1_pts[triangle[2]][1]]),
                               max([img1_pts[triangle[0]][1], img1_pts[triangle[1]][1], img1_pts[triangle[2]][1]]), 1)
        Image1Values = img1[int(im1XParam[0]):int(im1XParam[-1] + 1), int(im1YParam[0]):int(im1YParam[-1] + 1)]

        im2XValues = im2_invmatrice_affine[1, 1] * xp + im2_invmatrice_affine[1, 0] * yp + im2_invmatrice_affine[1, 2]
        im2YValues = im2_invmatrice_affine[0, 1] * xp + im2_invmatrice_affine[0, 0] * yp + im2_invmatrice_affine[0, 2]
        im2YParam = np.arange(min([img2_pts[triangle[0]][0], img2_pts[triangle[1]][0], img2_pts[triangle[2]][0]]),
                               max([img2_pts[triangle[0]][0], img2_pts[triangle[1]][0], img2_pts[triangle[2]][0]]), 1)
        im2XParam = np.arange(min([img2_pts[triangle[0]][1], img2_pts[triangle[1]][1], img2_pts[triangle[2]][1]]),
                               max([img2_pts[triangle[0]][1], img2_pts[triangle[1]][1], img2_pts[triangle[2]][1]]), 1)
        Image2Values = img2[int(im2XParam[0]):int(im2XParam[-1] + 1), int(im2YParam[0]):int(im2YParam[-1] + 1)]

        for z in range(3):
            res1[xp, yp, z] = RectBivariateSpline(im1XParam, im1YParam, Image1Values[:, :, z], kx=1, ky=1).ev(im1XValues, im1YValues)
            res2[xp, yp, z] = RectBivariateSpline(im2XParam, im2YParam, Image2Values[:, :, z], kx=1, ky=1).ev(im2XValues, im2YValues)

    return (1-dissolve_frac) * res1 + dissolve_frac * res2

#Charger les images
im1 = imread('pain720.jpg')
im1 = skimage.util.img_as_float(im1)
im2 = imread('toast720.jpg')
im2 = skimage.util.img_as_float(im2)
res1 = copy.deepcopy(im1)
res2 = copy.deepcopy(im2)

#Charger les points des fichiers text
pts1 = np.loadtxt('pain.txt', delimiter='\t', usecols=(1,2))
pts2 = np.loadtxt('toast.txt', delimiter='\t', usecols=(1,2))

#on ajoute des points sur les coins de l'image pour avoir un rendu parfait
coin = np.array([[1, 1], [718, 1], [718, 718], [1, 718]])
pts1 = np.concatenate((pts1, coin))
pts2 = np.concatenate((pts2, coin))

pts_moy = (pts1 + pts2)/2.0

tri = Delaunay(pts_moy)
n = 100
warp_frac = 1/n
dissolve_frac = 1/n

for r in range(n-1):
    result = morph(im1, im2, pts1, pts2, tri.simplices, warp_frac, dissolve_frac)
    warp_frac += 1/n
    dissolve_frac += 1/n
    fname = f'image/file_{r:05d}.png'
    img_RGB_ubyte = sk.util.img_as_ubyte(result)
    skio.imsave(fname, img_RGB_ubyte)