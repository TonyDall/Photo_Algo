#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TP1, code python pour débuter

# quelques librairies suggérées
# vous pourriez aussi utiliser matplotlib et opencv pour lire, afficher et sauvegarder des images

import numpy as np
import skimage as sk
import skimage.io as skio

# nom du fichier d'image
imname = 'images/00907v.jpg'

# lire l'image
im = skio.imread(imname)

# conversion en double
im = sk.util.img_as_float(im)

# calculer la hauteur de chaque partie (1/3 de la taille de l'image)
height = int(np.floor(im.shape[0] / 3.0))

# séparer les canaux de couleur
b = im[:height]
g = im[height: 2 * height]
r = im[2 * height: 3 * height]

# Spécifier le nombre de pixel à retirer du contour de l'image pour le calcul SSD
crop_pixels = 50

# aligner les images... c'est ici que vous commencez à coder!
# ces quelques fonctions pourraient vous être utiles:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(image_to_align, reference_image, max_shift=15):
    best_score = np.inf
    best_shift = (0, 0)
    ref_cropped = reference_image[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_image = np.roll(image_to_align, (dy, dx), axis=(0, 1))
            shifted_im_cropped = shifted_image[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
            score = np.sum(((shifted_im_cropped - ref_cropped) ** 2))
            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    return np.roll(image_to_align, best_shift, axis=(0, 1))

ag = align(g, b)
ar = align(r, b)

# créer l'image couleur
im_out = np.dstack([ar, ag, b])

#convert image to 0,255 value
img_RGB_ubyte = sk.util.img_as_ubyte(im_out)

# sauvegarder l'image
fname = 'out_fname.jpg'
skio.imsave(fname, img_RGB_ubyte)

# afficher l'image
skio.imshow(im_out)
skio.show()