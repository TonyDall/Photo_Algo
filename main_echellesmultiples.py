#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TP1, code python pour débuter

# quelques librairies suggérées
# vous pourriez aussi utiliser matplotlib et opencv pour lire, afficher et sauvegarder des images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as skt
import matplotlib.pyplot as plt
import multiprocessing

# nom du fichier d'image
imname = 'images/01047u.tif'

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

# aligner les images... c'est ici que vous commencez à coder!
# ces quelques fonctions pourraient vous être utiles:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(image_to_align, reference_image, max_shift=15):
    best_score = np.inf
    best_shift = (0, 0)

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_image = np.roll(image_to_align, (dx, dy), axis=(1, 0))
            score = np.sum(np.square(np.subtract(shifted_image, reference_image)))

            if score < best_score:
                best_score = score
                best_shift = (dx, dy)
    return best_shift


def pyramid(image_to_align, reference_image, num_scales=3):
    best_align = (0, 0)

    for scale in range(num_scales, 0, -1):
        best_align = (best_align[0] * 2, best_align[1] * 2)
        image_to_align_rescaled = skt.rescale(image_to_align, 1 / 2 ** scale, anti_aliasing=True)
        ref_image_rescaled = skt.rescale(reference_image, 1 / 2 ** scale, anti_aliasing=True)

        image_to_align_rescaled = np.roll(image_to_align_rescaled, (best_align[0], best_align[1]), axis=(1, 0))
        alignment = align(image_to_align_rescaled, ref_image_rescaled)
        best_align = (alignment[0] + best_align[0], alignment[1] + best_align[1])

    aligned_image = np.roll(image_to_align, 2*best_align[0], axis=1)
    aligned_image = np.roll(aligned_image, 2*best_align[1], axis=0)
    return aligned_image


def crop(image, px_amount):
    return image[px_amount:-px_amount, px_amount:-px_amount]


# Specify the number of pixels to crop from each side
crop_pixels = 250

g = crop(g, crop_pixels)
r = crop(r, crop_pixels)
b = crop(b, crop_pixels)

ag = pyramid(g, b)
ar = pyramid(r, b)

# créer l'image couleur
im_out = np.dstack([ar, ag, b])

#convert image to 0,255 value
img_RGB_ubyte = sk.util.img_as_ubyte(im_out)

# sauvegarder l'image
fname = 'res2.jpg'
skio.imsave(fname, img_RGB_ubyte)

# afficher l'image
skio.imshow(im_out)
plt.axis('off')
skio.show()