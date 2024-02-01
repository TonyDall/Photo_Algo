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
import warnings

# nom du fichier d'image
im_for_r = 'personal image/IMG_3780.jpg'
im_for_b = 'personal image/IMG_3781.jpg'
im_for_g = 'personal image/IMG_3782.jpg'

warnings.filterwarnings("ignore")

# lire l'image
im_r = skio.imread(im_for_r)
im_b = skio.imread(im_for_b)
im_g = skio.imread(im_for_g)

r = im_r[:, :, 0]
g = im_r[:, :, 1]
b = im_r[:, :, 2]

# conversion en double
r = sk.util.img_as_float(r)
b = sk.util.img_as_float(b)
g = sk.util.img_as_float(g)

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


ag = pyramid(g, b)
ar = pyramid(r, b)

# créer l'image couleur
im_out = np.dstack([ar, ag, b])

#convert image to 0,255 value
img_RGB_ubyte = sk.util.img_as_ubyte(im_out)

# sauvegarder l'image
fname = 'groot.jpg'
skio.imsave(fname, img_RGB_ubyte)

# afficher l'image
skio.imshow(im_out)
plt.axis('off')
skio.show()