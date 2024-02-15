import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt

# Chargment des image
imname_1 = 'eagle.jpg'
imname_2 = 'Drake.jpg'

im = skio.imread(imname_1)

im = sk.util.img_as_float(im)

r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]

# Application du filtre gaussien
sigma = 1
im_filtered = sk.filters.gaussian(im, sigma=sigma, channel_axis=-1)

# Accentuation de l'image
alpha = 1.5
im_sharpened = sk.img_as_ubyte(np.clip(im + (alpha * (im - im_filtered)), 0, 1))

# Sauvegarder l'image filtree
fname = 'filtree_1.jpg'
skio.imsave(fname, sk.img_as_ubyte(np.clip(im_filtered, 0, 1)))

# sauvegarder l'image
fname = 'detail_1.jpg'
skio.imsave(fname, sk.img_as_ubyte(np.clip(im - im_filtered, 0, 1)))

# sauvegarder l'image
fname = 'accentuee_1.jpg'
skio.imsave(fname, im_sharpened)

# afficher l'image
skio.imshow(im_sharpened)
plt.axis('off')
skio.show()


