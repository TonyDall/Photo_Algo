import skimage.util
from imageio import imread
from align_images import align_images
from crop_image import crop_image
from stacks import stacks
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
def hybrid_image(im1, im2, cutoff_low, cutoff_high):

    passe_bas = sk.filters.gaussian(im1, sigma=cutoff_low)

    passe_haut = im2 - sk.filters.gaussian(im2, sigma=cutoff_high)

    hybride = passe_bas + passe_haut

    tf1 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(passe_bas))))
    tf2 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(passe_haut))))

    skio.imshow(tf1, cmap='gray')
    plt.axis('off')
    plt.title('Passe_Bas')
    skio.show()

    skio.imshow(tf2, cmap='gray')
    plt.axis('off')
    plt.title('Passe_Haut')
    skio.show()

    fname = 'TF_monroe_passe_bas.jpg'
    skio.imsave(fname, sk.img_as_ubyte(np.clip(tf1, 0, 1)))

    fname = 'TF_einstein_passe_haut.jpg'
    skio.imsave(fname, sk.img_as_ubyte(np.clip(tf2, 0, 1)))

    return hybride

# read images
im1 = imread('./Marilyn_Monroe.png', pilmode='L')
im1 = skimage.util.img_as_float(im1)
im2 = imread('./Albert_Einstein.png', pilmode='L')
im2 = skimage.util.img_as_float(im2)

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
im1, im2 = align_images(im1, im2)

# Choose the cutoff frequencies and compute the hybrid image (you supply
# this code)
arbitrary_value_1 = 25
arbitrary_value_2 = 10
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2
im12 = hybrid_image(im1, im2, cutoff_low, cutoff_high)

# Crop resulting image (optional)
assert im12 is not None, "im12 is empty, implement hybrid_image!"
im12 = crop_image(im12)

skio.imshow(im12, cmap='gray')
plt.axis('off')
plt.title('Hybride')
skio.show()
