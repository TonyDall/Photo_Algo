import skimage.util
from imageio import imread
import matplotlib.pyplot as plt
import skimage as sk
import numpy as np


def stacks(im, n, sigma):
    gauss_stack = [im]
    laplace_stack = []

    for i in range(1, n + 1):
        im_filtered = sk.filters.gaussian(im, sigma=sigma * i)
        if i < n:
            gauss_stack.append(im_filtered)
        im_laplace = gauss_stack[i - 1] - im_filtered
        laplace_stack.append(im_laplace)

    return [gauss_stack, laplace_stack]


# read images
im1 = imread('../image/apple.jpeg', pilmode='L')
im1 = skimage.util.img_as_float(im1)
im2 = imread('../image/orange.jpeg', pilmode='L')
im2 = skimage.util.img_as_float(im2)

# Creation d'un masque de meme taille que les images
mask = np.zeros((300, 300), dtype=np.float64)

# Mettre la moitié en blanc
mask[:, :150] = 1.0

# Afficher le masque
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# Compute and display Gaussian and Laplacian Stacks (you supply this code)
num = 5  # number of pyramid levels (you may use more or fewer, as needed)
# Pile pour im1, im2 et mask
stack_im1 = stacks(im1, num, sigma=2)
stack_im2 = stacks(im2, num, sigma=2)
stack_mask = stacks(mask, num, sigma=8)
stack_res = []
stack_res_im1 = []
stack_res_im2 = []
compose_image = np.zeros((300, 300))

for i in range(num):
    stack_res.append(((stack_im1[1][i] * stack_mask[0][i]) + (stack_im2[1][i] * (1.0 - stack_mask[0][i]))))
    stack_res_im1.append(stack_im1[1][i] * stack_mask[0][i])
    stack_res_im2.append(stack_im2[1][i] * (1.0 - stack_mask[0][i]))

# Seulement la moyenne de la premiere image composee est calcule pour l'ajustement
moy_im = (np.mean(im1) + np.mean(im2)) / 2
compose_image = sum(stack_res)
moy_compose = np.mean(compose_image)

offset = moy_im - moy_compose
print('Différence entre les deux moyennes: {}'.format(offset))

# Weighting every composed image in the laplace stack
compose_fix = 0.4*stack_res[0] + 0.8*stack_res[1] + 1.2*stack_res[2] + 1.6*stack_res[3] + 2.0*stack_res[4]

compose_fix = compose_fix + offset

stack_res.append(compose_fix)
stack_res_im1.append(sum(stack_res_im1))
stack_res_im2.append(sum(stack_res_im2))

plt.imshow(compose_fix, cmap="gray")
plt.axis('off')
plt.title('Compose')
plt.show()

im1_row = np.hstack(stack_res_im1)
im2_row = np.hstack(stack_res_im2)
compose_row = np.hstack(stack_res)

plt.figure(figsize=(20, 8))
plt.subplot(3, 1, 1)
plt.imshow(im1_row, cmap='gray')
plt.axis('off')
plt.title('Apple')

plt.subplot(3, 1, 2)
plt.imshow(im2_row, cmap='gray')
plt.axis('off')
plt.title('Orange')

plt.subplot(3, 1, 3)
plt.imshow(compose_row, cmap='gray')
plt.axis('off')
plt.title('Composée')

plt.tight_layout()
plt.show()


