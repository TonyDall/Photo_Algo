import numpy as np
import skimage as sk
import matplotlib.pyplot as plt


def stacks(im12, n):

    gauss_stack = [im12]
    laplace_stack = []

    for i in range(1, n+1):
        im_filtered = sk.filters.gaussian(im12, sigma=2*i)
        if i < n:
            gauss_stack.append(im_filtered)
        im_laplace = gauss_stack[i-1] - im_filtered
        laplace_stack.append(im_laplace)

    gauss_row = np.hstack(gauss_stack)
    laplace_row = np.hstack(laplace_stack)

    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(gauss_row, cmap='gray')
    plt.axis('off')
    plt.title('Gaussian Stack')

    plt.subplot(2, 1, 2)
    plt.imshow(laplace_row, cmap='gray')
    plt.axis('off')
    plt.title('Laplacian Stack')

    plt.tight_layout()
    plt.show()
