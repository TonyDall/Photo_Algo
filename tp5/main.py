import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def weight(I):
    I_max = 255.
    if I <= 127.5:
        return I
    return I_max - I


def sample_intensity(stack):
    I_min, I_max = 0., 255.
    num_intensities = int(I_max - I_min + 1)
    num_images = len(stack)
    sample = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = stack[num_images // 2]

    for i in range(int(I_min), int(I_max + 1)):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(len(stack)):
                sample[i, j] = stack[j][rows[idx], cols[idx]]

    return sample


def estimate_curve(sample, exps, l):
    I_min, I_max = 0., 255.
    n = 255
    A = np.zeros((sample.shape[0] * sample.shape[1] + n, n + sample.shape[0] + 1), dtype=np.float64)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)

    k = 0

    # 1. data fitting
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            I_ij = sample[i, j]
            w_ij = weight(I_ij)
            A[k, I_ij] = w_ij
            A[k, n + 1 + i] = -w_ij
            b[k, 0] = w_ij * exps[j]
            k += 1

    # 2. smoothing
    for I_k in range(int(I_min + 1), int(I_max)):
        w_k = weight(I_k)
        A[k, I_k - 1] = w_k * l
        A[k, I_k] = -2 * w_k * l
        A[k, I_k + 1] = w_k * l
        k += 1

    # 3. Color centering
    A[k, int((I_max - I_min) // 2)] = 1

    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b)

    g = x[0: n + 1]

    return g[:, 0]


def computeRadiance(stack, exps, curve):
    stack_shape = stack.shape
    img_rad = np.zeros(stack_shape[1:], dtype=np.float64)

    num_imgs = stack_shape[0]

    for i in range(stack_shape[1]):
        for j in range(stack_shape[2]):
            g = np.array([curve[int(stack[k][i, j])] for k in range(num_imgs)])
            w = np.array([weight(stack[k][i, j]) for k in range(num_imgs)])

            sumW = np.sum(w)
            if sumW > 0:
                img_rad[i, j] = np.sum(w * (g - exps) / sumW)
            else:
                img_rad[i, j] = g[num_imgs // 2] - exps[num_imgs // 2]
    return img_rad


def globalTonemap(img, l):
    return cv2.pow(img / 255., 1.0 / l)


def intensityAdjustment(image, template):
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:, :, ch]), np.average(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output


def load(path_test):
    filenames = []
    exposure_times = []
    f = open(os.path.join(path_test, 'liste_images.txt'))
    for line in f:
        # (filename, exposure, *rest) = line.split()
        (filename, exposure) = line.split()
        filenames += [os.path.join(path_test, filename)]
        # exposure_times += [math.log(float(exposure),2)]
        exposure_times += [float(exposure)]
    return filenames, exposure_times


def read(path_list):
    shape = cv2.imread(path_list[0]).shape

    stack = np.zeros((len(path_list), shape[0], shape[1], shape[2]))
    for i in path_list:
        im = cv2.imread(i)
        stack[path_list.index(i), :, :, :] = im
    return stack


def gsolve(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    # Include the data-fitting equations
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k, 0] = wij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(1, n - 1):
        A[k, i - 1] = l * w[i]
        A[k, i] = -2 * l * w[i]
        A[k, i + 1] = l * w[i]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n].flatten()
    lE = x[n:].flatten()

    return g, lE


def weight(z):
    if z <= 127.5:
        return z
    else:
        return 255 - z


gamma = True
cmap = True

#Logique Principale pour faire l'image HDR
list_file, exps = load('./image')
stack = read(list_file)

num_channels = stack.shape[-1]
hdr_img = np.zeros(stack[0].shape, dtype=np.float64)
gF = []

for c in range(num_channels):
    layer_stack = [img[:, :, c] for img in stack]

    sample = sample_intensity(layer_stack)

    w = np.array([weight(z) for z in range(256)])
    g, lE = gsolve(sample, exps, 10, w)

    # curve = estimate_curve(sample, exps, 10.)
    gF.append(g)

    img_rad = computeRadiance(np.array(layer_stack), exps, lE)

    hdr_img[:, :, c] = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

plt.figure()
plt.plot(gF[0], range(256), 'r')
plt.plot(gF[1], range(256), 'g')
plt.plot(gF[2], range(256), 'b')
plt.ylabel('Pixel Value')
plt.xlabel('Log Exposure')
plt.title('Imaging System Response Function')
plt.grid(True)
plt.show()

if gamma:
    output = np.uint8(globalTonemap(hdr_img, 1.3) * 255.)
else:
    tm = cv2.createTonemapMantiuk()
    output = np.uint8(255. * tm.process((hdr_img / 255.).astype(np.float32)))

if cmap:
    from matplotlib.pylab import cm

    colorize = cm.jet
    cmap = np.float32(cv2.cvtColor(np.uint8(hdr_img), cv2.COLOR_BGR2GRAY) / 255.)
    cmap = colorize(cmap)
    cv2.imwrite(('cmap.jpg'), np.uint8(cmap * 255.))

template = stack[len(stack) // 2]
image_tuned = intensityAdjustment(output, template)
output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Save the HDR image as a TIFF or EXR file
cv2.imwrite('sphere.hdr', hdr_img.astype(np.float32))  # Save as a HDR file (OpenCV supports HDR files with .hdr extension)