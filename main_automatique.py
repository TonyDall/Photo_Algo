import skimage.util
from imageio import imread
import matplotlib.pyplot as plt
import skimage.transform as skt
import skimage as sk
import skimage.io as skio
import numpy as np
import sys
from Harris import detect_corners
import math
from operator import itemgetter
import pywt

def get_point_coor(x, y, H):
    input = np.array(([x], [y], [1]))
    output = np.dot(H, input)
    return int(output[0] / output[2]), int(output[1] / output[2])


def appliqueTransformation(img, H):
    left, up = 0, 0
    right, down = img.shape[1], img.shape[0]
    Hinv = np.linalg.inv(H)

    # Obtention de la nouvelle taille d'image
    height_max = max(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                     get_point_coor(right, down, H)[1])
    width_max = max(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                    get_point_coor(right, down, H)[0])
    height_min = min(get_point_coor(left, up, H)[1], get_point_coor(left, down, H)[1], get_point_coor(right, up, H)[1],
                     get_point_coor(right, down, H)[1])
    width_min = min(get_point_coor(left, up, H)[0], get_point_coor(left, down, H)[0], get_point_coor(right, up, H)[0],
                    get_point_coor(right, down, H)[0])

    minmax = [height_max, width_max, height_min, width_min]

    # Enlève les bordune noir inutile dans l'image resultante
    if height_min < 0 and width_min < 0:
        new_height = abs(height_max) + abs(height_min)
        new_width = abs(width_max) + abs(width_min)
    elif height_min < 0 < width_min:
        new_height = abs(height_max) + abs(height_min)
        new_width = abs(width_max) - abs(width_min)
    elif width_min < 0 < height_min:
        new_height = abs(height_max) - abs(height_min)
        new_width = abs(width_max) + abs(width_min)
    elif width_min > 0 < height_min:
        new_height = abs(height_max) - abs(height_min)
        new_width = abs(width_max) - abs(width_min)
    else:
        new_height = abs(height_max)
        new_width = abs(width_max)

    # Check if the image is grayscale or color
    is_color = len(img.shape) == 3 and img.shape[2] == 3

    if is_color:
        imgTrans = np.zeros((new_height, new_width, img.shape[2]))
    else:
        imgTrans = np.zeros((new_height, new_width))

    # Boucle sur chaque pixel de l'image de sortie
    for i in range(new_width):
        for j in range(new_height):
            # Appliquer la transformation de perspective inverse pour obtenir le pixel correspondant dans img
            xw, yw, w = Hinv.dot(np.array(([i + width_min], [j + height_min], [1])))
            x = int(round(xw[0] / w[0]))
            y = int(round(yw[0] / w[0]))

            # Regarde si les pixels sont à l'intérieur de l'image
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                imgTrans[j, i] = img[y, x]

    return imgTrans, minmax


def calculerHomographie(im1_pts, im2_pts):
    A = np.zeros((8, 9))
    for i in range(4):
        x, y = im1_pts[i]
        u, v = im2_pts[i]
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def get_descriptor(image, keypoint):
    # Obtenir les coordonnées du point d'intérêt
    x, y = keypoint

    # Définir la taille de la fenêtre
    window_size = 40

    # Définir la taille de l'échantillon
    sample_size = 8

    # Définir les coordonnées de la fenêtre
    top_left_x = int(x - window_size/2)
    top_left_y = int(y - window_size/2)
    bottom_right_x = int(top_left_x + window_size)
    bottom_right_y = int(top_left_y + window_size)

    # Extraire la fenêtre de l'image
    window = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Vérifier si la fenêtre est vide
    if window.size == 0:
        return None

    # Réduire la taille de la fenêtre (sample au 5 pixel)
    sample = skimage.transform.resize(window, (sample_size, sample_size))

    # Normaliser les valeurs des pixels
    mean = np.mean(sample)
    std = np.std(sample)
    normalized_sample = (sample - mean) / std

    coeffs = pywt.dwt2(normalized_sample, 'haar')
    feature = pywt.idwt2(coeffs, 'haar')
    # Retourner le descripteur normalisé (1D)
    return feature.flatten()


def get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size):
    potential_matches = []
    offset = win_size // 2
    breakloop = False

    # Feature descriptor with brute-force matching
    for i in range(len(ft1[0])):

        r1, w1, h1 = ft1[0][i], ft1[1][i], ft1[2][i]

        # Copy image as to not modify it by reference
        img_left_tmp = np.copy(img_left)

        # Copy coordinates incase changed by border
        h1_tmp = h1
        w1_tmp = w1

        # Run multiscale oriented patches descriptor
        feature_left = get_descriptor(img_left_tmp, (w1, h1))

        if feature_left is None:
            continue  # Ignore if descriptor is None

        lowest_dist = math.inf
        potential_match = ()
        for j in range(len(ft2[0])):

            r2, w2, h2 = ft2[0][j], ft2[1][j], ft2[2][j]

            # Copy image as to not modify it by reference
            img_right_tmp = np.copy(img_right)

            # Copy coordinates incase changed by border
            h2_tmp = h2
            w2_tmp = w2

            # Run multiscale oriented patches descriptor
            feature_right = get_descriptor(img_right_tmp, (w2, h2))

            if feature_right is None:
                continue  # Ignore if descriptor is None

            # Check distance between features
            curr_dist = np.linalg.norm(feature_left - feature_right)
            if curr_dist < lowest_dist:
                lowest_dist = curr_dist
                potential_match = ([h1_tmp, w1_tmp, r1], [h2_tmp, w2_tmp, r2], curr_dist)

        potential_matches.append(potential_match)

    # Sort matches from smallest distance up
    if all(match == () for match in potential_matches):
        breakloop = True
        return np.array(left_match, dtype=np.float32), np.array(right_match, dtype=np.float32)

    matches = sorted(potential_matches, key=itemgetter(2))

    for match in matches:
        # Ensure no duplicates
        if match[0][0:2] not in left_match and match[1][0:2] not in right_match:
            # Add to matches
            left_match.append(match[0][0:2])
            right_match.append(match[1][0:2])
            # Remove from potential matches
            for m in reversed(range(len(ft1[0]))):
                if np.all([ft1[0][m] == match[0][2], ft1[1][m] == match[0][1], ft1[2][m] == match[0][0]]):
                    ft1 = np.delete(ft1, m, axis=1)
            for n in reversed(range(len(ft2[0]))):
                if np.all([ft2[0][n] == match[1][2], ft2[1][n] == match[1][1], ft2[2][n] == match[1][0]]):
                    ft2 = np.delete(ft2, n, axis=1)

    # Recursively keep going until every point has a match
    # while (len(left_match) < max_matches and len(right_match) < max_matches and breakloop != True):
    if(breakloop == False):
        get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size)

    return np.array(left_match, dtype=np.float32), np.array(right_match, dtype=np.float32)


def ransac(pts1, pts2, max_iters=1000, epsilon=1):
    best_matches = []
    # Number of samples
    N = 4

    for i in range(max_iters):
        # Get 4 random samples from features
        idx = np.random.randint(0, len(pts1) - 1, N)
        src = pts1[idx]
        dst = pts2[idx]

        # Calculate the homography matrix H
        H = calculerHomographie(src, dst)

        # Transform points using H
        Hp = []
        for x, y in pts1:
            x_p, y_p = get_point_coor(x, y, H)
            Hp.append([x_p, y_p])

        # Find the inliers by computing the SSD(p',Hp) and saving inliers (feature pairs) that are SSD(p',Hp) < epsilon
        inliers = []
        for i in range(len(pts1)):
            ssd = np.sum(np.square(pts2[i] - Hp[i]))
            if ssd < epsilon:
                inliers.append([pts1[i], pts2[i]])

        # Keep the largest set of inliers and the corresponding homography matrix
        if len(inliers) > len(best_matches):
            best_matches = inliers

    return best_matches


#Charger les images
im0pth = './images/2-PartieAutomatique/Serie1/goldengate-00.PNG'
im0 = imread(im0pth)
im0 = skimage.util.img_as_float(im0)

im1pth = './images/2-PartieAutomatique/Serie1/goldengate-01.PNG'
im1 = imread(im1pth)
im1 = skimage.util.img_as_float(im1)

im2pth = './images/2-PartieAutomatique/Serie1/goldengate-02.PNG'
im2 = imread(im2pth)
im2 = skimage.util.img_as_float(im2)

im3pth = './images/2-PartieAutomatique/Serie1/goldengate-03.PNG'
im3 = imread(im3pth)
im3 = skimage.util.img_as_float(im3)

im4pth = './images/2-PartieAutomatique/Serie1/goldengate-04.PNG'
im4 = imread(im4pth)
im4 = skimage.util.img_as_float(im4)

im5pth = './images/2-PartieAutomatique/Serie1/goldengate-05.PNG'
im5 = imread(im5pth)
im5 = skimage.util.img_as_float(im5)

# ================ Changer les im ICI ================
values, x_coords, y_coords = detect_corners(im2pth)
values1, x_coords1, y_coords1 = detect_corners(im3pth)

ftL = values, x_coords, y_coords
ftR = values1, x_coords1, y_coords1

max_matches = min(len(values), len(values1))

# ================ Changer les im ICI ================
ptsL, ptsR = get_matches(list(ftL), list(ftR), [], [], im2, im3, max_matches, win_size=40)

matches = ransac(ptsL, ptsR, 1000, 1)

pts12 = [points[0] for points in matches]
pts21 = [points[1] for points in matches]


#Afficher les points d'interet sur im1t
#================ Changer les im ICI ================
plt.imshow(im2, cmap='gray')
plt.axis('off')
for i in range(4):
    plt.scatter(pts12[i][1], pts12[i][0], color='red', marker='o', label='Pts1')
    plt.scatter(pts21[i][1], pts21[i][0], color='blue', marker='x', label='Pts2')
plt.show()

plt.imshow(im3, cmap='gray')
plt.axis('off')
for i in range(4):
    plt.scatter(pts12[i][1], pts12[i][0], color='red', marker='o', label='Pts1')
    plt.scatter(pts21[i][1], pts21[i][0], color='blue', marker='x', label='Pts2')
plt.show()

H = calculerHomographie(pts12, pts21)
print(H)

#================ Changer les im ICI ================
im1T, minmax = appliqueTransformation(im2, H)


plt.imshow(im1T, cmap='gray')
plt.axis('off')
plt.show()

fname = './images/resultat/im2.jpg'
img_RGB_ubyte = sk.util.img_as_ubyte(im1T)
skio.imsave(fname, img_RGB_ubyte)