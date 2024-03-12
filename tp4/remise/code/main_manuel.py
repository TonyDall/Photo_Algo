import skimage.util
from imageio import imread
import matplotlib.pyplot as plt
import skimage.transform as skt
import skimage as sk
import skimage.io as skio
import numpy as np

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

    imgTrans = np.zeros((new_height, new_width, img.shape[2]))

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

#Charger les images
im1 = imread('./images/1-PartieManuelle/Serie1/IMG_2415.JPG')
im1 = skimage.util.img_as_float(im1)

im2 = imread('./images/1-PartieManuelle/Serie1/IMG_2416.JPG')
im2 = skimage.util.img_as_float(im2)

im3 = imread('./images/1-PartieManuelle/Serie1/IMG_2417.JPG')
im3 = skimage.util.img_as_float(im3)

#Charger les points des fichiers text
pts12 = np.loadtxt('./images/1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt', delimiter=',')
pts21 = np.loadtxt('./images/1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt', delimiter=',')

pts23 = np.loadtxt('./images/1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt', delimiter=',')
pts32 = np.loadtxt('./images/1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt', delimiter=',')

# plt.imshow(im2)
# plt.axis('off')
# # Afficher les points sur l'image
# plt.scatter(pts32[4:8, 0], pts32[4:8, 1], color='red', marker='o', label='Pts3')
# plt.scatter(pts23[4:8, 0], pts23[4:8, 1], color='blue', marker='x', label='Pts2')
#
# plt.legend()
# plt.show()

H = calculerHomographie(pts12[:4], pts21[:4])
im1T, minmax = appliqueTransformation(im1, H)

#On prend les points de 4 à 8 car les résultats donnes ce qui est attendu
H3 = calculerHomographie(pts32[4:8], pts23[4:8])
im3T, minmax_3 = appliqueTransformation(im3, H3)

# Appliquer la transformation à la deuxième image pour les superposer
result2, _ = appliqueTransformation(im2, np.eye(3))

tx = abs(minmax[3])
ty = abs(minmax[2])
tx2 = abs(minmax_3[3])
ty2 = abs(minmax_3[2])

#Matrice pour translation
Htr2 = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]])
Htr3 = np.array([[1, 0, (-tx2 - tx)], [0, 1, -abs(ty2 - ty)], [0, 0, 1]])

# Superposer les deux images en utilisant la moyenne des pixels
mosaic_heigt = max([im1T.shape[0], im3T.shape[0], result2.shape[0]])
mosaic_width = abs(minmax[3]) + abs(minmax_3[3]) + result2.shape[1]
mosaic = np.zeros((mosaic_heigt, mosaic_width, result2.shape[2]))

test1 = skt.warp(im1T, np.eye(3), output_shape=mosaic.shape) #reshape l'image
test2 = skt.warp(im2, Htr2, output_shape=mosaic.shape)
test3 = skt.warp(im3T, Htr3, output_shape=mosaic.shape)

# it's stupid I know
for x in range(mosaic.shape[1]):
    for y in range(mosaic.shape[0]):
        for z in range(3):
            # mosaic[y, x, z] = max(test1[y, x, z], test2[y, x, z], test3[y, x, z])
            if test1[y, x, z] != 0 and test2[y, x, z]!= 0 and test3[y, x, z] != 0:
                mosaic[y, x, z] = (test1[y, x, z] + test2[y, x, z] + test3[y, x, z])/3
            elif test1[y, x, z] != 0 and test2[y, x, z] != 0 and test3[y, x, z] == 0:
                mosaic[y, x, z] = (test1[y, x, z] + test2[y, x, z]) / 2
            elif test1[y, x, z] == 0 and test2[y, x, z] != 0 and test3[y, x, z] != 0:
                mosaic[y, x, z] = (test2[y, x, z] + test3[y, x, z]) / 2
            elif test1[y, x, z] != 0 and test2[y, x, z] == 0 and test3[y, x, z] == 0:
                mosaic[y, x, z] = test1[y, x, z]
            elif test1[y, x, z] == 0 and test2[y, x, z] != 0 and test3[y, x, z] == 0:
                mosaic[y, x, z] = test2[y, x, z]
            elif test1[y, x, z] == 0 and test2[y, x, z] == 0 and test3[y, x, z] != 0:
                mosaic[y, x, z] = test3[y, x, z]

# shenanigans pour avoir la mosaique (trop compliqué)
# for x in range(mosaic.shape[1]):
#     for y in range(mosaic.shape[0]):
#         if x < tx and y < im1T.shape[0]:
#             mosaic[y, x] = im1T[y, x]
#         elif tx <= x <= (tx2 + tx) + 20 and y > ty and y < (im2.shape[0] + ty):
#             mosaic[y, x] = np.maximum.reduce([im2[y - ty, x - tx], im1T[y, x]])
#             mosaic[y, x] = im2[y - ty, x - tx]
#         elif x > (tx2 + tx) + 20 and y < im3T.shape[0] - abs(ty2-ty) and y > abs(ty-ty2):
#             mosaic[y + abs(ty2-ty), x] = im3T[y, x - (tx2 + tx)]

plt.imshow(mosaic)

#Afficher les points d'interet sur im1t
# plt.scatter(pts12[:4, 0], pts12[:4, 1], color='red', marker='o', label='Pts1')
# plt.scatter(pts21[:4, 0], pts21[:4, 1], color='blue', marker='x', label='Pts2')
# plt.scatter(pts21[:4, 0] - minmax[3], pts21[:4, 1] - minmax[2], color='green', marker='+', label='Ptsx')

#Afficher les points d'interet sur im3t
# plt.scatter(pts32[4:8, 0], pts32[4:8, 1], color='red', marker='o', label='Pts1')
# plt.scatter(pts23[4:8, 0], pts23[4:8, 1], color='blue', marker='x', label='Pts2')
# plt.scatter(pts23[4:8, 0] - minmax_3[3], pts23[4:8, 1] - minmax_3[2], color='green', marker='+', label='Ptsx')

plt.legend()
plt.axis('off')
plt.title('Tete')
plt.show()

fname = './images/resultat/serie1_panorama_moyenne.jpg'
img_RGB_ubyte = sk.util.img_as_ubyte(mosaic)
skio.imsave(fname, img_RGB_ubyte)