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
im1 = imread('./images/1-PartieManuelle/Serie3/IMG_2409.JPG')
im1 = skimage.util.img_as_float(im1)

im2 = imread('./images/1-PartieManuelle/Serie3/IMG_2410.JPG')
im2 = skimage.util.img_as_float(im2)

im3 = imread('./images/1-PartieManuelle/Serie3/IMG_2411.JPG')
im3 = skimage.util.img_as_float(im3)

#Charger les points des fichiers text
pts12 = np.loadtxt('./images/1-PartieManuelle/Serie3/pts12_2409.txt', delimiter='\t', usecols=(1,2))
pts21 = np.loadtxt('./images/1-PartieManuelle/Serie3/pts21_2410.txt', delimiter='\t', usecols=(1,2))

pts23 = np.loadtxt('./images/1-PartieManuelle/Serie3/pts23_2410.txt', delimiter='\t', usecols=(1,2))
pts32 = np.loadtxt('./images/1-PartieManuelle/Serie3/pts32_2411.txt', delimiter='\t', usecols=(1,2))

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
H3 = calculerHomographie(pts32[:4], pts23[:4])
im3T, minmax_3 = appliqueTransformation(im3, H3)

# Appliquer la transformation à la deuxième image pour les superposer
result2, _ = appliqueTransformation(im2, np.eye(3))

tx = abs(minmax[3])
ty = abs(minmax[2])
tx2 = abs(minmax_3[3])
ty2 = abs(minmax_3[2])

print(minmax, minmax_3)
#Matrice pour translation
Htr2 = np.array([[1, 0, -tx], [0, 1, -ty2], [0, 0, 1]])
Htr3 = np.array([[1, 0, (-tx2 - tx)], [0, 1, 0], [0, 0, 1]])
Htr1 = np.array([[1, 0, 0], [0, 1, -(ty2-ty)], [0, 0, 1]])

# Superposer les deux images en utilisant la moyenne des pixels
mosaic_heigt = max([im1T.shape[0], im3T.shape[0], result2.shape[0]])
mosaic_width = abs(minmax[1]) + abs(minmax_3[1]) + minmax_3[2]
mosaic = np.zeros((mosaic_heigt, mosaic_width, result2.shape[2]))

test1 = skt.warp(im1T, Htr1, output_shape=mosaic.shape)
test2 = skt.warp(im2, Htr2, output_shape=mosaic.shape)
test3 = skt.warp(im3T, Htr3, output_shape=mosaic.shape)

# it's stupid I know
for x in range(mosaic.shape[1]):
    for y in range(mosaic.shape[0]):
        for z in range(3):
            # mosaic[y, x, z] = max(test1[y, x, z], test2[y, x, z], test3[y, x, z])
            if test1[y, x, z] != 0 and test2[y, x, z] != 0 and test3[y, x, z] != 0:
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

fname = './images/resultat/serie3_panorama_moyenne.jpg'
img_RGB_ubyte = sk.util.img_as_ubyte(mosaic)
skio.imsave(fname, img_RGB_ubyte)