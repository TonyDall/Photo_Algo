import cv2
import numpy as np

# Charger les images R, E, M et I
R = cv2.imread('./image/res/R0001.png')
E = cv2.imread('./image/res/E0001.png')
M = cv2.imread('./image/res/M0001.png')
I = cv2.imread('./image/res/I0001.png')

# Convertir M en une image binaire
M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)[1]
M = M.astype(np.float32) / 255.0  # Normaliser le masque entre 0 et 1

# Calculer l'image de composition C
c = 5  # Facteur de pondération pour ajuster l'opacité des objets
C = M * R + (1 - M) * (I + c * (R - E))

# Convertir C en image uint8
C = np.clip(C, 0, 255).astype(np.uint8)

# Afficher ou enregistrer l'image C
cv2.imshow('Composition', C)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('composition.png', C)  # Enregistrer l'image si nécessaire