import numpy as np
import math
from skimage import io, util, transform
import heapq
from skimage.color import rgb2gray
from skimage.filters import gaussian
from matplotlib import pyplot as plt


def quilt_random(sample, outsize, patchsize):
    h, w, _ = sample.shape
    outH = outsize*patchsize
    outW = outsize*patchsize
    res = np.zeros((outH, outW, sample.shape[2]))

    for i in range(outsize):
        for j in range(outsize):
            x = i * (patchsize)
            y = j * (patchsize)

            xi = np.random.randint(h - patchsize)
            yj = np.random.randint(w - patchsize)

            res[y:y + patchsize, x:x + patchsize] = sample[xi:xi + patchsize, yj:yj + patchsize]

    return res


def ssd(patch, patchsize, overlap, res, y, x):
    erreur = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y + patchsize, x:x + overlap]
        erreur += np.sum(left ** 2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + patchsize]
        erreur += np.sum(up ** 2)

    if x > 0 and y > 0:
        coin = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        erreur -= np.sum(coin ** 2)

    return erreur


def quilt_simple(sample, outsize, patchsize, overlap):
    h, w, _ = sample.shape
    outH = outsize*patchsize - ((outsize - 1) * overlap)
    outW = outsize*patchsize - ((outsize - 1) * overlap)
    res = np.zeros((outH, outW, sample.shape[2]))

    for i in range(outsize):
        for j in range(outsize):
            x = i * (patchsize - overlap)
            y = j * (patchsize - overlap)

            if i == 0 and j == 0:
                patch = quilt_random(sample, 1, patchsize)
                res[y:y + patchsize, x:x + patchsize] = patch
            else:
                errors = np.zeros((h - patchsize, w - patchsize))
                for xi in range(h - patchsize):
                    for yj in range(w - patchsize):
                        patch = sample[xi:xi + patchsize, yj:yj + patchsize]
                        e = ssd(patch, patchsize, overlap, res, y, x)
                        errors[xi, yj] = e

                xi, yj = np.unravel_index(np.argmin(errors), errors.shape)
                patch = sample[xi:xi + patchsize, yj:yj + patchsize]

            res[y:y + patchsize, x:x + patchsize] = patch

    return res


def cutDijk(bndcost):
    # dijkstra
    pq = [(error, [i]) for i, error in enumerate(bndcost[0])]
    heapq.heapify(pq)

    h, w = bndcost.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + bndcost[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def cutDyna(bndcost):
    # dynamique
    errors = np.pad(bndcost, [(0, 0), (1, 1)], mode='constant', constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)

    paths -= 1

    minCutPath = [np.argmin(cumError)]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])

    return map(lambda x: x - 1, reversed(minCutPath))


def quilt_cut(sample, outsize, patchsize, overlap):
    h, w, _ = sample.shape
    outH = outsize * patchsize - ((outsize - 1) * overlap)
    outW = outsize * patchsize - ((outsize - 1) * overlap)
    res = np.zeros((outH, outW, sample.shape[2]))

    for i in range(outsize):
        for j in range(outsize):
            x = i * (patchsize - overlap)
            y = j * (patchsize - overlap)

            if i == 0 and j == 0:
                patch = quilt_random(sample, 1, patchsize)
                res[y:y + patchsize, x:x + patchsize] = patch
            else:
                errors = np.zeros((h - patchsize, w - patchsize))
                for xi in range(h - patchsize):
                    for yj in range(w - patchsize):
                        patch = sample[xi:xi + patchsize, yj:yj + patchsize]
                        err = ssd(patch, patchsize, overlap, res, y, x)
                        errors[xi, yj] = err

                xi, yj = np.unravel_index(np.argmin(errors), errors.shape)
                patch = sample[xi:xi + patchsize, yj:yj + patchsize]

                patch = patch.copy()
                dy, dx, _ = patch.shape
                minCut = np.zeros_like(patch, dtype=bool)

                if x > 0:
                    gauche = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
                    bndcost = np.sum(gauche ** 2, axis=2)
                    for gi, gj in enumerate(cutDijk(bndcost)):
                        minCut[gi, :gj] = True

                if y > 0:
                    haut = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
                    bndcost = np.sum(haut ** 2, axis=2)
                    for hj, hi in enumerate(cutDijk(bndcost.T)):
                        minCut[:hi, hj] = True

                np.copyto(patch, res[y:y + dy, x:x + dx], where=minCut)

            res[y:y + patchsize, x:x + patchsize] = patch
            # io.imshow(res)
            # io.show()
    return res

def texture_transfer(sample, image, patchsize, alpha):
    samplecorr = rgb2gray(sample)
    imagecorr = rgb2gray(image)

    samplecorr = gaussian(samplecorr, sigma=3)
    imagecorr = gaussian(imagecorr, sigma=3)

    # remove alpha channel
    sample = util.img_as_float(sample)[:,:,:3]
    image = util.img_as_float(image)[:,:,:3]

    h, w, _ = image.shape
    overlap = patchsize // 6

    nbrPatchesH = math.ceil((h - patchsize) / (patchsize - overlap)) + 1 or 1
    nbrPatchesL = math.ceil((w - patchsize) / (patchsize - overlap)) + 1 or 1

    res = np.zeros_like(image)
    h, w, _ = sample.shape

    for i in range(nbrPatchesH):
        for j in range(nbrPatchesL):
            y = i * (patchsize - overlap)
            x = j * (patchsize - overlap)

            if i == 0 and j == 0:
                errors = np.zeros((h - patchsize, w - patchsize))
                corrImPatch = imagecorr[y:y+patchsize, x:x+patchsize]
                curH, curW = corrImPatch.shape
                for xi in range(h - patchsize):
                    for yj in range(w - patchsize):
                        corrSamPatch = samplecorr[xi:xi + curH, yj:yj + curW]
                        e = corrSamPatch - corrImPatch
                        errors[xi, yj] = np.sum(e ** 2)

                xi, yj = np.unravel_index(np.argmin(errors), errors.shape)
                patch = sample[xi:xi + curH, yj:yj + curW]

            else:
                errors = np.zeros((h - patchsize, w - patchsize))
                corrImPatch = imagecorr[y:y+patchsize, x:x+patchsize]
                curH, curW = corrImPatch.shape
                for xxi in range(h - patchsize):
                    for yyj in range(w - patchsize):
                        patch = sample[xxi:xxi+curH, yyj:yyj+curW]
                        err = ssd(patch, patchsize, overlap, res, y, x)
                        overlapErr = np.sum(err)

                        corrSamplePatch = samplecorr[xxi:xxi + curH, yyj:yyj + curW]
                        corrError = np.sum((corrSamplePatch - corrImPatch) ** 2)

                        errors[xxi, yyj] = alpha * overlapErr + (1 - alpha) * corrError

                xxi, yyj = np.unravel_index(np.argmin(errors), errors.shape)
                patch = sample[xxi:xxi + curH, yyj:yyj + curW]

                patch = patch.copy()
                dy, dx, _ = patch.shape
                minCut = np.zeros_like(patch, dtype=bool)

                if x > 0:
                    gauche = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
                    bndcost = np.sum(gauche ** 2, axis=2)
                    for gi, gj in enumerate(cutDyna(bndcost)):
                        minCut[gi, :gj] = True

                if y > 0:
                    haut = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
                    bndcost = np.sum(haut ** 2, axis=2)
                    for hj, hi in enumerate(cutDyna(bndcost.T)):
                        minCut[:hi, hj] = True

                np.copyto(patch, res[y:y + dy, x:x + dx], where=minCut)

            res[y:y + patchsize, x:x + patchsize] = patch

    return res

# s = "Image/textures/"
#
# texture = io.imread(s + "roches.jpg")
# io.imshow(texture)
# io.show()
#
# texture = util.img_as_float(texture)
# io.imshow(quilt_random(texture, 6, 101))
# io.show()
#
# io.imshow(quilt_simple(texture, 6, 101, 101//6))
# io.show()
#
# io.imshow(quilt_cut(texture, 6, 101, 101//6))
# io.show()

s = "Image/transfert/"
# sketch = io.imread(s + "sketch.tiff")
# feynman = io.imread(s + "feynman.tiff")

texture = io.imread(s + "starry-night.jpg")
village = io.imread(s + "ggb.jpg")

# resized_texture = transform.resize(texture, (280, 350))
# resized_village = transform.resize(village, (469, 750))
# io.imshow(resized_village)
# io.show()

io.imshow(texture_transfer(texture, village, 21, 0.1))
io.show()


