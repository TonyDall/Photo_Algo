#!/usr/bin/env python
# -*- noplot -*-

"""
This example shows how to use matplotlib to provide a data cursor.  It
uses matplotlib to draw the cursor and may be a slow since this
requires redrawing the figure with every mouse move.

Faster cursoring is possible using native GUI drawing, as in
wxcursor_demo.py
"""
from __future__ import print_function
from pylab import *
# import scipy.misc
from skimage import io
import sys
import skimage.util
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as skt
import skimage.io as skio
import skimage as sk

class Cursor:
    def __init__(self, ax, s):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.f = sys.argv[2]
        self.count = 1
        self.s = s

    def mouseclick(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata
        print(x,y)

        h = open(self.f, 'a')
        h.write("\t{}\t{}\n".format(x,y))
        self.ax.text(x+4, y-4, str(self.count), fontsize=14, color='r')
        self.ax.plot(x, y, '.r')
        self.count += 1
        draw()

im1 = imread('../tp3/renard.jpg')
im1 = skimage.util.img_as_float(im1)

fig, ax = subplots()
p = io.imread(sys.argv[1])
ax.imshow(p)
cursor = Cursor(ax, p.shape)
connect('button_press_event', cursor.mouseclick)

show()