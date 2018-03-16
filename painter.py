#!/usr/local/opt/python/bin/python2.7
#
# The Python Imaging Library
# $Id$
#
# this demo script illustrates pasting into an already displayed
# photoimage.  note that the current version of Tk updates the whole
# image every time we paste, so to get decent performance, we split
# the image into a set of tiles.
#

import sys

if sys.version_info[0] > 2:
    import tkinter
else:
    import Tkinter as tkinter

import numpy
from PIL import Image, ImageTk
from math import sqrt, acos

# vec3 utils

def vecLength(a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def normalize(a):
    length = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
    if length < 0.00001: return [0.0, 0.0, 0.0]  # replace?
    length = sqrt(length)
    return [a[0] / length, a[1] / length, a[2] / length]

def cross(a, b):
    res = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return res

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def intensity(a):
    b = [float(a[0]) / 255.0, float(a[1]) / 255.0, float(a[2]) / 255.0]
    return dot(b, [0.2126, 0.7152, 0.0722])

def clamp(a, v0, v1):
    return max(v0, min(v1, a))

# Widget

class PaintCanvas(tkinter.Canvas):
    def __init__(self, master, image):
        tkinter.Canvas.__init__(self, master,
                                width=image.size[0], height=image.size[1])

        self.heightIntensity = 4.0

        # get initial data
        self.imageData = numpy.array(im.convert('RGB'))
        self.heightData = numpy.zeros((len(self.imageData[0]), len(self.imageData)))
        self.slopes = numpy.zeros((len(self.imageData[0]), len(self.imageData), 3))

        self.deltaX = 1.0 / len(self.imageData[0])
        self.deltaY = 1.0 / len(self.imageData)

        for y in range(0, len(self.imageData)):
            for x in range(0, len(self.imageData[0])):
                self.heightData[y][x] = intensity(self.imageData[y][x])

        for y in range(0, len(self.imageData)):
            for x in range(0, len(self.imageData[0])):
                self.slopes[y][x] = self.sobelNormaln11(x * self.deltaX, y * self.deltaY, self.heightIntensity)
                # print(self.slopes[y][x])

        # fill the canvas
        self.tile = {}
        self.tilesize = tilesize = 32
        xsize, ysize = image.size
        for x in range(0, xsize, tilesize):
            for y in range(0, ysize, tilesize):
                box = x, y, min(xsize, x+tilesize), min(ysize, y+tilesize)
                tile = ImageTk.PhotoImage(image.crop(box))
                self.create_image(x, y, image=tile, anchor=tkinter.NW)
                self.tile[(x, y)] = box, tile

        self.image = image

        self.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        xy = event.x - 10, event.y - 10, event.x + 10, event.y + 10

        # process the image in some fashion
        norsnippet = numpy.zeros((20, 20, 3))
        for y in range(0, 20):
            for x in range (0, 20):
                cx = (xy[0] + x) * self.deltaX
                cy = (xy[1] + y) * self.deltaY
                cx = clamp(cx, 0.0, 1.0 - self.deltaX)
                cy = clamp(cy, 0.0, 1.0 - self.deltaY)
                ix = (xy[0] + x)
                iy = (xy[1] + y)
                ix = clamp(ix, 0, len(self.imageData[0]) - 1)
                iy = clamp(iy, 0, len(self.imageData) - 1)
                nor = self.sobelNormal01(cx, cy, self.heightIntensity)
                norsnippet[y][x] = [(round(nor[0] * 255.0)), (round(nor[1] * 255.0)), (round(nor[2] * 255.0))]
                self.image.putpixel((ix, iy), (int(norsnippet[y][x][0]), int(norsnippet[y][x][1]), int(norsnippet[y][x][2])))

        self.repair(xy)


    def repair(self, box):
        # update canvas
        dx = box[0] % self.tilesize
        dy = box[1] % self.tilesize
        for x in range(box[0]-dx, box[2]+1, self.tilesize):
            for y in range(box[1]-dy, box[3]+1, self.tilesize):
                try:
                    xy, tile = self.tile[(x, y)]
                    tile.paste(self.image.crop(xy))
                except KeyError:
                    pass  # outside the image
        self.update_idletasks()


    def heightSample(self, x, y):
        x = max(0, min(len(self.heightData[0]) - 1, x))
        y = max(0, min(len(self.heightData) - 1, y))
        return self.heightData[y][x]


    def bilinearSample(self, x, y):
        fx0 = int(x * 255.0)
        fy0 = int(y * 255.0)
        fx1 = fx0 + 1
        fy1 = fy0 + 1
        h00 = self.heightSample(fx0, fy0);
        h01 = self.heightSample(fx1, fy0);
        h10 = self.heightSample(fx0, fy1);
        h11 = self.heightSample(fx1, fy1);
        # return h11
        # print(x)
        tx = x * 255.0 - float(fx0)
        ty = y * 255.0 - float(fy0)
        h0 = tx * h01 + (1.0 - tx) * h00
        h1 = tx * h11 + (1.0 - tx) * h10
        return ty * h1 + (1.0 - ty) * h0


    # given 0-1 x and y and a strength, get a normal for this slope (put between 0 and +1 for drawing)
    # strength determines the z component before normalization, higher strength --> more horizontal
    def sobelNormal01(self, x, y, strength):
        skx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        sky = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0]

        gx = 0.0
        gy = 0.0

        k = 0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                height = self.bilinearSample(x - ox * self.deltaX, y - oy * self.deltaY)
                gx += skx[k] * height
                gy += sky[k] * height
                k = k + 1
        gz = 1.0 / strength
        nor = normalize([gx, gy, gz])

        nor[0] = 0.5 + 0.5 * nor[0]
        nor[1] = 0.5 + 0.5 * nor[1]
        nor[2] = 0.5 + 0.5 * nor[2]
        return nor

    # same as above but does not restrict to positive
    def sobelNormaln11(self, x, y, strength):
        skx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        sky = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0]

        gx = 0.0
        gy = 0.0

        k = 0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                height = self.bilinearSample(x - ox * self.deltaX, y - oy * self.deltaY)
                gx += skx[k] * height
                gy += sky[k] * height
                k = k + 1
        gz = 1.0 / strength
        nor = normalize([gx, gy, gz])
        print(nor)
        return nor



#
# main

if len(sys.argv) != 2:
    print("Usage: painter file")
    sys.exit(1)

root = tkinter.Tk()

im = Image.open(sys.argv[1])

if im.mode != "RGB":
    im = im.convert("RGB")

PaintCanvas(root, im).pack()

root.mainloop()
