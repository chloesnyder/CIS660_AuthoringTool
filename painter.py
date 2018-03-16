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
    import ttk

else:
    import Tkinter as tkinter
    import ttk

import numpy
from PIL import Image, ImageTk
from math import sqrt, acos, floor


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


def pxToFloat(a):
    return [float(a[0]) / 255.0, float(a[1]) / 255.0, float(a[2]) / 255.0]

# Widget

class PaintCanvas(tkinter.Canvas):
    def __init__(self, master, image, mode):
        tkinter.Canvas.__init__(self, master,
                                width=image.size[0], height=image.size[1])
        mode = int(mode)

        self.heightIntensity = 4.0

        # get initial data
        self.imageData = numpy.array(im.convert('RGB'))
        self.heightData = numpy.zeros((len(self.imageData[0]), len(self.imageData)))
        self.slopes = numpy.zeros((len(self.imageData[0]), len(self.imageData), 3))
        self.remappedSlopes = numpy.zeros((len(self.imageData[0]), len(self.imageData), 3))

        self.deltaX = 1.0 / len(self.imageData[0])
        self.deltaY = 1.0 / len(self.imageData)

        self.maxPx = len(self.imageData[0]) - 1
        self.maxPy = len(self.imageData) - 1

        for y in range(0, len(self.imageData)):
            for x in range(0, len(self.imageData[0])):
                self.heightData[y][x] = intensity(self.imageData[y][x])

        for y in range(0, len(self.imageData)):
            for x in range(0, len(self.imageData[0])):
                self.slopes[y][x] = self.sobelNormaln11(x * self.deltaX, y * self.deltaY, self.heightIntensity)
                self.remappedSlopes[y][x] = (self.remap(self.slopes[y][x][0], 1, -1, 1, 0),
                                             self.remap(self.slopes[y][x][1], 1, -1, 1, 0),
                                             self.remap(self.slopes[y][x][2], 1, -1, 1, 0)); #remaps from [-1,1] to [0,1] * 255

        if mode == 1:
            # if mode is 1, then grayscale image
            # fill the canvas
            self.tile = {}
            self.tilesize = tilesize = 32
            xsize, ysize = image.size
            for x in range(0, xsize, tilesize):
                for y in range(0, ysize, tilesize):
                    box = x, y, min(xsize, x + tilesize), min(ysize, y + tilesize)
                    tile = ImageTk.PhotoImage(image.crop(box))
                    self.create_image(x, y, image=tile, anchor=tkinter.NW)
                    self.tile[(x, y)] = box, tile
            self.image = image
            self.bind("<B1-Motion>", self.paint)


        if mode == 0:
            # otherwise, if mode is 0, then show the normal map
            self.tile = {}
            self.tilesize = tilesize = 1
            xsize, ysize = image.size
            img2 = Image.fromarray(self.remappedSlopes, 'RGB')
            #print(self.remappedSlopes)
            #self.image = ImageTk.PhotoImage(img2, image.size)
            for x in range(0, xsize, tilesize):
                for y in range(0, ysize, tilesize):
                    rvalue = int(self.remappedSlopes[y][x][0])
                    gvalue = int(self.remappedSlopes[y][x][1])
                    bvalue = int(self.remappedSlopes[y][x][2])
                    img2.putpixel((x, y), (rvalue, gvalue, bvalue))
                    box = x, y, min(xsize, x + tilesize), min(ysize, y + tilesize)
                    tile = ImageTk.PhotoImage(img2.crop(box))
                    self.create_image(x, y, image=tile, anchor=tkinter.NW)
                    self.tile[(x, y)] = box, tile
            self.image = image
            self.bind();



    def remap(self, OldValue, OldMax, OldMin, NewMax, NewMin):
        return int(255 * (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin)

    def droplet(self, x, y):
        dropPos = [x, y]
        for i in range(0, 200):
            if (dropPos[0] < 0.0 or dropPos[0] >= 1.0 or dropPos[1] < 0.0 or dropPos[1] >= 1.0): return
            slope = self.bilinearSlopeSample(dropPos[0], dropPos[1])
            slope[1] *= -1.0
            self.bilinearTint(dropPos[0], dropPos[1])
            dropPos[0] += slope[0] * self.deltaX
            dropPos[1] += slope[1] * self.deltaY
            # print(dropPos)


        return 0

    def paint(self, event):
        xy = event.x - 10, event.y - 10, event.x + 10, event.y + 10

        self.droplet(event.x * self.deltaX, event.y * self.deltaY)
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
        x = max(0, min(self.maxPx, x))
        y = max(0, min(self.maxPy, y))
        return self.heightData[y][x]


    def slopeSample(self, x, y):
        x = max(0, min(self.maxPx, x))
        y = max(0, min(self.maxPy, y))
        return self.slopes[y][x]


    def bilinearTint(self, x, y):
        # self.image.putpixel((ix, iy), (int(norsnippet[y][x][0]), int(norsnippet[y][x][1]), int(norsnippet[y][x][2])))
        fx0 = clamp(int(x * len(self.heightData[0])), 0, self.maxPx)
        fy0 = clamp(int(y * len(self.heightData)), 0, self.maxPy)
        fx1 = min(fx0 + 1, self.maxPx)
        fy1 = min(fy0 + 1, self.maxPy)
        tx = x * float(len(self.slopes[0]))
        ty = y * float(len(self.slopes))
        ty -= floor(ty)
        tx -= floor(tx)

        p00 = pxToFloat(self.image.getpixel((fx0, fy0)))
        p10 = pxToFloat(self.image.getpixel((fx0, fy1)))
        p11 = pxToFloat(self.image.getpixel((fx1, fy1)))
        p01 = pxToFloat(self.image.getpixel((fx1, fy0)))

        # tint red
        p00[1] *= (1.0 - tx) * (1.0 - ty)
        p00[2] *= (1.0 - tx) * (1.0 - ty)
        p01[1] *= (tx) * (1.0 - ty)
        p01[2] *= (tx) * (1.0 - ty)
        p10[1] *= (1.0 - tx) * (ty)
        p10[2] *= (1.0 - tx) * (ty)
        p11[1] *= (tx) * (ty)
        p11[2] *= (tx) * (ty)

        self.image.putpixel((fx0, fy0), (int(p00[0] * 255.0), int(p00[1] * 255.0), int(p00[2] * 255.0)))
        self.image.putpixel((fx1, fy0), (int(p01[0] * 255.0), int(p01[1] * 255.0), int(p01[2] * 255.0)))
        self.image.putpixel((fx0, fy1), (int(p10[0] * 255.0), int(p10[1] * 255.0), int(p10[2] * 255.0)))
        self.image.putpixel((fx1, fy1), (int(p11[0] * 255.0), int(p11[1] * 255.0), int(p11[2] * 255.0)))


    def bilinearSample(self, x, y):
        fx0 = clamp(int(x * len(self.heightData[0])), 0, self.maxPx)
        fy0 = clamp(int(y * len(self.heightData)), 0, self.maxPy)
        fx1 = min(fx0 + 1, self.maxPx)
        fy1 = min(fy0 + 1, self.maxPy)
        h00 = self.heightSample(fx0, fy0)
        h01 = self.heightSample(fx1, fy0)
        h10 = self.heightSample(fx0, fy1)
        h11 = self.heightSample(fx1, fy1)

        tx = x * len(self.heightData[0]) - float(fx0)
        ty = y * len(self.heightData) - float(fy0)
        h0 = tx * h01 + (1.0 - tx) * h00
        h1 = tx * h11 + (1.0 - tx) * h10
        return ty * h1 + (1.0 - ty) * h0

    def bilinearSlopeSample(self, x, y):
        fx0 = clamp(int(x * len(self.heightData[0])), 0, self.maxPx)
        fy0 = clamp(int(y * len(self.heightData)), 0, self.maxPy)
        fx1 = min(fx0 + 1, self.maxPx)
        fy1 = min(fy0 + 1, self.maxPy)
        h00 = self.slopeSample(fx0, fy0)
        h01 = self.slopeSample(fx1, fy0)
        h10 = self.slopeSample(fx0, fy1)
        h11 = self.slopeSample(fx1, fy1)

        tx = x * float(len(self.slopes[0]))
        ty = y * float(len(self.slopes))
        ty -= floor(ty)
        tx -= floor(tx)

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
        return nor


#
# main

if len(sys.argv) != 2:
    print("Usage: painter file")
    sys.exit(1)

root = tkinter.Tk()
root.title("CIS 660 Authoring Tool")

im = Image.open(sys.argv[1])

if im.mode != "RGB":
    im = im.convert("RGB")

nb = ttk.Notebook(root)
page1 = tkinter.Frame(nb)
page2 = tkinter.Frame(nb)
nb.add(page1, text='Paint')
nb.add(page2, text='orig image')
nb.pack(expand=1, fill="both")

PaintCanvas(page1, im, 1).pack() # mode 1 = paint droplets
PaintCanvas(page2, im, 0).pack() # mode 2 = show normal map

root.mainloop()
