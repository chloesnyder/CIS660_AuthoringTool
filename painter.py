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
from math import sqrt, acos, floor, fabs

import Tkinter, Tkconstants, tkFileDialog




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


def mix(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return t * b + (1.0 - t) * a


# image processing

def pxToFloat(a):
    return [float(a[0]) / 255.0, float(a[1]) / 255.0, float(a[2]) / 255.0]


def bilinearSample(image, x, y):
    lx = len(image[0])
    ly = len(image)
    fx0 = clamp(int(x * lx), 0, lx - 1)
    fy0 = clamp(int(y * ly), 0, ly - 1)
    fx1 = min(fx0 + 1, lx - 1)
    fy1 = min(fy0 + 1, ly - 1)
    h00 = image[fy0][fx0]
    h10 = image[fy1][fx0]
    h01 = image[fy0][fx1]
    h11 = image[fy1][fx1]

    tx = x * float(lx)
    ty = y * float(ly)
    ty -= floor(ty)
    tx -= floor(tx)

    h0 = mix(h00, h01, tx)
    h1 = mix(h10, h11, tx)
    return mix(h0, h1, ty)


# data processing
def sobelNormal(heightMap, x, y, strength):
    deltaX = 1.0 / len(heightMap[0])
    deltaY = 1.0 / len(heightMap)

    skx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
    sky = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0]

    gx = 0.0
    gy = 0.0

    k = 0
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            height = bilinearSample(heightMap, x - ox * deltaX, y - oy * deltaY)
            gx += skx[k] * height
            gy += sky[k] * height
            k = k + 1
    gz = 1.0 / strength
    nor = normalize([gx, gy, gz])
    return nor


# convert RGB image into grayscale float array (0, 1)
def getHeightmap(image):
    heightData = numpy.zeros((len(image[0]), len(image)))
    for y in range(0, len(image)):
        for x in range(0, len(image[0])):
            heightData[y][x] = intensity(image[y][x])

    return heightData

# convert Heightmap ([0.0, 1.0]) to normals ([-1.0, +1.0] * 3)
def getNormalMap(height, strength):
    deltaX = 1.0 / len(height[0])
    deltaY = 1.0 / len(height[1])
    normalData = numpy.zeros((len(height[0]), len(height), 3))
    for y in range(0, len(height)):
        for x in range(0, len(height[0])):
            normalData[y][x] = sobelNormal(height, x * deltaX, y * deltaY, strength)

    return normalData


# convert heightmap data into curvature map
def getCurvatureMap(heights, radius):
    curveData = numpy.zeros((len(heights[0]), len(heights)))
    for y in range(0, len(heights)):
        for x in range(0, len(heights[0])):
            for d in range(1, radius+1):
                if(x-d < 0 or (x+d) >= len(heights[0]) or (y-d) < 0 or (y+d) >= len(heights)):
                    pass
                else:
                    h = heights[y][x]
                    hv1 = heights[y-d][x]
                    hv2 = heights[y+d][x]
                    hh1 = heights[y][x-d]
                    hh2 = heights[y][x+d]

                    if hv1 < hv2:
                        vcurviness = (h - hv2) - (hv1 - h)
                    else:
                        vcurviness = (h - hv1) - (hv2 - h)

                    if hh1 < hh2:
                        hcurviness = (h - hh2) - (hh1 - h)
                    else:
                        hcurviness = (h - hh1) - (hh2 - h)

                    curveData[y][x] = (vcurviness + hcurviness)/(2.0)

    return curveData

def getCurvatureMapDrawable(curves):
    curveDraw = numpy.zeros((len(curves[0]), len(curves), 3))
    for y in range(0, len(curves)):
        for x in range(0, len(curves[0])):
            cur = 0.5 * (1.0 + curves[y][x])
            curveDraw[y][x] = [int(round(cur * 255.0))] * 3

    return curveDraw


def getNormalMapDrawable(normals):
    normalDraw = numpy.zeros((len(normals[0]), len(normals), 3))
    for y in range(0, len(normals)):
        for x in range(0, len(normals[0])):
            nor = 0.5 * (1.0 + normals[y][x])
            normalDraw[y][x] = [int(round(nor[0] * 255.0)), int(round(nor[1] * 255.0)), int(round(nor[2] * 255.0))]

    return normalDraw




# Widget

class PaintCanvas(tkinter.Canvas):

    # static data to pass to all canvases
    heightImg = Image.fromarray(numpy.zeros((1, 1, 3)), 'RGB')
    heightData = numpy.array([[0.0]])
    normalImg = Image.fromarray(numpy.array([[[127, 127, 255]]]), 'RGB')
    normalData = numpy.array([[[0.0, 0.0, 1.0]]])
    curveImg = Image.fromarray(numpy.array([[[127, 127, 127]]]), 'RGB')
    curveData = numpy.array([[0.5]])

    def __init__(self, master, image, mode):
        tkinter.Canvas.__init__(self, master, width=image.size[0], height=image.size[1])
        mode = int(mode)

        self.alt = False
        self.shift = False

        self.heightIntensity = 4.0

        # get initial data
        self.imageData = numpy.array(im.convert('RGB'))
        # self.heightData = height
        # self.slopes = slope

        self.deltaX = 1.0 / len(self.imageData[0])
        self.deltaY = 1.0 / len(self.imageData)

        self.maxPx = len(self.imageData[0]) - 1
        self.maxPy = len(self.imageData) - 1

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

        self.mode = mode

        if mode == 1:
            # draw droplets mode
            self.bind("<B1-Motion>", self.paint)
            self.bind("<1>", self.paint)
        elif mode == 0:
            # paint terrain mode
            self.bind("<B1-Motion>", self.addHeight)
            self.bind("<1>", self.addHeight)
        else:
            # invalid
            print("Specified invalid mode: " + repr(mode))
            sys.exit(1)

        self.bind("<Alt_L>", self.altOn)
        self.bind("<KeyRelease-Alt_L>", self.altOff)
        self.bind("<Shift_L>", self.shiftOn)
        self.bind("<KeyRelease-Shift_L>", self.shiftOff)
        self.bind("<Alt_R>", self.altOn)
        self.bind("<KeyRelease-Alt_R>", self.altOff)
        self.bind("<Shift_R>", self.shiftOn)
        self.bind("<KeyRelease-Shift_R>", self.shiftOff)
        self.bind("<Visibility>", self.onVisibility)



    def altOn(self, event):
        self.alt = True

    def altOff(self, event):
        self.alt = False

    def shiftOn(self, event):
        self.shift = True

    def shiftOff(self, event):
        self.shift = False

    def remap(self, OldValue, OldMax, OldMin, NewMax, NewMin):
        return int(255 * (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin)


    def droplet(self, x, y):
        dropPos = [x, y]
        for i in range(0, 200):
            if (dropPos[0] < 0.0 or dropPos[0] >= 1.0 or dropPos[1] < 0.0 or dropPos[1] >= 1.0): return
            slope = bilinearSample(PaintCanvas.normalData, dropPos[0], dropPos[1])

            slope[1] *= -1.0
            self.bilinearTint(dropPos[0], dropPos[1])
            dropPos[0] += slope[0] * self.deltaX
            dropPos[1] += slope[1] * self.deltaY

    def paint(self, event):
        brush_size = brushSizeSlider.get()
        xy = event.x - brush_size, event.y - brush_size, event.x + brush_size, event.y + brush_size
        self.droplet(event.x * self.deltaX, event.y * self.deltaY)
        self.repair(xy)

    def addHeight(self, event):
        brush_size = brushSizeSlider.get()
        xy = event.x - brush_size, event.y - brush_size, event.x + brush_size, event.y + brush_size

        strength = 0.3 if self.shift else (-0.05 if self.alt else 0.05)
        bmode = 'polish' if self.shift and self.alt else ('flatten' if self.shift else 'add')
        self.dataHeightChangeBrush(xy, strength,
                                    falloff='smooth',
                                    operation=bmode)
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

    def onVisibility(self, event):
        self.alt = False
        self.shift = False
        self.tile = {}
        self.tilesize = tilesize = 32
        xsize, ysize = self.image.size
        for x in range(0, xsize, tilesize):
            for y in range(0, ysize, tilesize):
                box = x, y, min(xsize, x + tilesize), min(ysize, y + tilesize)
                tile = ImageTk.PhotoImage(PaintCanvas.heightImg.crop(box)) if self.mode == 1 else ImageTk.PhotoImage(self.image.crop(box))
                self.create_image(x, y, image=tile, anchor=tkinter.NW)
                self.tile[(x, y)] = box, tile
        if (self.mode == 1): self.image = PaintCanvas.heightImg.copy()
        self.update_idletasks()

    def bilinearTint(self, x, y):
        fx0 = clamp(int(x * len(PaintCanvas.heightData[0])), 0, self.maxPx)
        fy0 = clamp(int(y * len(PaintCanvas.heightData)), 0, self.maxPy)
        fx1 = min(fx0 + 1, self.maxPx)
        fy1 = min(fy0 + 1, self.maxPy)
        tx = x * float(len(PaintCanvas.heightData[0]))
        ty = y * float(len(PaintCanvas.heightData))
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


    def dataHeightChangeBrush(self, xy, delta, falloff='smooth', operation='add'):
        deltaX = 1.0 / len(PaintCanvas.heightData[0])
        deltaY = 1.0 / len(PaintCanvas.heightData)
        center = (xy[0] + xy[2]) * 0.5, (xy[1] + xy[3]) * 0.5
        centerVal = PaintCanvas.heightData[clamp(int(round(center[1])), 0, len(PaintCanvas.heightData) - 1)][clamp(int(round(center[0])), 0, len(PaintCanvas.heightData[0]) - 1)]
        centerNor = PaintCanvas.normalData[clamp(int(round(center[1])), 0, len(PaintCanvas.heightData) - 1)][
            clamp(int(round(center[0])), 0, len(PaintCanvas.heightData[0]) - 1)]

        # change the height map
        for y in range(xy[1], xy[3]):
            for x in range(xy[0], xy[2]):
                x1 = clamp(x, 0, len(PaintCanvas.heightData[0]) - 1)
                y1 = clamp(y, 0, len(PaintCanvas.heightData) - 1)

                # get intensity based on falloff, 'else' makes square
                k = 1.0
                if falloff == 'sphere':
                    distX = fabs(float(x1 - center[0])) / float(xy[2] - center[0])
                    distY = fabs(float(y1 - center[1])) / float(xy[3] - center[1])
                    k = max(0.0, 1.0 - sqrt(distX * distX + distY * distY))
                elif falloff == 'linear':
                    distX = fabs(float(x1 - center[0])) / float(xy[2] - center[0])
                    distY = fabs(float(y1 - center[1])) / float(xy[3] - center[1])
                    k = max(0.0, 1.0 - (distX * distX + distY * distY))
                elif falloff == 'smooth':
                    distX = fabs(float(x1 - center[0])) / float(xy[2] - center[0])
                    distY = fabs(float(y1 - center[1])) / float(xy[3] - center[1])
                    k = max(0.0, 1.0 - (distX * distX + distY * distY))
                    k = k * k * (3.0 - 2.0 * k)

                h = PaintCanvas.heightData[y1][x1]
                if operation == 'add':
                    h = clamp(h + k * delta, 0.0, 1.0)
                elif operation == 'flatten':
                    h = mix(h, centerVal, k * delta)
                elif operation == 'polish':
                    h = PaintCanvas.heightData[y1][x1]
                    c = PaintCanvas.curveData[y1][x1]
                    c = 1.0 - clamp(-c, 0.0, 1.0)
                    x0 = deltaX * (center[0] - x1)
                    y0 = deltaX * (center[1] - y1)
                    z0 = centerVal * centerNor[2] / self.heightIntensity
                    z1 = (centerNor[0] * x0 - centerNor[1] * y0 + z0) / centerNor[2] * self.heightIntensity
                    h = mix(h, clamp(z1, 0.0, 1.0), c * k * delta)

                PaintCanvas.heightData[y1][x1] = h

                PaintCanvas.heightImg.putpixel((x1, y1), (int(round(h * 255.0)), int(round(h * 255.0)), int(round(h * 255.0))))

        # change the normal map
        for y in range(xy[1], xy[3]):
            for x in range(xy[0], xy[2]):
                x1 = clamp(x, 0, len(PaintCanvas.heightData[0]) - 1)
                y1 = clamp(y, 0, len(PaintCanvas.heightData) - 1)
                nor = sobelNormal(PaintCanvas.heightData, x * deltaX, y * deltaY, self.heightIntensity)
                PaintCanvas.normalData[y1][x1] = nor
                nor = 0.5 * (1.0 + PaintCanvas.normalData[y1][x1])
                PaintCanvas.normalImg.putpixel((x1, y1), (
                int(round(nor[0] * 255.0)), int(round(nor[1] * 255.0)), int(round(nor[2] * 255.0))))

        #change the curve map
        for y in range(xy[1], xy[3]):
            for x in range(xy[0], xy[2]):
                x1 = clamp(x, 0, len(PaintCanvas.curveData[0]) - 1)
                y1 = clamp(y, 0, len(PaintCanvas.curveData) - 1)
                d = 4
                if (x - d < 0 or (x + d) >= len(PaintCanvas.heightData[0]) or (y - d) < 0 or (y + d) >= len(PaintCanvas.heightData)):
                    pass
                else:
                    h = PaintCanvas.heightData[y1][x1]
                    hv1 = PaintCanvas.heightData[y1 - d][x1]
                    hv2 = PaintCanvas.heightData[y1 + d][x1]
                    hh1 = PaintCanvas.heightData[y1][x1 - d]
                    hh2 = PaintCanvas.heightData[y1][x1 + d]
    
                    if hv1 < hv2:
                        vcurviness = (h - hv2) - (hv1 - h)
                    else:
                        vcurviness = (h - hv1) - (hv2 - h)

                    if hh1 < hh2:
                        hcurviness = (h - hh2) - (hh1 - h)
                    else:
                        hcurviness = (h - hh1) - (hh2 - h)

                    PaintCanvas.curveData[y1][x1] = (vcurviness + hcurviness) / 2.0 #(d * d)
                    cur = 0.5 * (1.0 + PaintCanvas.curveData[y1][x1])
                    PaintCanvas.curveImg.putpixel((x1, y1), (
                    int(round(cur * 255.0)), int(round(cur * 255.0)), int(round(cur * 255.0))))



#
# main

# if len(sys.argv) != 2:
#     print("Usage: painter file")
#     sys.exit(1)

root = tkinter.Tk()
root.title("CIS 660 Authoring Tool")
root.filename = tkFileDialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
print (root.filename)


im = Image.open(root.filename)

if im.mode != "RGB":
    im = im.convert("RGB")

# preprocess the data
imageData = numpy.array(im.convert('RGB'))

heightData = getHeightmap(imageData)
heightDraw = Image.fromarray(numpy.copy(imageData).astype('uint8'), 'RGB')
normalData = getNormalMap(heightData, 4.0)
normalDraw = (Image.fromarray(getNormalMapDrawable(normalData).astype('uint8'), 'RGB'))
curveData = getCurvatureMap(heightData, 4)
curveDraw = (Image.fromarray(getCurvatureMapDrawable(curveData).astype('uint8'), 'RGB'))

# set the shared data for all canvases
PaintCanvas.heightData = heightData
PaintCanvas.heightImg = heightDraw
PaintCanvas.normalData = normalData
PaintCanvas.normalImg = normalDraw
PaintCanvas.curveData = curveData
PaintCanvas.curveImg = curveDraw

nb = ttk.Notebook(root)
page1 = tkinter.Frame(nb)
page2 = tkinter.Frame(nb)
page3 = tkinter.Frame(nb)
page4 = tkinter.Frame(nb)
nb.add(page1, text='Droplets')
nb.add(page2, text='Normal')
nb.add(page3, text='Height')
nb.add(page4, text='Curvature')
nb.pack(expand=1, fill="both")

brushSizeSlider = tkinter.Scale(root, from_=2, to=20, orient=tkinter.HORIZONTAL, label="Brush Size")
brushSizeSlider.pack()

PaintCanvas(page1, im, 1).pack() # mode 1 = paint droplets
PaintCanvas(page2, normalDraw, 0).pack() # mode 2 = show normal map
PaintCanvas(page3, heightDraw, 0).pack()
PaintCanvas(page4, curveDraw, 0).pack()



root.mainloop()
