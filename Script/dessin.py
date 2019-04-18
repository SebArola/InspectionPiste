from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

width = 200
height = 200
center = height//2
white = (255, 255, 255)
green = (0,128,0)
global indice
indice = 101


def saveCercle():
    global indice
    global image1
    global draw
    for i in range(10):
        filename = "train/cercle/cercle_"+str(indice)+".png"
        image1.save(filename)
        indice+=1
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.delete("all")

def savePasCercle():
    global indice
    global image1
    global draw
    for i in range(10):
        filename = "train/pas_cercle/pas_cercle_"+str(indice)+".png"
        image1.save(filename)
        indice+=1
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.delete("all")

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

if __name__ == "__main__":

    root = Tk()

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    global image1
    global draw
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings (visible)
    # cv.create_line([0, center, width, center], fill='green')

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # do the PIL image/draw (in memory) drawings
    # draw.line([0, center, width, center], green)

    # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
    # filename = "my_drawing.png"
    # image1.save(filename)
    buttonSC=Button(text="save cercle",command=saveCercle)
    buttonSC.pack()
    buttonSPC=Button(text="save pas cercle",command=savePasCercle)
    buttonSPC.pack()
    root.mainloop()
