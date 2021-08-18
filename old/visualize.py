
from Tkinter import *
import numpy as np
import time

root = Tk()
myCanvas = Canvas(root,width= 700)
myCanvas.pack()

def create_circle(x, y, r, canvasName): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1)

layer_1 = [create_circle(25, 25 + i*50, 20, myCanvas) for i in range(5)]
layer_2 = [create_circle(125, 75 + i*50, 20, myCanvas) for i in range(3)]
layer_3 = [create_circle(225, 100 + i*50, 20, myCanvas) for i in range(2)]
layer_4 = [create_circle(325, 75 + i*50, 20, myCanvas) for i in range(3)]
layer_5 = [create_circle(425, 25 + i*50, 20, myCanvas) for i in range(5)]
layers = [layer_1,layer_2,layer_3,layer_4,layer_5]
def show_activations():
    for layer in layers:
        for node in layer:
	    myCanvas.itemconfigure(node, fill="gray{}".format(np.random.randint(1,99)))
        time.sleep(.1)
    root.after(250,show_activations)
	
root.after(0,show_activations)
root.mainloop()


