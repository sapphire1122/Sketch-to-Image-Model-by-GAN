import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import network
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


window = tk.Tk()
window.title('sketch to image')
window.geometry('300x150')

path_var = tk.StringVar()
entry = tk.Entry(window, textvariable=path_var)
entry.place(x=10, y=10, anchor='nw')
file_name = ''

def click():
    filetypes = [("sketch", "*.jpg"),("sketch", "*.png")]
    file_name = filedialog.askopenfilename(title='select a file',
                                  filetypes=filetypes,
                                  initialdir='./'
                                             )
    path_var.set(file_name)



def sketchToImage(filename):
    G_A = network.generator(3,3, 32,9)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset=datasets.ImageFolder(filename, transform=transform)
    #print(dataset[0][0])
    print(dataset[0][0].size())



    G_A.load_state_dict(torch.load("CUHK_generatorA_param.pkl", map_location="cpu"))
    print(G_A)
    real = dataset[0][0].unsqueeze(0)
    genB = G_A(real)
    print(genB)
    plt.imsave("a/out.png", (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

def show(filename):
    canvas = tk.Canvas(window, width=300, height=300)
    canvas.pack()
    #vtn.iconbitmap("logo.ico")
    img = Image.open('1_input.jpg')
    imagen = ImageTk.PhotoImage(img)
    canvas.create_image(20,20, anchor='nw', image=imagen)

selectButton = tk.Button(window, text='select', command=click)
selectButton.place(x=100, y=10, anchor='nw')


selectButton2 = tk.Button(window,text='sketch-to-image', command=sketchToImage('a'))
selectButton2.place(x=100, y=30, anchor='nw')

selectButton3 = tk.Button(window, text='show result', command=show('a'))
selectButton3.place(x=100, y=50, anchor='nw')

window.mainloop()