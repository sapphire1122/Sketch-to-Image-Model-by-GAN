import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import network
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(600, 500)
        self.setWindowTitle("sketch-to-image model")

        self.label = QLabel(self)
        self.label.setText("                  show your image")
        self.label.setFixedSize(200, 250)
        self.label.move(50, 150)

        self.label2 = QLabel(self)
        self.label2.setText("                  show your result")
        self.label2.setFixedSize(200, 250)
        self.label2.move(350, 150)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label2.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn = QPushButton(self)
        btn.setText("select image")
        btn.move(80, 30)
        btn.clicked.connect(self.openimage)

        btn2 = QPushButton(self)
        btn2.setText("sketch-to-image")
        btn2.move(380, 20)
        btn2.clicked.connect(self.result)

        btn3 = QPushButton(self)
        btn3.setText("image-to-sketch")
        btn3.move(380, 50)
        btn3.clicked.connect(self.resultImageToSketch)

        btn4 = QPushButton(self)
        btn4.setText("image-to-Vango")
        btn4.move(380, 80)
        btn4.clicked.connect(self.resultImageToVango)

    def openimage(self):
        imgName,imgType = QFileDialog.getOpenFileName(self,  "open picture", "*.jpg;;*.png;;All ""Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    def sketchToImage(self):
        G_A = network.generator(3, 3, 32, 9)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = datasets.ImageFolder('sketch', transform=transform)
        # print(dataset[0][0])
        print(dataset[0][0].size())

        G_A.load_state_dict(torch.load("CUHK_generatorA_param.pkl", map_location="cpu"))
        print(G_A)
        real = dataset[0][0].unsqueeze(0)
        genB = G_A(real)
        print(genB)
        plt.imsave("sketch/outImage.png", (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)


    def ImageToSketch(self):
        G_A = network.generator(3, 3, 32, 9)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = datasets.ImageFolder('imageT', transform=transform)
        # print(dataset[0][0])
        print(dataset[0][0].size())

        G_A.load_state_dict(torch.load("CUHK_generatorB_param.pkl", map_location="cpu"))
        print(G_A)
        real = dataset[0][0].unsqueeze(0)
        genB = G_A(real)
        print(genB)
        plt.imsave("imageT/outSketch.png", (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    def imageToVango(self):
        G_A = network.generator(3, 3, 32, 9)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = datasets.ImageFolder('vangotest', transform=transform)
        # print(dataset[0][0])
        print(dataset[0][0].size())

        G_A.load_state_dict(torch.load("vangogh2photo_generatorB_param.pkl", map_location="cpu"))
        print(G_A)
        real = dataset[0][0].unsqueeze(0)
        genB = G_A(real)
        print(genB)
        plt.imsave("vangotest/outVango.png", (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    def result(self):
        self.sketchToImage()
        #imgName, imgType = QFileDialog.getOpenFileName(self, "open picture", "", "*.jpg;;*.png;;All Files(*)")
        jpg2 = QtGui.QPixmap('sketch/outImage.png').scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(jpg2)

    def resultImageToSketch(self):
        self.ImageToSketch()
        # imgName, imgType = QFileDialog.getOpenFileName(self, "open picture", "", "*.jpg;;*.png;;All Files(*)")
        jpg2 = QtGui.QPixmap('imageT/outSketch.png').scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(jpg2)

    def resultImageToVango(self):
        self.imageToVango()
        # imgName, imgType = QFileDialog.getOpenFileName(self, "open picture", "", "*.jpg;;*.png;;All Files(*)")
        jpg2 = QtGui.QPixmap('vangotest/outVango.png').scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(jpg2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())

