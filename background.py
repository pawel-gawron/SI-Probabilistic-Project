import cv2
import numpy as np
from typing import List

class Background:
    def __init__(self) -> None:
        self.image_ = []
        self.imageMeanH = []
        self.imageMeanS = []
        self.imageMeanGray = []

    def calcMeanChannel(self, image: cv2.Mat) -> List:
        avgH = 0
        avgS = 0
        avgGray = 0

        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.imageMeanH.append(imageHSV[:,:,0])
        self.imageMeanS.append(imageHSV[:,:,1])
        self.imageMeanGray.append(imageGray)

        if len(self.imageMeanGray) >= 5:
            self.imageMeanGray.pop()
        if len(self.imageMeanH) >= 5:
            self.imageMeanH.pop()
        if len(self.imageMeanS) >= 5:
            self.imageMeanS.pop()

        for imageH in range(len(self.imageMeanH)):
            avgH += self.imageMeanH[imageH].mean(axis=0).mean(axis=0)

        for imageS in range(len(self.imageMeanS)):
            avgS += self.imageMeanS[imageS].mean(axis=0).mean(axis=0)

        for imageGray in range(len(self.imageMeanGray)):
            avgGray += self.imageMeanGray[imageGray].mean(axis=0).mean(axis=0)

        return [avgH, avgS, avgGray]

