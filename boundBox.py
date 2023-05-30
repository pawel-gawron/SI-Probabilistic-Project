import cv2
import numpy as np
from typing import List

class BoundBox:
    def __init__(self, coordinates, image) -> None:
        self.coordinates_ = coordinates
        self.nodes_ = []
        self.image_ = image
        self.boundBoxStorage_ = []

    def computeCoordinates(self, coordinates):
        self.x_ = int(float(coordinates[0]))
        self.y_ = int(float(coordinates[1]))
        self.w_ = int(float(coordinates[2]))
        self.h_ = int(float(coordinates[3]))

    def computeBoundBox(self):
        for iterator, coordinates in enumerate(self.coordinates_):
            self.nodes_.append(str(iterator))
            self.computeCoordinates(coordinates)
            self.x_ = int(float(coordinates[0]))
            self.y_ = int(float(coordinates[1]))
            self.w_ = int(float(coordinates[2]))
            self.h_ = int(float(coordinates[3]))
            self.boundBoxStorage_.append(self.image_[self.y_:self.y_+self.h_, self.x_:self.x_+self.w_])

    def returnBoundBox(self):
        return self.boundBoxStorage_
    
    def returnNodes(self):
        return self.nodes_
