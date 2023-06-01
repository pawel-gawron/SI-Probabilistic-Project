import cv2
import numpy as np
from typing import List

class BoundBox:
    def __init__(self, coordinates: List, image: cv2.Mat) -> None:
        self.coordinates_ = coordinates
        self.nodes_ = []
        self.image_ = image
        self.boundBoxStorage_ = []

    def computeCoordinates(self, coordinates: List) -> None:
        self.x_ = int(float(coordinates[0]))
        self.y_ = int(float(coordinates[1]))
        self.w_ = int(float(coordinates[2]))
        self.h_ = int(float(coordinates[3]))

    def computeBoundBox(self) -> None:
        for iterator, coordinates in enumerate(self.coordinates_):
            self.nodes_.append(str(iterator))
            self.computeCoordinates(coordinates)
            # self.boundBoxStorage_.append(self.image_[self.y_ + int(self.h_/4) : self.y_ + int(3*(self.h_)/4),
            #                                          self.x_ + int(self.w_/4) : self.x_ + int(3*(self.w_)/4)])
            self.boundBoxStorage_.append(self.image_[self.y_:self.y_+self.h_, self.x_:self.x_+self.w_])

    def returnBoundBox(self) -> List:
        return self.boundBoxStorage_
    
    def returnNodes(self) -> List:
        return self.nodes_
