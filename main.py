from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

from pathlib import Path
import numpy as np
import cv2
import os
import argparse

def computeProbability(imagePath, boundingBoxPath):
    imagesPath = imagePath
    boundingBoxFile = boundingBoxPath

    for i in range(len(imagesPath)):
        coordinatesBoundingBoxes = []
        imageName = boundingBoxFile.readline().rstrip("\n")
        image = cv2.imread(str(imagePath[i]))


        # print(imageName)

        if not imageName:
            break

        boundingBoxNumber = boundingBoxFile.readline().rstrip("\n")
        # print(boundingBoxNumber)

        for bb in range(int(boundingBoxNumber)):
            coordinates = boundingBoxFile.readline().rstrip("\n").split(" ")
            coordinatesBoundingBoxes.append(coordinates)
            x = int(float(coordinates[0]))
            y = int(float(coordinates[1]))
            w = int(float(coordinates[2]))
            h = int(float(coordinates[3]))
            cv2.imshow("image", image[y:y+h, x:x+w])
            cv2.waitKey(0)
            cv2.destroyAllWindows


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    dataDir = Path(args.data_dir)
    imagesDir = Path(os.path.join(dataDir, 'frames'))
    boundingBoxDir = Path(os.path.join(dataDir, 'bboxes.txt'))

    imagesPath = sorted([image_path for image_path in imagesDir.iterdir() if image_path.name.endswith('.jpg')])

    file = open(boundingBoxDir, 'r')

    computeProbability(imagesPath, file)
    # for image_path in imagesPath:
    #     image = cv2.imread(str(image_path))
    #     if image is None:
    #         print(f'Error loading image {image_path}')
    #         continue