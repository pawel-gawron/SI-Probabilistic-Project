from pgmpy.models import FactorGraph
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from collections import OrderedDict
from histogram import Histogram

from pathlib import Path
import numpy as np
import cv2
import os
import argparse
from itertools import combinations
from boundBox import BoundBox

def computeProbability(imagePath, boundingBoxPath):
    imagesPath = imagePath
    boundingBoxFile = boundingBoxPath
    histogramsCurrent = []
    boundBoxCurrent = []
    probNewObject = 0.3
    noBB = 0
    histogramsPrevious = []

    for imageNumber in range(len(imagesPath)):
        nodes = []
        coordinatesBoundingBoxes = []
        factorGraph = FactorGraph()
        _ = boundingBoxFile.readline().rstrip("\n")
        image = cv2.imread(str(imagePath[imageNumber]), cv2.IMREAD_UNCHANGED)

        boundingBoxNumber = boundingBoxFile.readline().rstrip("\n")

        histogramsPrevious = histogramsCurrent

        if boundingBoxNumber == "0":
            print('')
            noBB = 1
            continue
        histogramsCurrent = []
        boundBoxCurrent = []

        for _ in range(int(float(boundingBoxNumber))):
            coordinatesBoundingBoxes.append(boundingBoxFile.readline().rstrip("\n").split(" "))

        boundBox = BoundBox(coordinatesBoundingBoxes, image)
        boundBox.computeBoundBox()

        boundBoxCurrent = boundBox.returnBoundBox()
        nodes = boundBox.returnNodes()

        factorGraph.add_nodes_from(nodes)

        histogram = Histogram(boundBoxCurrent, probNewObject, factorGraph)
        histogramsCurrent = histogram.calcHistBoundBox()

        if noBB == 1:
            noBB = 0
            for i in range(int(float(boundingBoxNumber))):
                print("-1", end=" ")
            continue

        matrixSize = int(float(len(histogramsPrevious)))+1
        nodesPossibilityMatrix = np.ones((matrixSize, matrixSize))

        nodesPossibilityMatrix = [[0 if row == column else nodesPossibilityMatrix[row][column] for row in range(matrixSize)] for column in range(matrixSize)]
        nodesPossibilityMatrix[0][0] = 1

        if histogramsPrevious != 0:
            factorGraph = histogram.compareHistBoundBox(histogramsCurrent, histogramsPrevious)

            for currentHistrogram, prevHistrogram in combinations(range(int(boundingBoxNumber)), 2):
                factor = DiscreteFactor([str(currentHistrogram), str(prevHistrogram)], [matrixSize,
                                                                                        matrixSize],
                                                                                        nodesPossibilityMatrix)
                factorGraph.add_factors(factor)
                factorGraph.add_edge(str(currentHistrogram), factor)
                factorGraph.add_edge(str(prevHistrogram), factor)

            # print(factorGraph)
            # break

            beliefPropagation = BeliefPropagation(factorGraph)
            beliefPropagation.calibrate()

            result = (beliefPropagation.map_query(factorGraph.get_variable_nodes()))
            sorted_keys = sorted(result.keys())
            values = [result[key] for key in sorted_keys]
            print(*([x - 1 for x in values]), sep=" ")
            # with open('myScore.txt', 'a') as file:
            #     file.write(' '.join(map(str, [x - 1 for x in values])) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    dataDir = Path(args.data_dir)
    imagesDir = Path(os.path.join(str(dataDir), 'frames'))
    boundingBoxDir = Path(os.path.join(dataDir, 'bboxes.txt'))

    imagesPath = sorted([image_path for image_path in imagesDir.iterdir() if image_path.name.endswith('.jpg')])

    file = open(boundingBoxDir, 'r')

    computeProbability(imagesPath, file)