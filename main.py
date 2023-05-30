from pgmpy.models import BayesianModel, FactorGraph
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.factors.discrete import JointProbabilityDistribution, DiscreteFactor
from collections import OrderedDict

from pathlib import Path
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from itertools import combinations

def calcHistBoundBox(boundBox):
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    v_ranges = [0, 256]

    h_bins = 50
    s_bins = 60
    v_bins = 60

    boundBox = cv2.resize(boundBox, (500, 500))

    histH = cv2.calcHist([boundBox],[0], None, [h_bins], h_ranges, accumulate=False)
    cv2.normalize(histH, histH, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    histS = cv2.calcHist([boundBox],[1], None, [s_bins], s_ranges, accumulate=False)
    cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    histV = cv2.calcHist([boundBox],[2], None, [v_bins], v_ranges, accumulate=False)
    cv2.normalize(histV, histV, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    return [histH, histS]

def compareHistBoundBox(boundBoxesCurrentHist, boundBoxesPreviousHist, factorGraph, newObjProb):

    for counterCurr, histCurr in enumerate(boundBoxesCurrentHist):
        similarityVect = []
        for counterPrev, histPrev in enumerate(boundBoxesPreviousHist):
            histCompH = cv2.compareHist(histPrev[0], histCurr[0], cv2.HISTCMP_CORREL)
            histCompS = cv2.compareHist(histPrev[1], histCurr[1], cv2.HISTCMP_CORREL)

            similarity = (histCompH + histCompS)/2
            if similarity <= 0.0:
                similarity = 0.01
            similarityVect.append(similarity)

        factor = DiscreteFactor([str(counterCurr)], [len(boundBoxesPreviousHist) + 1], [[newObjProb] + similarityVect])
        factorGraph.add_factors(factor)
        factorGraph.add_edge(str(counterCurr),factor)


def computeProbability(imagePath, boundingBoxPath):
    imagesPath = imagePath
    boundingBoxFile = boundingBoxPath
    boundingBoxNumberPrev = 0
    histogramsCurrent = []
    probNewObject = 0.3
    noBB = 0
    histogramsPrevious = []

    for imageNumber in range(len(imagesPath)):
        coordinatesBoundingBoxes = []
        nodes = []
        factorGraph = FactorGraph()
        _ = boundingBoxFile.readline().rstrip("\n")
        image = cv2.imread(str(imagePath[imageNumber]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        boundingBoxNumber = boundingBoxFile.readline().rstrip("\n")

        histogramsPrevious = histogramsCurrent
        boundingBoxNumberPrev = boundingBoxNumber

        if boundingBoxNumber == "0":
            print('')
            noBB = 1
            continue
        histogramsCurrent = []

        for bbObject in range(int(float(boundingBoxNumber))):
            nodes.append(str(bbObject))
            coordinates = boundingBoxFile.readline().rstrip("\n").split(" ")
            coordinatesBoundingBoxes.append(coordinates)
            x = int(float(coordinates[0]))
            y = int(float(coordinates[1]))
            w = int(float(coordinates[2]))
            h = int(float(coordinates[3]))
            histogramsCurrent.append(calcHistBoundBox(image[y:y+h, x:x+w]))

        factorGraph.add_nodes_from(nodes)

        if noBB == 1:
            noBB = 0
            for i in range(int(float(boundingBoxNumber))):
                print("-1", end=" ")
            continue

        matrixSize = int(float(len(histogramsPrevious)))+1
        nodesPossibilityMatrix = np.ones((matrixSize, matrixSize))

        nodesPossibilityMatrix = [0 if k == l else nodesPossibilityMatrix[k][l] for k in range(matrixSize) for l in range(matrixSize)]
        nodesPossibilityMatrix[0][0] = 1

        if histogramsPrevious != 0:
            compareHistBoundBox(histogramsCurrent, histogramsPrevious, factorGraph, probNewObject)

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

            result = (beliefPropagation.map_query(factorGraph.get_variable_nodes(),show_progress=False))
            orderedResult = OrderedDict(sorted(result.items()))
            result = list(orderedResult.values())
            probabilityResult = []
            for i in range(len(result)):
                value = result[i] - 1
                probabilityResult.append(value)
            print(*probabilityResult,sep = ' ')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    dataDir = Path(args.data_dir)
    imagesDir = Path(os.path.join(str(dataDir) + '/', 'frames'))
    boundingBoxDir = Path(os.path.join(dataDir, 'bboxes.txt'))

    imagesPath = sorted([image_path for image_path in imagesDir.iterdir() if image_path.name.endswith('.jpg')])

    file = open(boundingBoxDir, 'r')

    computeProbability(imagesPath, file)