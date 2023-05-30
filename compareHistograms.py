import cv2
from pgmpy.factors.discrete import DiscreteFactor

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