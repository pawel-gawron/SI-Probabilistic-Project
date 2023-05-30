import cv2
import numpy as np
from typing import List
from pgmpy.factors.discrete import DiscreteFactor

class Histogram:
    def __init__(self, boundBox: List, newObjProb: float, factorGraph: DiscreteFactor) -> None:
        self.boundBox_ = boundBox
        self.newObjProb_ = newObjProb
        self.factorGraph_ = factorGraph

    def calcHistBoundBox(self) -> List:
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        gray_ranges = [0, 256]

        h_bins = 50
        s_bins = 60
        gray_bins = 60

        hist = []

        for iterator, boundbox in enumerate(self.boundBox_):

            boundbox = cv2.resize(boundbox, (500, 500))

            boundboxHSV = cv2.cvtColor(boundbox, cv2.COLOR_BGR2HSV)
            boundboxGray = cv2.cvtColor(boundbox, cv2.COLOR_BGR2GRAY)


            histH = cv2.calcHist([boundboxHSV],[0], None, [h_bins], h_ranges, accumulate=False)
            cv2.normalize(histH, histH, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

            histS = cv2.calcHist([boundboxHSV],[1], None, [s_bins], s_ranges, accumulate=False)
            cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

            histGray = cv2.calcHist([boundboxGray],[0], None, [gray_bins], gray_ranges, accumulate=False)
            cv2.normalize(histGray, histGray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

            hist.append([histH, histS, histGray])

        return hist
    
    def compareHistBoundBox(self, boundBoxesCurrentHist: List, boundBoxesPreviousHist: List) -> DiscreteFactor:

        for counterCurr, histCurr in enumerate(boundBoxesCurrentHist):
            similarityVect = []
            for counterPrev, histPrev in enumerate(boundBoxesPreviousHist):
                histCompH = cv2.compareHist(histPrev[0], histCurr[0], cv2.HISTCMP_CORREL)
                histCompS = cv2.compareHist(histPrev[1], histCurr[1], cv2.HISTCMP_CORREL)
                histCompGray = cv2.compareHist(histPrev[2], histCurr[2], cv2.HISTCMP_CORREL)

                similarity = (histCompH + histCompS + histCompGray)/3
                if similarity <= 0.0:
                    similarity = 0.01
                similarityVect.append(similarity)

            factor = DiscreteFactor([str(counterCurr)], [len(boundBoxesPreviousHist) + 1], [[self.newObjProb_] + similarityVect])
            self.factorGraph_.add_factors(factor)
            self.factorGraph_.add_edge(str(counterCurr),factor)

        return self.factorGraph_
