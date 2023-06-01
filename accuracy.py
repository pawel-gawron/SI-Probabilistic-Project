from pathlib import Path
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from itertools import combinations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    dataDir = Path(args.data_dir)
    imagesDir = Path(os.path.join(str(dataDir) + '/', 'frames'))
    boundingBoxDir = Path(os.path.join(dataDir, 'bboxes_gt.txt'))


    file = open(boundingBoxDir, 'r')
    result = []

    imagesPath = sorted([image_path for image_path in imagesDir.iterdir() if image_path.name.endswith('.jpg')])

    for imageNumber in range(len(imagesPath)):
        _ = file.readline().rstrip("\n")
        boundingBoxNumber = file.readline().rstrip("\n")

        solution = []

        for bbObject in range(int(float(boundingBoxNumber))):
            coordinates = file.readline().rstrip("\n").split(" ")
            solution.append(coordinates[0])
            # print(coordinates[0], end=' ')
        print(*solution,sep = ' ')
        result.append(solution)

    with open('accuracy.txt', 'a') as file:
        file.write('\n'.join(' '.join(sublist) for sublist in result) + '\n')

        # print("")
        # if imageNumber >=400:
        #     break
