import argparse
import cv2 as cv
from resnet18.detection import detect

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--landmarks_number', default=98)
parser.add_argument('-i', '--image_path', default='./../Adrien_Brody.png')
args = parser.parse_args()

if __name__ == '__main__':
    landmarks_number = args.landmarks_number
    image_path = args.image_path

    image = cv.imread(image_path)
    image = detect(image, landmarks_number)

    cv.imwrite("./../result.png", image)
