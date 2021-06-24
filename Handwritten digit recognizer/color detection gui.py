import cv2
import numpy as np
import pandas as pd
import argparse



#Reading image with opencv
img = cv2.imread("image", cv2.IMREAD_COLOR)
cv2.imshow("image",img)
