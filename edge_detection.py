import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


image = cv2.imread("mosque.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray)
