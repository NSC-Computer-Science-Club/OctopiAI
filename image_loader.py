import cv2
import os
import random
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split


IMG_WIDTH = 100 # Width of the image
IMG_HEIGHT = 100 # Height of the image

def load_data(data_dir, num_categories):
    """
    Returns a tuple contaiining a list of images, its corresponding label,
    and the index of the prediction image
    """
    images = []
    labels = []
    for i in range(num_categories):
        # Progress bar for loading data
        #sys.stdout.write('\r')
        #sys.stdout.write("Loading data... [%-43s] %d / %d" % ('='*i, i, NUM_CATEGORIES))
        #sys.stdout.flush()
        directory = os.path.join(data_dir, str(i))
        # Finds all files in a directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for fileName in files:
            if ".jpg" in fileName or ".png" in fileName:
                img = cv2.imread(os.path.join(directory, fileName))
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(i)
            else:
                print("Failed: " + fileName)
    return (images, labels)  


def load_one_image(data_dir, category):
    """
    Load a random image from the specified category
    """
    directory = os.path.join(data_dir, str(category))
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    image_file_name = random.choice(files)
    img = cv2.imread(os.path.join(directory, image_file_name))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img

def load_one_image(directory):
    img = cv2.imread(directory)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img