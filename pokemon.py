# Dependencies:
# opencv-python
# scikit-learn
# tensorflow
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from image_loader import *

SAVE_MODEL = True
LOAD_MODEL = False
MODEL_NAME = "model.h5"
EVALUATE = True # Evaluate how accurate the model is on the test data
PREDICT = True # Use the model to make a prediction on one image

DATA_DIRECTORY = "pokemon"
POKEMON = ["Charizard", "Turtwig", "Chimchar", "Piplup", "Dedenne"]
NUM_CATEGORIES = len(POKEMON)

# Stuff for the neural network
EPOCHS = 100 # How many times the network "learns"
TEST_SIZE = 0.4 # How much of the data we use for testing (originally 0.4)

def main():
    # Get image arrays and labels for all image files
    images, labels = load_data(DATA_DIRECTORY, NUM_CATEGORIES)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    image_list = np.array(images)
    label_list = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        image_list, label_list, test_size = TEST_SIZE
    )

    # Get a compiled neural network
    if LOAD_MODEL:
        # Load saved model
        print("Loading model...")
        model = keras.models.load_model(MODEL_NAME)
        # Show a summary of the model
        print("Model summary:")
        model.summary()
    else:
        # Create new model
        print("Creating model...")
        model = get_model()
        # Show a summary of the model
        print("Model summary:")
        model.summary()
        # Fit model on training data
        print("Training model...")
        model.fit(x_train, y_train, epochs=EPOCHS)
        # Save model to file
        if SAVE_MODEL:
            model.save(MODEL_NAME)
            print(f"Model saved to {MODEL_NAME}.")

    if EVALUATE:
        # Evaluate neural network performance
        print("Evaluate how accurate the model is:")
        model.evaluate(x_test,  y_test, verbose=1)

    if PREDICT:
        while (True):
            predict(model)

def predict(model):
    # Make sure user input is good
    good_input: bool = False
    actual_image_path: str = ""
    while not good_input:
        print("\nPossible pokemon to make predictions on: ")
        for mon in POKEMON:
            print(mon)
        actual_image_path: str = input("Enter the path of the image to predict: ")
        if os.path.isfile(actual_image_path) and any(extension in actual_image_path for extension in [".jpg", ".jpeg", ".png"]):
            good_input = True
        else:
            print("Path/file is invalid. Did you enter the path correctly? (Note: Accepts JPGs and PNGs only.)")
    
    actual_image = load_one_image(actual_image_path)
    # Make prediction on that image
    prediction_list = model.predict(np.array([actual_image]))
    prediction = prediction_list[0]
    prediction_index = 0
    top_prediction = 0
    
    # Display Information for prediction vs actual
    for i in range(NUM_CATEGORIES):
        if prediction[i] > top_prediction:
            top_prediction = prediction[i]
            prediction_index = i
            
    prediction_category = POKEMON[prediction_index]
    
    print("Prediction: " + prediction_category)
    
    # Display an image of the prediction vs actual
    #print("Images are displayed in a separate window")
    #directory: str = os.path.join(DATA_DIRECTORY, str(prediction_index), "0.jpg")
    #prediction_image = cv2.imread(directory)
    #prediction_image = cv2.resize(prediction_image, (100, 100))
    #actual_image = cv2.resize(actual_image, (100, 100))
    #both_images = np.hstack((prediction_image, actual_image))
    #cv2.imshow("Prediction vs Actual", both_images)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

# Play around with this!
# https://www.tensorflow.org/guide/keras/sequential_model
def get_model():
    """
    Returns a compiled convolutional neural network model
    """
    # Get a model object
    model = keras.Sequential()

    # Input layer
    # IMG_WIDTH and IMG_HEIGHT from image_loader.py
    model.add(keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Apply filters
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))

    # Shrink the image to simplify it more
    model.add(layers.MaxPooling2D(pool_size=2))

    # Apply more filters
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))

    # Shrink it even more
    model.add(layers.MaxPooling2D(pool_size=2))

    # Combine filters into one
    model.add(layers.Flatten())

    # Run it through nodes
    model.add(layers.Dense(128, activation="relu"))

    # Only test with half the nodes at a time; so that you don't
    # overcorrect to incorrectly specific data
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
