# First we need to import the packages we will be using. We will use numpy
# for generic matrix operations and tensorflow for deep learning operations
# such as convolutions, pooling and training (backpropagation).
from datetime import datetime
import pathlib
import sys

from clearml import Task
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Next we define a function that can be used to build a neural network. The
# neural network is a simple CNN (convolutional neural network) used for
# classification. The structure of the network is not important for this
# exercise, you can instead see it as a black box that can be trained to
# classify an input image.
def create_cnn(input_shape, output_classes):
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(output_classes, activation="softmax"),
        ]
    )


# The neural network will be trained on a digit classification dataset called
# MNIST. This code downloads and loads the images together with their true
# labels. The code also does some preprocessing of the data to make it more
# suitable for a neural network.
def get_mnist_data(dataset_path):
    # Load the images and read the label from the folder name
    x, y = zip(*map(
            lambda path: (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), int(path.parent.name)),
            dataset_path.rglob("*.jpg")
    ))

    # Convert the list of 2D numpy arrays to a 3D numpy array
    x = np.stack(x, axis=0)

    # Scale images to the [0, 1] range
    x = x.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x = np.expand_dims(x, -1)

    # convert class vectors to binary class matrices (one-hot)
    num_classes = 10
    y = keras.utils.to_categorical(y, num_classes)

    # Shuffle the data since rglob sorts the input on Unix systems
    idx = list(range(len(x)))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


# Finally we will train the network on the data to teach it how to classify a
# digit. We create a model which expects a 28x28 pixel monocolor image since
# this is the format the images in the *MNIST* dataset are. We then create an
# optimizer and calls the `fit()` method to start the training.
def train(dataset_dir, batch_size, epochs):
    task = Task.init(project_name='MNIST project', task_name='train')

    # Get the training data
    print("Loading the training data...")
    x, y = get_mnist_data(pathlib.Path(dataset_dir))

    # Create a Convolutional Neural Network that
    # expects a 28x28 pixel image with 1 color chanel (gray) as input
    model = create_cnn((28, 28, 1), 10)

    print("Training the model...")
    model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
    )

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
            x, y,
            batch_size=batch_size, epochs=epochs,
            validation_split=0.1,
            callbacks=[tensorboard_callback],
    )


if __name__ == "__main__":
    train(sys.argv[1], batch_size=128, epochs=15)
