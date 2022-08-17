# First we need to import the packages we will be using. We will use numpy
# for generic matrix operations and tensorflow for deep learning operations
# such as convolutions, pooling and training (backpropagation).
import wandb
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
def get_mnist_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# Finally we will train the network on the data to teach it how to classify a
# digit. We create a model which expects a 28x28 pixel monocolor image since
# this is the format the images in the *MNIST* dataset are. We then create an
# optimizer and calls the `fit()` method to start the training.
def train(batch_size, epochs):
    wandb.init(project="ai-ml-exercise",
               config={"batch_size": batch_size, "epochs": epochs},
               settings=wandb.Settings(start_method="fork"))

    # Get the training data
    print("Loading the training data...")
    x_train, y_train = get_mnist_data()[0]

    # Create a Convolutional Neural Network that
    # expects a 28x28 pixel image with 1 color chanel (gray) as input
    model = create_cnn((28, 28, 1), 10)

    print("Training the model...")
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1,
              callbacks=[wandb.keras.WandbCallback()])


if __name__ == "__main__":
    train(batch_size=128, epochs=10)
