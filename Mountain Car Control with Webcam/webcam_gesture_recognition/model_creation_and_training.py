# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense


def build_model():
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()

    # First convolution layer and pooling
    cnn.add(Conv2D(filters=32, kernel_size=2, activation='relu', input_shape=(64, 64, 3)))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Dropout(0.25))

    # Second convolution layer and pooling (input_shape is going to be the pooled feature maps from the previous
    # convolution layer)
    cnn.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Dropout(0.25))

    # Third convolution layer and pooling (input_shape is going to be the pooled feature maps from the previous
    # convolution layer)
    cnn.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Dropout(0.25))

    # Fourth convolution layer and pooling (input_shape is going to be the pooled feature maps from the previous
    # convolution layer)
    cnn.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Dropout(0.25))

    # Flattening the layers
    cnn.add(Flatten())

    # Adding a fully connected layer
    cnn.add(Dense(units=516, activation='relu'))

    # Adding output Layer (softmax for more than 2 classes)
    cnn.add(Dense(units=3, activation='softmax'))

    # Compiling the CNN
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn


def preprocess_data():
    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('dataset2/train', target_size=(64, 64), batch_size=32,
                                                     class_mode='categorical')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('dataset2/test', target_size=(64, 64), batch_size=32,
                                                class_mode='categorical')

    return training_set, test_set


def train_model(created_model, training_set, test_set):
    created_model.fit(x=training_set,
                      validation_data=test_set,
                      epochs=30,
                      verbose=1)
    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(test_loss, test_acc)
    return created_model


def save_model(trained_model_):
    model_json = trained_model_.to_json()
    with open("../model-bw.json", "w") as json_file:
        json_file.write(model_json)
    trained_model_.save_weights('model-bw.h5')


model = build_model()
train, test = preprocess_data()
trained_model = train_model(model, train, test)
save_model(trained_model)
