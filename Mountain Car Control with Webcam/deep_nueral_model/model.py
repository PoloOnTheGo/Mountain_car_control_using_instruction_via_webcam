from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense


def model_creation(model_metrics, input_shape):
    # Initialising the CNN
    model = Sequential()

    # Convolution layer and pooling
    no_of_neurons_conv_layer = model_metrics.conv_model_metrics.no_of_neurons
    for filters in no_of_neurons_conv_layer:
        model.add(Conv2D(filters=filters, kernel_size=model_metrics.conv_model_metrics.kernel_size,
                         activation=model_metrics.conv_model_metrics.act_func, input_shape=input_shape))
        if model_metrics.pooling == 'MaxPool2D':
            model.add(MaxPool2D(pool_size=2, strides=2))

        if model_metrics.dropout:
            model.add(Dropout(0.25))

    # Flattening the layers
    model.add(Flatten())

    # Adding a fully connected layer
    model.add(Dense(units=model_metrics.dense_layer_metrics.no_of_neurons,
                    activation=model_metrics.dense_layer_metrics.act_func))

    # Adding output Layer (softmax for more than 2 classes)
    model.add(Dense(units=3, activation='softmax'))

    # Compiling the CNN
    model.compile(optimizer=model_metrics.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, model_metrics, training_set, test_set):
    history = model.fit(x=training_set,
                        validation_data=test_set,
                        epochs=model_metrics.epochs,
                        verbose=1)
    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(test_loss, test_acc)
    return history, model


def save_model(trained_model_):
    model_json = trained_model_.to_json()
    with open("../model-bw.json", "w") as json_file:
        json_file.write(model_json)
    trained_model_.save_weights('../model-bw.h5')


def preprocess_data_for_model(dataset_name, batch_size):
    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(dataset_name + '/train', target_size=(64, 64),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(dataset_name + '/test', target_size=(64, 64), batch_size=batch_size,
                                                class_mode='categorical')

    # Preprocessing the Test set
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_set = validation_datagen.flow_from_directory(dataset_name + '/validation', target_size=(64, 64),
                                                            batch_size=75, class_mode='categorical')

    return training_set, test_set, validation_set
