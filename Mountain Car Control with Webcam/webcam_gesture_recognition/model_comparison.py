import time
import matplotlib.pyplot as plt
from deep_nueral_model.conv_model_metrics import ConvModelMetrics
from deep_nueral_model.dense_layer_metrics import DenseLayerMetrics
from keras.preprocessing.image import ImageDataGenerator
from deep_nueral_model.model import Model
from deep_nueral_model.model_metrics import ModelMetrics


def preprocess_data(dataset, batch_size):
    target_size = (64, 64)
    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(str(dataset + '/train'), target_size=(64, 64), batch_size=32,
                                                     class_mode='categorical')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(str(dataset + '/test'), target_size=target_size, batch_size=batch_size,
                                                class_mode='categorical')

    return target_size, training_set, test_set


def model_evaluation(dataset_name: str, model_no, model_metrics_obj):
    input_shape, train, test = preprocess_data(dataset_name, model_metrics_obj.batch_size)
    print(model_metrics_obj.str_format())
    model = Model(model_metrics_obj, input_shape)
    cnn = model.get_seq_model()
    start_time = time.time()
    history = model.train_model(cnn, train, test)
    elapsed_time = time.time() - start_time

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].legend(['accuracy', 'val_accuracy'])

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].legend(['loss', 'val_loss'])
    fig.suptitle('Model {}, Time takes {} s'.format(model_no, elapsed_time))
    plt.show()


def different_models(dataset):
    '''----------------------1st Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2, no_of_neurons=[64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=False, pooling=None,
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=10, batch_size=32)
    model_evaluation(dataset, '2', model_metrics)

    '''----------------------2nd Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2, no_of_neurons=[64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=10, batch_size=32)
    model_evaluation(dataset, '2', model_metrics)

    '''----------------------3rd Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2, no_of_neurons=[64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=20, batch_size=32)
    model_evaluation(dataset, '3', model_metrics)

    '''----------------------4th Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2, no_of_neurons=[32, 64, 64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=20, batch_size=32)
    model_evaluation(dataset, '4', model_metrics)

    '''----------------------5th Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2,
                                    no_of_neurons=[64, 128, 128])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=20, batch_size=32)
    model_evaluation(dataset, '5', model_metrics)

    '''----------------------6th Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2,
                                    no_of_neurons=[32, 64, 64, 64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=256)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=25, batch_size=32)
    model_evaluation(dataset, '6', model_metrics)

    '''----------------------7th Comparative Model Metrics--------------------'''
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2,
                                    no_of_neurons=[32, 64, 64, 64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=516)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=30, batch_size=32)
    model_evaluation(dataset, '7', model_metrics)


# -------------------------------------------------- Dataset1 ---------------------------------------------
dataset = 'dataset1'
different_models(dataset)

# -----------------------------------------------End of Dataset1-------------------------------------------

# -------------------------------------------------- Dataset2 ---------------------------------------------
dataset = 'dataset2'
different_models(dataset)
# -----------------------------------------------End of Dataset2-------------------------------------------
