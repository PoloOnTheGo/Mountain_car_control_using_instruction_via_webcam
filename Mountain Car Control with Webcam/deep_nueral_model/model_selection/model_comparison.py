import time
import matplotlib.pyplot as plt
import numpy as np
from deep_nueral_model.conv_model_metrics import ConvModelMetrics
from deep_nueral_model.dense_layer_metrics import DenseLayerMetrics
from deep_nueral_model.model_metrics import ModelMetrics
from deep_nueral_model import model as md


def model_evaluation(dataset_name: str, model_no, model_metrics_obj):
    train, test, val = md.preprocess_data_for_model(dataset_name, model_metrics_obj.batch_size)
    print(model_metrics_obj.str_format())
    cnn = md.model_creation(model_metrics_obj, (64, 64, 3))
    start_time = time.time()
    history, trained_cnn = md.train_model(cnn, model_metrics_obj, train, test)
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

    label_list = val[0][1]
    output = trained_cnn.predict(val)
    i = 0
    for x in range(75):
        predicted_cat = list(output[x]).index(max(output[x]))
        label = label_list[x].argmax()
        if predicted_cat != label:
            i += 1
    print('# Wrong prediction :' + str(i))


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
dataset = '../webcam_gesture_recognition/dataset1'
different_models(dataset)

# -----------------------------------------------End of Dataset1-------------------------------------------

# -------------------------------------------------- Dataset2 ---------------------------------------------
dataset = '../webcam_gesture_recognition/dataset2'
different_models(dataset)
# -----------------------------------------------End of Dataset2-------------------------------------------
