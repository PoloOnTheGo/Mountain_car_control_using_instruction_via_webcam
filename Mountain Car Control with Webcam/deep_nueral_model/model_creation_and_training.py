# Importing the libraries
from deep_nueral_model import model as md
from deep_nueral_model.conv_model_metrics import ConvModelMetrics
from deep_nueral_model.dense_layer_metrics import DenseLayerMetrics
from deep_nueral_model.model_metrics import ModelMetrics


def build_model():
    conv_metrics = ConvModelMetrics(model_name='Conv2D', act_func='relu', kernel_size=2,
                                    no_of_neurons=[32, 64, 64, 64])
    dense_metrics = DenseLayerMetrics(act_func='relu', no_of_neurons=516)
    model_metrics = ModelMetrics(conv_model_metrics=conv_metrics, dropout=True, pooling='MaxPool2D',
                                 dense_layer_metrics=dense_metrics, opt='adam', epochs=30, batch_size=32)
    cnn = md.model_creation(model_metrics, (64, 64, 3))

    return cnn, model_metrics


model, metrics = build_model()
train, test, val = md.preprocess_data_for_model('../webcam_gesture_recognition/dataset2', metrics.batch_size)
history, trained_model = md.train_model(model, metrics, train, test)
md.save_model(trained_model)
