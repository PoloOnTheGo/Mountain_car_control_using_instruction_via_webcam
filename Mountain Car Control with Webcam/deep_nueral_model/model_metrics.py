import json
import tensorflow as tf

from tensorflow.python.keras.layers import Conv3D, Conv2D, Dropout, MaxPool2D, Flatten, Dense

from deep_nueral_model.conv_model_metrics import ConvModelMetrics
from deep_nueral_model.dense_layer_metrics import DenseLayerMetrics


class ModelMetrics:
    def __init__(self, conv_model_metrics: ConvModelMetrics, dropout, pooling, dense_layer_metrics: DenseLayerMetrics,
                 opt, batch_size, epochs):
        self.conv_model_metrics: ConvModelMetrics = conv_model_metrics
        self.dropout = dropout
        self.pooling = pooling
        self.dense_layer_metrics: DenseLayerMetrics = dense_layer_metrics
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = epochs

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

