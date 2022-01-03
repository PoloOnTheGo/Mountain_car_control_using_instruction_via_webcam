import json


class DenseLayerMetrics:
    def __init__(self, act_func, no_of_neurons):
        self.act_func = act_func
        self.no_of_neurons = no_of_neurons

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
