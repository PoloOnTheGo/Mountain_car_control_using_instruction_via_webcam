import json


class ConvModelMetrics:
    def __init__(self, model_name, act_func, kernel_size, no_of_neurons):
        self.model_name = model_name
        self.act_func = act_func
        self.kernel_size = kernel_size
        self.no_of_neurons = no_of_neurons
        
    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
