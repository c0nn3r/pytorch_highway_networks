import torch
import torch.nn as nn


class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.elu()):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data[:] = gate_bias

        self.normal_layer = nn.Linear(input_size, input_size)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = nn.functional.sigmoid(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)


class HighwayCNN(nn.Module):

    # TODO

    def __init__(self,
                 input_size,
                 gate_bias=-1,
                 activation_function=nn.ReLU()):

        super(HighwayCNN, self).__init__()

        self.softmax = nn.Softmax()
        self.activation_function = activation_function

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data[:] = gate_bias

        self.normal_layer = nn.Linear(input_size, input_size)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.softmax(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
