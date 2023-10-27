import torch
import logging
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from utils.helper_functions import Gaussian

class WeightsMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden1_size, hidden2_size, output_size, softmax, relu, X, Y, pol_ratios):
        super(WeightsMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.use_relu = relu
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden1_size)
        self.fc3 = torch.nn.Linear(self.hidden1_size, self.hidden2_size)
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden2_size, self.output_size)
        self.softmax = softmax
        self.softmax_activation = torch.nn.Softmax(-1)
        self.X_mean, self.X_std = X.mean().item(), X.std().item()
        self.Y_mean, self.Y_std = Y.mean().item(), Y.std().item()
        self.pol_ratios_mean, self.pol_ratios_std  = pol_ratios.mean(), pol_ratios.std()

    def forward(self, x_norm):
        # the input must be normalised
        hidden = self.relu(self.fc1(x_norm))
        hidden = self.relu(self.fc2(hidden))
        hidden = self.relu(self.fc3(hidden))
        output = self.fc4(hidden)
        if self.softmax:
            output = self.softmax_activation(output)
        elif self.use_relu:
            output = self.relu(output)
        return output

    def get_weights(self, x, y):
        # The input must be un-normalised
        # Performs input normalisation and denormalises the output
        x_norm = (x - self.X_mean)/self.X_std
        y_norm = (y - self.Y_mean)/self.Y_std
        return self.forward(torch.cat([x_norm, y_norm], -1)).reshape(-1)*self.pol_ratios_std + self.pol_ratios_mean


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, softmax):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = softmax
        self.softmax_activation = torch.nn.Softmax(-1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        if self.softmax:
            output = self.softmax_activation(output)
        return output


class PolicyMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, softmax):
        super(PolicyMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = softmax
        self.softmax_activation = torch.nn.Softmax(-1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        if self.softmax:
            output = self.softmax_activation(output)
        return output

    def sample(self, x):
        actions_probs = self.forward(x)
        m = Categorical(probs=actions_probs)
        actions = F.one_hot(m.sample())
        return actions

    def density_value(self, x, a):
        behav_mean = self.forward(x.reshape(-1, 1))[:, 0]
        behav_log_var = self.forward(x.reshape(-1, 1))[:, 1]
        behav_gaussian = Gaussian(behav_mean, torch.sqrt(torch.exp(behav_log_var)))
        return behav_gaussian(a).reshape(-1, 1)
