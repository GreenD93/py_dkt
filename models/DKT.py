import torch
import torch.nn as nn
from torch.autograd import Variable


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DKT, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        return h0, c0

    def forward(self, x):
        batch_size = x.shape[0]
        h0, c0 = self._init_hidden(batch_size)
        out, hn = self.lstm(x, (h0.detach(), c0.detach()))
        res = self.sig(self.fc(out))

        return res