import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from . import blocks as B


class ComboNN_single(nn.Module):
    def __init__(self, in_nch=18, hlnum=2, hlweights=[], out_nch=3):
        super(ComboNN_single, self).__init__()

        self.in_nch = in_nch

        layers = []
        for i in range(hlnum):

            if i == 0:
                out_features = int(hlweights[i])
                layers.append(
                    nn.Linear(in_features=in_nch, out_features=out_features, bias=True)
                )
                layers.append(nn.ReLU(inplace=False))
                in_features = out_features
            elif i == (hlnum - 1):
                layers.append(
                    nn.Linear(in_features=in_features, out_features=out_nch, bias=True)
                )
            else:
                out_features = int(hlweights[i])
                layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=out_features, bias=True
                    )
                )
                layers.append(nn.ReLU(inplace=False))
                in_features = out_features

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)

        return torch.sigmoid(x)


class ComboNN_video(nn.Module):
    def __init__(self, in_nch=18, hlnum=2, hlweights=[], out_nch=3):
        super(ComboNN_video, self).__init__()

        self.in_nch = in_nch
        self.out_nch = out_nch
        self.nlayers = 1
        self.nhid = out_nch

        layers = []
        for i in range(hlnum):

            if i == 0:
                out_features = int(hlweights[i])
                layers.append(
                    nn.Linear(
                        in_features=self.in_nch, out_features=out_features, bias=True
                    )
                )
                layers.append(nn.ReLU(inplace=False))
                in_features = out_features
            elif i == (hlnum - 1):
                layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=self.out_nch, bias=True
                    )
                )
            else:
                out_features = int(hlweights[i])
                layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=out_features, bias=True
                    )
                )
                layers.append(nn.ReLU(inplace=False))
                in_features = out_features

        self.layers = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=self.out_nch,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            batch_first=True,
        )

        self.post_lstm = nn.Sequential(
            nn.Linear(in_features=self.nhid, out_features=self.nhid, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.nhid, out_features=3, bias=True),
        )

    def forward(self, inpt, mask, X_lengths):

        bb, ll, ft = inpt.shape
        x = inpt.view(bb * ll, ft)

        ### Combo MLP step

        x = self.layers(x)

        lstm_input = x.view(bb, ll, self.out_nch)

        zeros = torch.zeros(lstm_input.shape).to(inpt.device)

        lstm_input = torch.where(
            mask.unsqueeze(2).repeat(1, 1, self.out_nch), lstm_input, zeros
        )

        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, X_lengths, batch_first=True, enforce_sorted=False
        )

        _, hidden = self.lstm(lstm_input)

        # LSTM
        out = self.post_lstm(hidden[0][0, :, :])

        return torch.sigmoid(out)
