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
    def __init__(self, in_nch=18, hlnum=3, hlweights=[], lstm_nch=64, out_nch=3):
        super(ComboNN_video, self).__init__()

        self.in_nch = in_nch
        self.out_nch = out_nch
        self.nlayers = 1
        self.nhid = lstm_nch

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
                        in_features=in_features, out_features=self.nhid, bias=True
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
            input_size=self.nhid,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            batch_first=True,
        )

        self.post_lstm = nn.Sequential(
            nn.Linear(in_features=self.nhid, out_features=self.nhid, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.nhid, out_features=self.out_nch, bias=True),
        )

    def forward(self, inpt, mask, X_lengths):

        bb, ll, ft = inpt.shape
        x = inpt.view(bb * ll, ft)

        ### Combo MLP step

        x = self.layers(x)

        lstm_input = x.view(bb, ll, self.nhid)

        zeros = torch.zeros(lstm_input.shape).to(inpt.device)

        lstm_input = torch.where(
            mask.unsqueeze(2).repeat(1, 1, self.nhid), lstm_input, zeros
        )

        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, X_lengths, batch_first=True, enforce_sorted=False
        )

        _, hidden = self.lstm(lstm_input)

        # LSTM
        out = self.post_lstm(hidden[0][0, :, :])

        return torch.sigmoid(out)


class ComboNN_video_pre(nn.Module):
    def __init__(self, in_nch=18, hlnum=3, hlweights=[], lstm_nch=64, out_nch=3):
        super(ComboNN_video_pre, self).__init__()

        self.in_nch = in_nch
        self.out_nch = out_nch
        self.nlayers = 1
        self.nhid = lstm_nch

        self.lstm = nn.LSTM(
            input_size=self.in_nch // 6,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            batch_first=True,
        )

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

        # self.post_lstm = nn.Sequential(
        #     nn.Linear(in_features=self.in_nch, out_features=64, bias=True),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(in_features=64, out_features=self.in_nch, bias=True),
        #     nn.ReLU(inplace=False),
        # )

    def forward(self, inpt, mask, X_lengths):

        bb, ll, _ = inpt.shape
        x = inpt.view(bb, ll, 6, 3)

        # LSTM
        hds = []
        for ii in range(0, 6):
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                x[:, :, ii, :], X_lengths, batch_first=True, enforce_sorted=False
            )

            _, hidden = self.lstm(lstm_input)

            hds.append(hidden[0][0, :, :])

        est = torch.cat(hds, 1)

        # est = self.post_lstm(est)

        ### Combo MLP step

        out = self.layers(est)

        return torch.sigmoid(out)
