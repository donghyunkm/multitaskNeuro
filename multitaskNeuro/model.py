import torch
import torch.nn as nn
import torch.nn.functional as F

# Models adapted from (Peng et al., 2021)


class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i == 0:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=1),
                )

            elif i < n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1)
                )
            elif i == n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=3, padding=1)
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, dropout=False, kernel_size=1, padding=0),
                )
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module("conv_%d" % i, nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        if dropout:
            layer.add_module("dropout", nn.Dropout(0.5))
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        return x


class SFCNMTL(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], dropout=True):
        super(SFCNMTL, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i == 0:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=1),
                )

            elif i < n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1)
                )
            elif i == n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=3, padding=1)
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, dropout=False, kernel_size=1, padding=0),
                )
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        self.final_0 = nn.Conv3d(in_channel, 40, padding=0, kernel_size=1)
        self.final_1 = nn.Conv3d(in_channel, 2, padding=0, kernel_size=1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        if dropout:
            layer.add_module("dropout", nn.Dropout(0.5))
        return layer

    def forward(self, x, label):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)

        if label == "age":
            x = self.final_0(x)
        elif label == "sex":
            x = self.final_1(x)

        x = F.log_softmax(x, dim=1)
        return x


class SFCNAUX(nn.Module):
    def __init__(self, aux_size=5, channel_number=[32, 64, 128, 256, 256, 64]):
        super(SFCNAUX, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i == 0:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=1),
                )

            elif i < n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1)
                )
            elif i == n_layer - 2:
                self.feature_extractor.add_module(
                    "conv_%d" % i, self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=3, padding=1)
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, dropout=False, kernel_size=1, padding=0),
                )
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        self.final_0 = nn.Conv3d(in_channel, 25, padding=0, kernel_size=1)
        self.final_1 = nn.Conv3d(in_channel, aux_size, padding=0, kernel_size=1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, dropout=False, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        if dropout:
            layer.add_module("dropout", nn.Dropout(0.5))
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)

        x_age = F.log_softmax(self.final_0(x), dim=1)
        x_aux = F.log_softmax(self.final_1(x), dim=1)
        return x_age, x_aux

    def embed(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        last_embedding = self.final_0(x)
        return x, last_embedding.squeeze(), F.log_softmax(last_embedding, dim=1)
