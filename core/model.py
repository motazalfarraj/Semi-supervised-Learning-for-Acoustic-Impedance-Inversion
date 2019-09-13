
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d



class inverse_model(nn.Module):
    def __init__(self, vertical_scale):
        super(inverse_model, self).__init__()
        self.vertical_scale = vertical_scale
        self.activation = nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=2,
                                           dilation=1),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=6,
                                           dilation=3),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=12,
                                           dilation=6),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation)

        self.gru = nn.GRU(input_size=1,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=8,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=8,
                                                   out_channels=8,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation)

        self.gru_out = nn.GRU(input_size=8,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=16, out_features=1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out
        x = self.up(x)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1,-2)
        return x



class forward_model(nn.Module):
    def __init__(self,num_channels):
        super(forward_model, self).__init__()
        self.activation = nn.ReLU()
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=num_channels, out_channels=4, kernel_size=9, padding=4),
                                 self.activation,
                                 nn.Conv1d(in_channels=4, out_channels=4,kernel_size=7, padding=3),
                                 self.activation,
                                 nn.Conv1d(in_channels=4, out_channels=1,kernel_size=3, padding=1))


        self.wavelet = nn.Conv1d(in_channels=num_channels,
                             out_channels=num_channels,
                             stride=4,
                             kernel_size=51,
                             padding=25,
                             groups=num_channels)


    def forward(self, x):
        x = self.cnn(x)
        x = self.wavelet(x)
        return x

#%%

# class InverseModel(nn.Module):
#     def __init__(self, num_angles):
#         super(InverseModel, self).__init__()
#         1 = num_angles
#         self.activation = nn.ReLU()
#
#         self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1,
#                                            out_channels=8,
#                                            kernel_size=5,
#                                            padding=2,
#                                            dilation=1),
#                                   nn.GroupNorm(num_groups=1,
#                                                num_channels=8))
#
#         self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1,
#                                            out_channels=8,
#                                            kernel_size=5,
#                                            padding=6,
#                                            dilation=3),
#                                   nn.GroupNorm(num_groups=1,
#                                                num_channels=8))
#
#         self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1,
#                                            out_channels=8,
#                                            kernel_size=5,
#                                            padding=12,
#                                            dilation=6),
#                                   nn.GroupNorm(num_groups=1,
#                                                num_channels=8))
#
#         self.cnn = nn.Sequential(self.activation,
#                                  nn.Conv1d(in_channels=8,
#                                            out_channels=12,
#                                            kernel_size=3,
#                                            padding=1),
#                                  nn.GroupNorm(num_groups=1,
#                                               num_channels=12),
#                                  self.activation,
#
#                                  nn.Conv1d(in_channels=12,
#                                            out_channels=16,
#                                            kernel_size=3,
#                                            padding=1),
#                                  nn.GroupNorm(num_groups=1,
#                                               num_channels=16),
#                                  self.activation)
#
#         self.gru = nn.GRU(input_size=1,
#                           hidden_size=8,
#                           num_layers=2,
#                           batch_first=True,
#                           bidirectional=True)
#
#
#
#         self.out = nn.Linear(in_features=16, out_features=1)
#
#
#     def forward(self, x):
#         cnn_out1 = self.cnn1(x)
#         cnn_out2 = self.cnn2(x)
#         cnn_out3 = self.cnn3(x)
#         cnn_out = self.cnn(cnn_out1+cnn_out2+cnn_out3)
#
#         tmp_x = x.permute(0,2,1).contiguous()
#         rnn_out, _ = self.gru(tmp_x)
#         rnn_out = rnn_out.permute(0,2,1)
#
#         x = rnn_out + cnn_out
#
#         tmp_x = x.transpose(2,1)
#         rnn_out = self.out(tmp_x)
#         x = rnn_out.transpose(2,1)
#
#         return x

#
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()
#
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation)
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation)
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         #self.init_weights()
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
#
# class InverseModel(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(InverseModel, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels,
#                                      out_channels,
#                                      kernel_size,
#                                      stride=1,
#                                      dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size,
#                                      dropout=dropout)]
#
#         self.network = nn.Sequential(*layers)
#
#         self.out = nn.Conv1d(in_channels=num_channels[-1], out_channels=1, kernel_size=1)
#
#     def forward(self, x):
#         return self.out(self.network(x))






#%%
