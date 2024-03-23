import torch.nn as nn
    
class Model(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__() 
        self.args = args
        # self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(),
                                   nn.AvgPool2d(3),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ReLU(),
                                   nn.AvgPool2d(3),
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(num_features=512),
                                   nn.ReLU(),
                                   nn.Dropout())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # self.gru = nn.GRU(input_size=4 * 4 * 512, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=4 * 4 * 512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden = None
        
        self.fc = nn.Sequential(nn.Linear(128 * 2, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 2))

    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.gru(x)
        # if self.training:
        #     x, _ = self.gru(x)
        # else:
        #     x, self.hidden = self.gru(x, self.hidden)
        # x = x.contiguous().view(batch_size, seq_len, -1)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x