import torch
import torch.nn as nn

class Inception1D(nn.Module):
    def __init__(self, in_channels, parallel_number, stride = 1):
        super().__init__()
        self.parallel_number = parallel_number

        self.branch1 = nn.Conv1d(in_channels=in_channels, out_channels = 1, kernel_size= 1,
                                     stride= stride, padding= 0)

        self.branch3 = nn.Conv1d(in_channels =in_channels, out_channels = 3, kernel_size= 3,
                                     stride= stride, padding= 1)

        self.branch5 = nn.Conv1d(in_channels =in_channels, out_channels = 3, kernel_size= 5,
                                     stride= stride, padding= 2)

        self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=7,
                                 stride=stride, padding=3)

        self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=9,
                                 stride=stride, padding=4)

        self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=11,
                                 stride=stride, padding= 5)

        self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=13,
                                  stride=stride, padding=6)

        self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=15,
                                  stride=stride, padding=7)
    
    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch3(x)
        f3 = self.branch5(x)
        f4 = self.branch7(x)
        f5 = self.branch9(x)
        f6 = self.branch11(x)
        f7 = self.branch13(x)
        f8 = self.branch15(x)
        if self.parallel_number == 2:
            output = torch.cat((f1, f2), dim=1)
        elif self.parallel_number == 3:
            output = torch.cat((f1, f2, f3), dim=1)
        elif self.parallel_number == 4:
            output = torch.cat((f1, f2, f3, f4), dim=1)
        elif self.parallel_number == 5:
            output = torch.cat((f1, f2, f3, f4, f5), dim=1)
        elif self.parallel_number == 6:
            output = torch.cat((f1, f2, f3, f4, f5, f6), dim=1)
        elif self.parallel_number == 7:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)
        elif self.parallel_number == 8:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
        else:
            output = "error"
        return output
    
class PNNGSModel(nn.Module):
    def __init__(self, hidden_dim, parallel_number=3):
        super().__init__()
        self.conv1 = Inception1D(in_channels=1, parallel_number= parallel_number, stride=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm1d(3 * parallel_number - 2)
        self.conv2 = Inception1D(in_channels= 3 * parallel_number - 2, parallel_number= parallel_number, stride=1)
        self.conv3 = nn.Conv1d(in_channels= 3 * parallel_number - 2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)