import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN,self).__init__()
        #build convolutional layer
        self.conv1 = self._make_block(3, 16, 3)
        self.conv2 = self._make_block(16, 32, 3)
        self.conv3 = self._make_block(32, 64, 3)
        self.conv4 = self._make_block(64, 128, 3)
        self.conv5 = self._make_block(128, 256, 3)
        #build fully connected layer
        self.fc1 = nn.Sequential(nn.Linear(in_features=12544,out_features=2048),
                                 nn.Dropout(0.5),
                                 nn.LeakyReLU(0.01))
        self.fc2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                 nn.Dropout(0.5),
                                 nn.LeakyReLU(0.01))
        self.fc3 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                 nn.Dropout(0.5),
                                 nn.LeakyReLU(0.01))
        self.fc4 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes))

    def _make_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels,kernel_size=kernel_size,padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding = "same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

    def forward(self, x):
        #built forward
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x= self.conv5(x)

        #flatten
        # print(x.shape)
        x= x.view(x.shape[0],-1)
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
        x= self.fc4(x)
        return x

if __name__ == '__main__':
    model = SimpleCNN()
    # random_in = torch.rand(8,3,224,224)
    # out = model.forward(random_in)
    # print(out.shape)
    print(model)
    pass
