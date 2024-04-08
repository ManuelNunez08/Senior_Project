import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust the input size accordingly
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, spec_data):
        x = self.pool(F.relu(self.conv1(spec_data)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.adaptive_pool(x)

        # Flatten the output for the dense layers
        x = x.view(-1, 64 * 5 * 5)

        # Pass the flattened output through dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = AudioCNN()
model.load_state_dict(torch.load('model_path.pth'))

with torch.no_grad():
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    labels = torch.argmax(labels, dim=1)
