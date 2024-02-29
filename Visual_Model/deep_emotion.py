# Import necessary libraries from PyTorch for neural network creation
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Deep_Emotion class which inherits from nn.Module, the base class for all neural network modules in PyTorch
class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        Initialize the Deep_Emotion model with its layers and components.
        This includes convolutional layers, pooling layers, batch normalization, and fully connected layers.
        '''
        super(Deep_Emotion,self).__init__() # Initialize the parent class (nn.Module)
        # Define the first set of convolutional layers and pooling layer
        self.conv1 = nn.Conv2d(1,10,3) # 1 input channel, 10 output channels, kernel size 3
        self.conv2 = nn.Conv2d(10,10,3) # 10 input channels (from conv1), 10 output channels, kernel size 3
        self.pool2 = nn.MaxPool2d(2,2) # Max pooling with kernel size 2 and stride 2

        # Define the second set of convolutional layers and pooling layer
        self.conv3 = nn.Conv2d(10,10,3) # Same configuration as before but operating on the output of pool2
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        # Batch normalization for the output of conv4
        self.norm = nn.BatchNorm2d(10) # Normalizes the output channels from conv4

        # Fully connected layers to output the final emotion predictions
        self.fc1 = nn.Linear(810,50) # Flatten conv4 output to 810 features, output 50 features
        self.fc2 = nn.Linear(50,7) # Map 50 features to 7 emotions

        # Spatial transformer network localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Fully connected layers for the spatial transformer network to predict the transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/biases of the transformation parameters to represent no transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        '''
        Spatial Transformer Network (STN) forward function.
        Processes input through the localization network, predicts transformation parameters, and applies the transformation.
        '''
        xs = self.localization(x) # Apply the localization network to the input
        xs = xs.view(-1, 640) # Flatten the output for the fully connected layer
        theta = self.fc_loc(xs) # Predict the transformation parameters
        theta = theta.view(-1, 2, 3) # Reshape theta to the transformation matrix

        # Generate a grid of coordinates in the input image and apply the transformation
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        # Apply the Spatial Transformer Network (STN) to the input.
        # This allows the network to learn to spatially transform the input image
        # in a way that enhances the model's performance on emotion recognition.
        out = self.stn(input)

        # Apply the first convolutional layer followed by a ReLU activation function.
        # This layer extracts low-level features from the transformed input image.
        out = F.relu(self.conv1(out))

        # Apply the second convolutional layer without an activation function.
        # This continues the process of feature extraction.
        out = self.conv2(out)

        # Apply a max pooling operation followed by a ReLU activation function.
        # The pooling reduces the spatial dimensions (width and height) of the output from the previous layer,
        # helping to reduce the computation and control overfitting.
        out = F.relu(self.pool2(out))

        # Repeat the pattern of convolution and pooling layers, including the application of ReLU activation functions.
        # These layers further process the features, extracting more complex patterns from the image.
        out = F.relu(self.conv3(out))
        # Apply batch normalization after the fourth convolutional layer.
        # This normalizes the output of the previous layer, improving the stability and speed of the training process.
        out = self.norm(self.conv4(out))
        # Apply another max pooling operation followed by a ReLU activation function.
        out = F.relu(self.pool4(out))

        # Apply dropout to the output of the last pooling layer.
        # Dropout randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution,
        # which helps prevent overfitting.
        out = F.dropout(out)

        # Flatten the output of the last layer to prepare it for the fully connected layer.
        # The "-1" in the view function indicates that this dimension should be inferred from the size of the input,
        # ensuring that the total size of the tensor does not change.
        out = out.view(-1, 810)

        # Apply the first fully connected (linear) layer followed by a ReLU activation function.
        # This layer begins the process of classifying the extracted features into one of the emotion categories.
        out = F.relu(self.fc1(out))

        # Apply the second fully connected layer.
        # This produces the final output, with each element representing the unnormalized score (logit) for one class.
        out = self.fc2(out)

        # Return the output. At this point, the output contains logits for each class.
        return out