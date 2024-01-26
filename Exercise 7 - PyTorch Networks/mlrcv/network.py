import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, BatchNorm1d, ReLU, Flatten

class Net(torch.nn.Module):
    
    def __init__(self):
        """
        This function initializes the Net class and defines the network architecture:

        Args:

        Returns:
        """

        super(Net,self).__init__()
        
        self.conv_net = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  # output size = 32*32
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),   # output size = 16*16
            
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),  # output size = 16*16
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # output size = 8*8
            
            Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),  # output size = 8*8
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # output size = 4*4
            
            Flatten(start_dim=1),
            Linear(in_features=4*4*128, out_features=1500),
            ReLU(),            
            Linear(in_features=1500, out_features=500),
            ReLU(),            
            Linear(in_features=500, out_features=10),
        )  
        
        


        
# old implematation which gives around 58% accuracy         
#     def __init__(self):
#         """
#         This function initializes the Net class and defines the network architecture:

#         Args:

#         Returns:
#         """

#         super(Net,self).__init__()

#         self.conv_net = torch.nn.Sequential(
#             Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
#             ReLU(),
#             Flatten(start_dim=1),
#             Linear(in_features=6144, out_features=2000),
#             ReLU(),
#             Linear(in_features=2000, out_features=10),
#         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function receives the input x and pass it over the network, returning the model outputs:

        Args:
            - x (tensor): input data

        Returns:
            - out (tensor): network output given the input x
        """
        out = self.conv_net(x)
        
        return out