import torch.nn as nn 
import torch 
architecture = [
    # kernel , out_channels , stride , padding 
    (7 , 64 , 2 ,3 ) , 
    "M",
    (3, 192 , 1 , 1) , 
    "M", 
    (1, 128 , 1, 0) , 
    (3, 256 , 1, 1) , 
    (1, 128 , 1, 0) , 
    (3, 256 , 1 ,1) , 
    "M",
    [(1 ,256 ,1 ,0 ),(3 ,512, 1, 1),4],
    (1 ,512 ,1 , 0),
    (3 ,1024 , 1 ,1),
    "M",
    [(1 ,512,1,0),(3 ,512, 1 ,1 ),2],
    (3 ,1024 ,1 ,1 ),
    (3 ,1024 ,2 ,1),
    (3 ,1024 ,1 ,1),
    (3 ,1024 ,1 ,1),
]
class CNNBlock(nn.Module):
    def __init__(self , in_channels,out_channels, **kwargs):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels , out_channels=out_channels,bias=False , **kwargs) 
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1) 

    def forward(self ,x ):
        return self.leakyrelu(self.batchnorm(self.conv(x))) 
    

class Yolov1(nn.Module):
    def __init__(self,in_channels= 3, **kwargs):
        super().__init__() 
        self.architecture = architecture 
        self.in_channels = in_channels 
        self.darknet = self.create_conv_layers(self.architecture) 
        self.fc = self.create_fc(**kwargs) 
    def forward(self ,x ):
        x = self.darknet(x)
        #print(x.shape)
        #print(torch.flatten(x , start_dim=1).shape)
        return self.fc(torch.flatten(x,start_dim=1))
    def create_conv_layers(self,architecture):
        layers =[] 
        in_channels = self.in_channels
        for x in architecture: 
            if type(x)==tuple:
                layers.append(CNNBlock(in_channels=in_channels,out_channels=x[1],kernel_size =x[0] , stride= x[2] , padding =x[3]))
                in_channels  = x[1] 
            elif type(x)==str:
                layers.append(nn.MaxPool2d(kernel_size=2 , stride=2)) 
            elif type(x)==list:
                for _ in range(x[2]): 
                    layers.append(CNNBlock(in_channels=in_channels,out_channels=x[0][1],kernel_size = x[0][0], stride =x[0][2] , padding =x[0][3]))
                    layers.append(CNNBlock(in_channels=x[0][1],out_channels=x[1][1],kernel_size = x[1][0], stride =x[1][2] , padding =x[1][3]))
                    in_channels  = x[1][1] 
        return nn.Sequential(*layers) 

    def create_fc(self,split_size , num_boxes , num_classes):
        S,B,C = split_size , num_boxes , num_classes 
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 496),
            nn.Dropout(0.0), 
            nn.LeakyReLU(0.1) , 
            nn.Linear(496 , S  * S * (C + B*5)) # have to reshape later to (S ,S , 30) 
        )
    
# model = Yolov1(in_channels=3,split_size = 7 , num_boxes = 2 , num_classes = 20 ) 
# x = torch.randn(1 , 3, 448, 448 ) 
# print(x.shape) 