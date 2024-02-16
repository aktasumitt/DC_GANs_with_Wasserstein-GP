import torch.nn as nn


class Generator(nn.Module):
    
    def __init__(self,f_dim,channel_img):
        super(Generator,self).__init__()
 
        self.conv_g=nn.Sequential(nn.ConvTranspose2d(in_channels=f_dim,out_channels=1024,kernel_size=4), #(f_dim=100,1,1)-->(1024,4,4)
                      nn.BatchNorm2d(num_features=1024),
                      nn.ReLU(), 
                      
                      nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1),           # (512,8,8) 
                      nn.BatchNorm2d(num_features=512), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1),            # (256,16,16) 
                      nn.BatchNorm2d(num_features=256), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1),            # (128,32,32)
                      nn.BatchNorm2d(num_features=128), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=128,out_channels=channel_img,kernel_size=4,stride=2,padding=1),    # (3,64,64)
                      nn.Tanh())    
        
    def forward(self,data):
        
        # data => random_noise (batch_size,100,1,1) 
        out=self.conv_g(data) 
        return out
    
    
    
class Discriminator(nn.Module):
    def __init__(self,channel_img) -> None:
        super(Discriminator,self).__init__()
        
        self.conv_d=nn.Sequential(nn.Conv2d(in_channels=channel_img,out_channels=128,kernel_size=4,stride=2,padding=1),    # (3,64,64)--->(128,32,32)
                             nn.InstanceNorm2d(128,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),             # (256,16,16)
                             nn.InstanceNorm2d(256,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),              # (512,8,8)
                             nn.InstanceNorm2d(512,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=4,stride=2,padding=1),             # (1024,4,4)
                             nn.InstanceNorm2d(1024,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=4),                                   # (1,1,1)
                            )        
    
    def forward(self,data):
            
        # data => Generated_image (batch_size,3,64,64)   
        out=self.conv_d(data)
        return out
        
        
        