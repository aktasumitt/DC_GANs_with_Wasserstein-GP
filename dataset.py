from torchvision import transforms,datasets
from torch.utils.data import DataLoader,Dataset



class DATASET(Dataset):
    def __init__(self,img_size):
        super(DATASET,self).__init__()
        
        transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),
                                    transforms.Resize((img_size,img_size))
                                    ])
                        
        self.data=datasets.CIFAR10(root="CIFAR",
                                    train=True,
                                    download=True,
                                    transform=transform)
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        return self.data[index]



def Create_Dataloader(dataset,batch_size):
    
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=True)