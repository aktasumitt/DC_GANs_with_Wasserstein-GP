import torch
import matplotlib.pyplot as plt


def Test_Visualize(Model_gen,devices,batch_Size:int=128,fdim:int=100):
    
    noise_gen=torch.randn((batch_Size,fdim,1,1)).to(devices)
    out_generated=Model_gen(noise_gen)
    
    out_generated=out_generated.cpu().detach()
    out_generated[:50]

    for i in range(0,50):
        plt.subplot(5,10,i+1)
        plt.imshow(torch.transpose(torch.transpose(out_generated[i],0,2),0,1))
    
    plt.show()