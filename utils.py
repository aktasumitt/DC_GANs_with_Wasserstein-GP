import torch
import torch.nn as nn


def Gradient_penalty(discriminator_model,fake_img,real_img,devices):
    
    BATCH_SIZE,CHANNEL,H,W=real_img.shape
    epsilon=torch.rand(BATCH_SIZE,1,1,1).repeat(1,CHANNEL,H,W).to(devices)
    interpolated_images=real_img*epsilon+fake_img*(1-epsilon)
    
    mixed_scores=discriminator_model(interpolated_images)
    
    gradient=torch.autograd.grad(inputs=interpolated_images,
                                 outputs=mixed_scores,
                                 grad_outputs=torch.ones_like(mixed_scores),
                                 retain_graph=True,
                                 create_graph=True)[0]
    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    
    return gradient_penalty