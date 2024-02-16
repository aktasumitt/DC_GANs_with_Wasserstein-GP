import torch,tqdm
from torchvision.utils import make_grid

def Training(EPOCHS,resume_epoch,BATCH_SIZE,F_DIM,gradient_penalty_fn,LAMDA_GP,Img_dataloader,Discriminator_Model,Generator_Model,Optim_Gen,Optim_Disc,Save_Checkpoints,CALLBACK_PATH,tensorboard,devices):
    loss_dict={"loss_gen":[],
                "loss_disc":[]}
        
    for epoch in range(resume_epoch,EPOCHS):

        loss_value_gen=0.0
        loss_value_disc=0.0
        progress_bar=tqdm.tqdm(range(160),"Training Process")
        
        for batch,(img,_label_) in enumerate(Img_dataloader):

            real_img=img.to(devices)
            
            # Discriminator Train   
            for _ in range(5):
                
                noise_=torch.randn((BATCH_SIZE,F_DIM,1,1)).to(devices)
                
                Discriminator_Model.zero_grad()
                
                fake_img=Generator_Model(noise_)
                out_fake=Discriminator_Model(fake_img).reshape(-1)
                out_real=Discriminator_Model(real_img).reshape(-1)
                
                penalty=gradient_penalty_fn(discriminator_model=Discriminator_Model,
                                            fake_img=fake_img,
                                            real_img=real_img,
                                            devices=devices)
                    
                loss_disc=(-(torch.mean(out_real)-torch.mean(out_fake))+(penalty*LAMDA_GP))
                loss_disc.backward(retain_graph=True)
                
                Optim_Disc.step()

            # Generator Train
            Generator_Model.zero_grad()
            out_fake=Discriminator_Model(fake_img).reshape(-1)
            loss_gen=-torch.mean(out_fake)
            loss_gen.backward()
            Optim_Gen.step()
            
            
            loss_value_gen+=loss_gen.item()
            loss_value_disc+=loss_disc.item()
            
            progress_bar.update(1)
            
            if batch%40==39:
                
                progress_bar.set_postfix({f"Epoch":epoch+1,"loss_gen":(loss_value_gen/(batch+1)),"loss_disc":loss_value_disc/(batch+1)})
            
            if batch==160:
                break
            
        progress_bar.close()
        # Giving real images and generated images to tensorboard each epochs
        grid_real=make_grid(real_img,nrow=10)
        grid_fake=make_grid(fake_img,nrow=10)    
        
        tensorboard.add_image("Real Images",grid_real,global_step=epoch+1)
        tensorboard.add_image("Generated Image",grid_fake,global_step=epoch+1)
        
        # Giving losses to tensorboard each epochs
        tensorboard.add_scalar("Discriminator loss", loss_value_disc/(batch+1),global_step=epoch+1)
        tensorboard.add_scalar("Generator loss", loss_value_gen/(batch+1),global_step=epoch+1)
        
        # Save Checkpoints each epoch
        Save_Checkpoints(optim_gen=Optim_Gen,
                        optim_disc=Optim_Disc,
                        model_gen=Generator_Model,
                        model_disc=Discriminator_Model,
                        epoch=epoch+1,
                        save_path=CALLBACK_PATH)
                
        loss_dict["loss_disc"].append(loss_value_disc/(batch+1))
        loss_dict["loss_gen"].append(loss_value_gen/(batch+1))
        
    return loss_dict
