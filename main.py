import torch,model,checkpoints,dataset,test,training,config,warnings,utils
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# Control Cuda
devices=("cuda" if torch.cuda.is_available() else "cpu")

#Tensorboard
Tensorboard_Writer=SummaryWriter(config.TENSORBOARD_PATH)

# Create Dataset
cifar_data=dataset.DATASET(img_size=config.IMG_SIZE)
cifar_dataloader=dataset.Create_Dataloader(dataset=cifar_data,batch_size=config.BATCH_SIZE)

# Generator
Generator_Model=model.Generator(f_dim=config.F_DIM,channel_img=config.CHANNEL_IMG)
Generator_Model.to(devices)

# Discriminator
Discriminator_Model=model.Discriminator(channel_img=config.CHANNEL_IMG)
Discriminator_Model.to(devices)

# Optimizers and loss function
Optim_Gen=torch.optim.Adam(params=Generator_Model.parameters(),lr=config.LEARNING_RATE,betas=(0.0,0.9))
Optim_Disc=torch.optim.Adam(params=Discriminator_Model.parameters(),lr=config.LEARNING_RATE,betas=(0.0,0.9))

# Load Checkpoınt if you want
if config.LOAD_CHECKPOINTS==True:
    checkpoint=torch.load(f=config.CALLBACK_PATH)
    resume_epoch=checkpoints.Load_Checkpoints(checkpoint=checkpoint,
                                 model_gen=Generator_Model,
                                 model_disc=Discriminator_Model,
                                 optim_gen=Optim_Gen,
                                 optim_disc=Optim_Disc)
    
    print(f"Training is going to start from {resume_epoch}.epoch... ")

else:
    resume_epoch=0
    print("Training is going to start from scratch...")




# Training
if config.TRAIN==True:
    loss_dict=training.Training(EPOCHS=config.EPOCHS,
                                resume_epoch=resume_epoch,
                                BATCH_SIZE=config.BATCH_SIZE,
                                F_DIM=config.F_DIM,
                                Img_dataloader=cifar_dataloader,
                                gradient_penalty_fn=utils.Gradient_penalty,
                                LAMDA_GP=config.LAMDA_GP,
                                Discriminator_Model=Discriminator_Model,
                                Generator_Model=Generator_Model,
                                Optim_Disc=Optim_Disc,
                                Optim_Gen=Optim_Gen,
                                Save_Checkpoints=checkpoints.Save_Checkpoints,
                                CALLBACK_PATH=config.CALLBACK_PATH,
                                devices=devices,
                                tensorboard=Tensorboard_Writer)


# Visualize Generated Image
if config.TEST==True:
    test.Test_Visualize(Model_gen=Generator_Model,devices=devices,batch_Size=config.BATCH_SIZE,fdim=config.F_DIM)



