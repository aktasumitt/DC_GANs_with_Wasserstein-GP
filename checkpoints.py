import torch

def Save_Checkpoints(optim_gen,optim_disc,model_gen,model_disc,epoch:int,save_path:str):
    callbacks={"Epoch":epoch,
               "Optim_Gen_State":optim_gen.state_dict(),
               "Optim_Disc_State":optim_disc.state_dict(),
               "Model_Gen_State":model_gen.state_dict(),
               "Model_Disc_State":model_disc.state_dict()}
    
    torch.save(callbacks,f=save_path)
    
    print("Checkpoints are saved...")




def Load_Checkpoints(checkpoint,model_gen,model_disc,optim_gen,optim_disc):
    
    model_gen.load_state_dict(checkpoint["Model_Gen_State"])
    model_disc.load_state_dict(checkpoint["Model_Disc_State"])
    optim_disc.load_state_dict(checkpoint["Optim_Disc_State"])
    optim_gen.load_state_dict(checkpoint["Optim_Gen_State"])
    
    
    return checkpoint["Epoch"]
