import torch
import os
def load_checkpoint(Path,args):
    # expID = input("Please input the last experiment ID:")
    checkpoint = torch.load(f"{Path}/checkpoint_{args.expid}.pth")
    return checkpoint

def save_checkpoint(expID, model, optmz, epoch, loss, Path):
    os.makedirs(Path,exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss
    }

    torch.save(checkpoint, f"{Path}/checkpoint_{expID}.pth")

