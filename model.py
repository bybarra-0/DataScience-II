import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import SEED, NUM_CLASSES, INPUT_SHAPE, RESULTS_PATH, PROJECT_ROOT, CLASS_NAMES, seed_everything
from data import get_class_weights
from preprocess import get_deep_loader
from metrics import compute_metrics

def log_deep_result(model_name: str, config_desc: str, metrics: dict, runtime_s: float) -> dict:
    # Maintain identical reporting schema across modules in central tracking file
    log_row = {
        "model_name": model_name,
        "module": "C",
        "config": config_desc,
        "balanced_accuracy": metrics["balanced_accuracy"], #key here
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "runtime_s": round(runtime_s, 2)
    }

    # Append row directly to shared csv without overwriting prior records
    row_df = pd.DataFrame([log_row])
    row_df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    return log_row

def conv_layout(in_channels, out_channels): #defines the layout of the convolutional layers; passes the variables from CustomCNN to the function
    return nn.Sequential( #module that allows you to create a sequential container of layers
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(), #takes out the negative values and replaces them with 0, allowing the model to learn non-linearities
        nn.MaxPool2d(2)
    )

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # TODO: Define convolutional feature extractor and fully connected classification head
        self.features = nn.Sequential(
            conv_layout(3, 32), #passes onto the conv_layout function the number of input channels (3 for RGB images) and the number of output channels (32)
            conv_layout(32, 64),
            conv_layout(64, 128),
            conv_layout(128, 256),
            nn.AdaptiveAvgPool2d((1,1)), #adaptive is an maxpooling that allows you to specify the output size of the pooling layer
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), #flattens the input tensor to a 1D vector
            nn.Dropout(0.3), #prevents overfitting by randomly setting a fraction of the input units to 0 during training

            nn.Linear(256, 128), #linear layer that maps the flattened feature vector to the number of classe
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(128,num_classes)
        )
        #the classifier takes the output of the feature extractor and flattens it, applies dropout for regularization, 
        #and then passes it through a linear layer to produce the final class logits

        # Input tensor geometry: (batch_size, 3, 224, 224)
        # Expected output shape: (batch_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #defining forward pass
        x = self.features(x) 
        x = self.classifier(x)
        return x # calls functions

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    # TODO: Iterate over mini-batches, execute forward pass, backpropagation, and optimizer step
    model.train() #put into training mode
    Closs = 0
    totalsample = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device) #to allow gpu with the batches

        optimizer.zero_grad() # clear out old gradiate

        outputs = model(inputs) # preforms the forward pass to go through features and classifier
        loss = criterion(outputs, targets) # how wrong it is to true label 

        loss.backward() #backward pass with gradiant and update weights 

        optimizer.step() # updates base on weights

        Closs = (loss.item() * inputs.size(0)) + Closs
        totalsample = targets.size(0) + totalsample
    epoch_loss = Closs / totalsample 
    return epoch_loss #sends out the loss. 

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, return_preds: bool = False):
    model.eval() #testing 

    Closs = 0.0
    totalchecked = 0 #Closs is runing across split which we must track

    #store into meteric 
    n = len(loader.dataset)
    preds = np.zeros(n, dtype=np.int64) #makes the array the size of data to fit in 
    trueL = np.zeros(n, dtype=np.int64) #makes the true lables.

    with torch.no_grad(): #dont track gradients -  saves memory
        for inputs, targets in loader: # load dataset
            inputs = inputs.to(device)
            targets = targets.to(device) #allows gpu/cpu usage
            
            #does forward pass 
            outputs = model(inputs) # raw logitics
            loss = criterion(outputs, targets) # how wrong it is

            batch_size = inputs.size(0) #size per batch

            #evalutation
            Closs = (loss.item() * inputs.size(0)) + Closs

            predectied = torch.argmax(outputs, dim= 1)
            predectied_in = predectied.cpu().numpy()
            targets_in = targets.cpu().numpy() #allows copying

            start = totalchecked #to grab for this batch iteration
            end = start + batch_size

            preds[start:end] = predectied_in
            trueL[start:end] = targets_in

            totalchecked = end #for nect batch
    #perfomance can be passed to metrics.py for us to reflect to 
    validation_loss = Closs / totalchecked
    metrics = compute_metrics(trueL[:totalchecked], preds[:totalchecked]) #iteration
    # Convert raw logits to integer class predictions via torch.argmax(logits, dim=1).cpu().numpy()
    # Evaluate performance by passing integer targets and predictions to compute_metrics(y_true, y_pred)
    if return_preds:
        return validation_loss, metrics, trueL[:totalchecked], preds[:totalchecked]
    return validation_loss, metrics 

def plot_training_curves(train_losses: list, val_losses: list, val_accs: list, output_path: Path = None):
    #send over to figure
    if output_path is None:
        output_path = PROJECT_ROOT / "figures" / "Training_Loop.png" #should make a file where this info is stored
    output_path = Path(output_path) #update path

    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1) #will make it on left
    plt.plot(train_losses, label = "Training Loss")
    plt.plot(val_losses, label = "Validation loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.legend()

    # make right hand side
    plt.subplot(1,2,2) #will make it on right
    plt.plot(val_accs, label = "balanced valdiation accuracy")
    plt.title("Balance validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy") #all the types of accruacy are averaged to one hence balanced
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi = 150)
    plt.close() 
    print("saved") #debug

def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    config_desc: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = None
) -> dict:

    #set the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #just decideing the device
    # for gpu.
    model = model.to(device) #evaulation model

    weights = get_class_weights(as_tensor=True).to(device) #grab weights
    critc = nn.CrossEntropyLoss(weight= weights) #computing loss
    optmize = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optmize, T_max = epochs)

    #checkpoint to be saved
    checkpoint = PROJECT_ROOT / "models" / "deep"
    checkpoint.mkdir(parents=True, exist_ok= True) # for collab
    checkpointPath = checkpoint / (model_name + "_" + config_desc + "_best.pt") #pythoch extension to store weights; we be used for the figure picture

    #tracking the variables/changes
    trainingLoss = []
    validationLoss =[]
    validationaccuracy = []
    best_accuracy_balanced = -1
    best_cycle = 0
    wasted = 15
    noimprove = 0
    start_time = time.time() #loggining in runtime

    #now loop for the epochs
    for epoch in range(epochs): # actually training and grading model.
        trainLoss = train_one_epoch(model, train_loader, critc, optmize, device)
        validloss, valmetric = evaluate(model, val_loader, critc, device)
        avgAccuracy = valmetric["balanced_accuracy"]
        scheduler.step()
        #early stop and to append

        trainingLoss.append(trainLoss)
        validationLoss.append(validloss) #grabbing the acc for the epochs
        validationaccuracy.append(avgAccuracy)

        #for us to see
        print("Epoch", epoch +1, "of", epochs, "training loss - ", round(trainLoss, 3), " validation loss - ", round(validloss,3), " validation accuracy balanced - ", round(avgAccuracy,3))
        print("learning rate now:", optmize.param_groups[0]["lr"])

        if avgAccuracy > best_accuracy_balanced: #saving the best cycles/ weights 
            best_accuracy_balanced = avgAccuracy
            best_cycle = epoch +1 # start at 0
            torch.save(model.state_dict(), checkpointPath) #save it
            noimprove = 0 # we improved counter restart
        else:
            noimprove = 1 + noimprove # closer to stop

        if noimprove >= wasted:
            print("Early detection stop no improvment for ", noimprove, " Cycles") 
            break
    #set save picture to not delete each other
    newpathing = PROJECT_ROOT / "figures" / (model_name+"_"+config_desc+"-"+"overallpathing.png") #models performance in picture
    plot_training_curves(trainingLoss, validationLoss, validationaccuracy, newpathing) #plotting that performance 

    return{ "modelnaming" : model_name, "configuration" : config_desc, "best_validation_accuracy_balanced" : best_accuracy_balanced, "best_epoch" : best_cycle, "checkpoint" : str(checkpointPath), "runTimeing_Seconds" : round(time.time() - start_time, 3),
    } #to send out to data. model name, configure, best testing accuracy, best cycle, turns path to string, and tracks run time (gpu)


#add new def to read from .pt to have the best weights and load it to update for csv.
def evaluation_checkpoint(model: nn.Module, model_name: str, config_desc: str, loader: DataLoader, split_name: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpointPath = PROJECT_ROOT / "models" / "deep" / (model_name + "_" + config_desc + "_best.pt")
    savedweights = torch.load(checkpointPath, map_location= device)
    model.load_state_dict(savedweights)

    weights = get_class_weights(as_tensor=True).to(device)
    critc = nn.CrossEntropyLoss(weight= weights)

    start_time = time.time()
    loss, metrics, y_true, y_pred = evaluate(model, loader, critc, device, return_preds= True)
    runT = time.time() - start_time

    print(split_name, "loss", round(loss,4), "balanced accuracy", round(metrics["balanced_accuracy"],4))

    for c in range(NUM_CLASSES):
        in_class = y_true == c
        caught = y_pred[in_class] == c
        print(" ", CLASS_NAMES[c], "recall", round(caught.mean(),3), "out_of", in_class.sum())

    savepath = PROJECT_ROOT / "figures" / (model_name + "_" + config_desc + "_" + split_name + "_preds.npz") # to make matrix later
    np.savez(savepath, y_true = y_true, y_pred = y_pred)

    if split_name == "test":
        log_deep_result(model_name, config_desc + "_test", metrics, runT)
    return metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module C: Deep Learning Model Training")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

#evalutate best
    parser.add_argument("--eval-split", type = str, default= None, choices= ["val", "test"])

    args = parser.parse_args()

    seed_everything(SEED)
    print("Module C: Initializing PyTorch DataLoaders from Module A...")
    train_loader = get_deep_loader(split_name="train", batch_size=args.batch_size, shuffle=True)
    val_loader = get_deep_loader(split_name="val", batch_size=args.batch_size, shuffle=False)

    config_desc = "adamw_lr" + str(args.lr) + "_BatchSize" + str(args.batch_size) + "_ep" + str(args.epochs) + "_cosine1cycle"
    if args.eval_split is not None:
        eval_loader = get_deep_loader(split_name=args.eval_split, batch_size= args.batch_size, shuffle= False )
        evaluation_checkpoint(CustomCNN(), "Custom_CNN", config_desc, eval_loader, args.eval_split)
        raise SystemExit

    print(f"DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = CustomCNN() #initalize
    summary = train_and_evaluate( 
        model= model,
        model_name= "Custom_CNN",
        config_desc= "adamw_lr" + str(args.lr) + "_BatchSize" + str(args.batch_size) + "_ep" + str(args.epochs) + "_cosine1cycle",
        train_loader= train_loader,
        val_loader= val_loader,
        epochs= args.epochs,
        lr= args.lr,
    )
    print (summary) #for visual reference
    #for our final we would load from models/deep. and write to results
   
