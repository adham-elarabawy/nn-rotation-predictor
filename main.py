import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data import Data
#from resnet import ResNet
from rotnet import RotNet
import time
import shutil
import yaml
import argparse

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_type', type=str, required=True, description='Actually model type, rot vs. cifar_resnet')

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, class_target, rot_target) in enumerate(train_loader):
        total_loss = 0

        if args.model_type == 'rot':
            target = rot_target
        else:
            target = class_target
        ## Actual training process:
        
        # reset gradients
        optimizer.zero_grad()
        # forward pass
        predicted_batch = model(input)
        # compute loss
        loss = criterion(predicted_batch, target)
        # compute gradients
        loss.backward()
        # update weights/biases
        optimizer.step()
        
        # update total loss
        total_loss += loss

    return total_loss

def validate(val_loader, model, criterion):
    model.eval()
    for i, (input, class_target, rot_target) in enumerate(val_loader):
        total_loss = 0

        if args.model_type == 'rot':
            target = rot_target
        else:
            target = class_target

        # forward pass
        predicted_batch = model(input)
        # compute loss
        loss = criterion(predicted_batch, target)
        
        # update total loss
        total_loss += loss
    return total_loss

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    n_epochs = config["num_epochs"]
    model = RotNet(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])

    dataset = Data(args.data_dir)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * .75), int(len(dataset) * .25)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config["batch_size"], shuffle = True)

    for epoch in range(n_epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        print(train_loss)
                            





if __name__ == "__main__":
    main()
