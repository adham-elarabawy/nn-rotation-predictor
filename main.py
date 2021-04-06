import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data import Data
import numpy as np
#from resnet import ResNet
from rotnet import RotNet
import time
import shutil
import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_type', type=str, required=True, help='Actually model type, rot vs. cifar_resnet')
parser.add_argument('--model_file', type=str, help='The filename of the model to be loaded')

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
dev = torch.device(dev)


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    total_loss = 0
    for i, (input, class_target, rot_target) in enumerate(train_loader):

        input = input.to(dev)

        if args.model_type == 'rot':
            target = rot_target
        else:
            target = class_target
        target = target.to(dev)
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

        # for cyclic lr, must step after every batch
        scheduler.step()

        # update total loss
        print("LOSS FOR BATCH {}: {}".format(i, loss), end = ' ' * 15 + '\r')
        total_loss += loss

    return total_loss

def validate(val_loader, model, criterion):
    model.eval()
    avg_precision = 0
    total_loss = 0
    for i, (input, class_target, rot_target) in enumerate(val_loader):

        if args.model_type == 'rot':
            target = rot_target
        else:
            target = class_target
        target = target.to(dev)

        # forward pass
        predicted_batch = model(input.to(dev))

        # compute loss
        #loss = criterion(predicted_batch, target)

        predicted_label = np.argmax(predicted_batch.to('cpu').detach().data.numpy().reshape(-1))

        # print(f'Predicted: {predicted_label}, Actual: {target.numpy()[0]}')
        if (target.to('cpu').detach().numpy()[0] == predicted_label):
            avg_precision += 1

        # update total loss
        #total_loss += loss
        if i % 100 == 0:
            print("Running average: " + str(avg_precision / (i + 1)) + " Index: " + str(i), end= ' '* 15 + '\r')

    avg_precision /= len(val_loader)
    print("Average Precision: ", avg_precision)

    return avg_precision

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

# loads up to 3rd block of resnet impl
def load_cifar_from_rot(model):
    state_dict = torch.load(args.model_file, map_location=torch.device('cpu'))
    rem_list = ['layer4.0.weight', 'layer4.0.bias', 'layer4.1.weight', 'layer4.1.bias', 'fc.weight', 'fc.bias']
    for rem in rem_list:
        state_dict.pop(rem)
    model.load_state_dict(state_dict, strict = False)

    for name, param in model.named_parameters():
        if name not in rem_list:
            param.required_grad = False
            print("froze {}".format(name))

# loads up to 3rd block of resnet impl
def load_model(model):
    state_dict = torch.load(args.model_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict = False)

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.model_type == 'rot':
        num_classes = 4
    else:
        num_classes = 10

    n_epochs = config["num_epochs"]
    model = RotNet(num_classes = num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum = config["momentum"])

    # loads frozen model parameters by mutating model up to and including nth block
    if args.model_file is not None:
        if args.train and args.model_type == 'cifar_from_rot':
            load_cifar_from_rot(model)
        else:
            load_model(model)

    dataset = Data(args.data_dir)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * .75), int(len(dataset) * .01), int(len(dataset) * .24)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config["batch_size"], num_workers = 4, pin_memory = True, shuffle = False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, num_workers = 4, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, num_workers = 4, shuffle = False)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, config["low_lr"], config["high_lr"], step_size_up = len(dataset) // 2, mode = 'exp_range', gamma = config["gamma"])

    if args.train:
        train_losses = []
        for epoch in range(n_epochs):
            train_loss = train(train_loader, model, criterion, optimizer, epoch, scheduler)
            val_loss = validate(val_loader, model, criterion)

            train_losses.append(train_loss.item())
            print("TOTAL LOSS FOR EPOCH {}: {}".format(epoch, train_loss.item()))
            print("VAL ACCURACY: {}".format(val_loss))
            
            if epoch % 5 == 0:
                save_checkpoint(model.state_dict(), False, filename = 'epoch_{}_model_{}.pth.tar'.format(epoch, args.model_type))
        # TODO: remove, per Adham's request
        print("LOSS_LIST: {}".format(train_losses))
        np.savetxt("train_losses.csv", train_losses, delimiter =", ", fmt ='% s')
    test_loss = validate(test_loader, model, criterion)
    print("Test accuracy: ", test_loss)




if __name__ == "__main__":
    main()
