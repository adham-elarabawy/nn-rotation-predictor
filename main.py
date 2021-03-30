import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data_pytorch import Data
from resnet import ResNet
from rotnet import RotNet
import time
import shutil
import yaml

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
    	#TODO: use the usual pytorch implementation of training

def validate(val_loader, model, criterion):
	model.eval()
    for i, (input, target) in enumerate(val_loader):
    	#TODO: implement the validation. Remember this is validation and not training
    	#so some things will be different.

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	#best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
	n_epochs = config["num_epochs"]
	model = ResNet(num_classes=4)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=config["learning rate"], momentum=config["momentum"])

    dataset = Data(args.data_dir)
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * .75), int(len(dataset) * .25)])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config["batch_size"], shuffle = True)

	 for epoch in range(n_epochs):
            total_loss = 0
            for batch in train_loader:
                X_batch, y_batch = batch[0].view(-1, 784), batch[1]

                ## Actual training process:
                
                # reset gradients
                optimizer.zero_grad()
                # forward pass
                predicted_batch = model(X_batch)
                # compute loss
                loss = criterion(predicted_batch, y_batch)
                # compute gradients
                loss.backward()
                # update weights/biases
                optimizer.step()
                
                # update total loss
                total_loss += loss

            print("Epoch {0}: {1}".format(epoch, total_loss))
            if epoch%5 == 0 and epoch!= 0:
                test_batch = next(iter(test_dataloader))
                X_test, y_test = test_batch[0].view(-1, 784), test_batch[1]
                predicted = model(X_test)
                test_acc = torch.sum(y_test == torch.argmax(predicted, dim=1), dtype=torch.double) / len(y_test)
                print("\tTest Accuracy {0}".format(test_acc))
                            





if __name__ == "__main__":
    main()
