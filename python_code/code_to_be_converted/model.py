import torch
from PIL import Image
from torch import nn, optim
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

class Classifier(nn.Module):  # classifier inherits  nn.Module
    def __init__(self, numChannels=1, classes=1):  # Constructor
        super().__init__()
        # Model Architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=2680, out_features=32),
            nn.ReLU(),  # Activation function
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),  # Activation function
            nn.Linear(in_features=8, out_features=1),
            nn.Sigmoid()  # Activation function
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim=2)   # transform X shape from [batch, 1, H, W] -> [batch, 1, H*W]
        x = self.model(x)
        return x


def train_one_epoch(epoch_index: int, tb_writer: SummaryWriter, model: nn.Module, training_loader: DataLoader,
                    optimizer: optim):
    '''
    Run one epoch of training
    :param epoch_index:
    :param tb_writer: tensorboard logger
    :param model: the model to be trained
    :param training_loader: the data loader
    :param optimizer: the optimizer to update the weights of the model
    :return: loss
    '''
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        labels = labels.to(torch.double)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = torch.squeeze(model(inputs)).to(torch.double)

        # Compute the loss and its gradients
        loss = nn.BCELoss(reduction='none')(outputs, labels).mean()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            tb_x = epoch_index * len(training_loader) + i + 1
            # Log training results
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss


def train(epochs, train_dataloader, val_dataloader):
    model = Classifier()
    loss_fn = nn.BCELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/wood_classifier{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 10.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_index=epoch, model=model, training_loader=train_dataloader, optimizer=optimizer,
                                   tb_writer=writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            vlabels = vlabels.to(torch.double)
            voutputs = torch.squeeze(model(vinputs)).to(torch.double)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.mean().item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{:2d}_{:.3f}'.format(epoch_number, avg_vloss)
            torch.save(model.state_dict(), model_path)
        epoch_number += 1


def inference(image_path, model_path, transformation):
    # Load the pretrained model
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Read the image
    im = Image.open(image_path)
    # Apply transformation on image
    im = transform(im)
    im = im[None, :]  # Adds a dummy dimension at the start => [1, 1, H, W]
    cls = model(im).item()
    cls = 1 if cls >= 0.5 else 0
    if cls == 1:
        print(f'image {image_path} is lug')
    else:
        print(f'image {image_path} is wood')

    return cls

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Change the scale of the image from 0-255 to 0-1
        torchvision.transforms.CenterCrop((536, 5)),  # Crop images to the size of 536,5
        torchvision.transforms.Grayscale(),  # get the gray scale
    ])

    # load the datasets
    train_path = "../../data/annotated/train"
    val_path = "../../data/annotated/val"
    # Initialize the dataset objects
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=transform)
    # Create the iterators that will load the data to memory when needed
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Training
    #train(epochs=20, train_dataloader=train_loader, val_dataloader=val_loader)

    # Inference
    inference(image_path='../../data/annotated/test/0/img_0_48.png',
              model_path='model_19_0.003', transformation=transform)
