import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

def load_data(data_directory="./flowers" ):
    train_dir = data_directory + '/train'
    validation_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),                 # randomly rotate the image by up to 30 degrees
        transforms.RandomResizedCrop(224),             # randomly crop the image to size 224x224
        transforms.RandomHorizontalFlip(),             # randomly flip the image horizontally
        transforms.ToTensor(),                         # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) # normalize the image data
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),                        # resize the image to 256x256
        transforms.CenterCrop(224),                    # crop the image to size 224x224 at the center
        transforms.ToTensor(),                         # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) # normalize the image data
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),                        # resize the image to 256x256
        transforms.CenterCrop(224),                    # crop the image to size 224x224 at the center
        transforms.ToTensor(),                         # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) # normalize the image data
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=validation_transforms)
    test_dataset = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # Define the dataloaders for the training, validation, and testing sets
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_data_loader , validation_data_loader, test_data_loader, train_dataset

def create_model(architecure='vgg16', dropout=0.5, hidden_units=100, learning_rate=0.001, device='cpu'):

    architecures = { "vgg16":25088,
                    "alexnet":9216 }
    
    if architecure == 'VGG':
        model = models.vgg16(pretrained=True)
        architecure='vgg16'
    elif architecure == 'ALEXNET':
        model = models.alexnet(pretrained = True)
        architecure='alexnet'
    else:
        print("Choose valid architecture!")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(architecures[architecure], hidden_units),
                                nn.ReLU(),
                                nn.Linear(hidden_units, 80),
                                nn.ReLU(),
                                nn.Linear(80, 70),
                                nn.ReLU(),
                                nn.Linear(70, 102),
                                nn.LogSoftmax(dim=1)
                            )

    model.classifier = classifier
    # Set the loss loss_criterion
    loss_criterion = nn.NLLLoss()
    
    # Set the learning rate and optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, loss_criterion, optimizer

def test_accuracy(model, test_data_loader, device="cpu"):
    accuracy = 0

    for inputs, labels in test_data_loader:
        # Move inputs and labels to device
        if torch.cuda.is_available() and device == 'gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        # Perform a forward pass
        with torch.no_grad():
            outputs = model(inputs)
        # Calculate probabilities and top predictions
        probs = torch.exp(outputs)
        top_probs, top_classes = probs.topk(1, dim=1)
        # Compare with true labels and calculate accuracy
        equals = top_classes == labels.view(*top_classes.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    # Calculate final accuracy and print result
    accuracy = accuracy / len(test_data_loader)
    print(f'Accuracy: {accuracy.item()*100:.2f}%')

def train_network(train_data_loader, validation_data_loader, model, loss_criterion, optimizer, num_epochs=1, print_every=10, device='cpu'):
    steps = 0
    train_losses, valid_losses = [], []
    
    # loop through each epoch
    for epoch in range(num_epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1

            # Move input and label tensors to the device
            if device=='gpu':
                device='cuda:0'
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Set the model to evaluation mode
                model.eval()
                
                # Initialize validation loss and accuracy
                validation_loss = 0
                accuracy = 0
                
                for ii, (inputs, labels) in enumerate(validation_data_loader):
                    optimizer.zero_grad()
                    
                    # Move input and label tensors to the device
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass and calculate loss
                    with torch.no_grad():
                        outputs = model.forward(inputs)
                        validation_loss = loss_criterion(outputs, labels)
                        ps = torch.exp(outputs).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        
                validation_loss = validation_loss / len(validation_data_loader)
                train_loss = running_loss / len(train_data_loader)
                
                train_losses.append(train_loss)
                valid_losses.append(validation_loss)
                
                accuracy = accuracy / len(validation_data_loader)
                
                print("Epoch: {}/{}... ".format(epoch+1, num_epochs),
                    "Training Loss: {:.5f}".format(train_loss),
                    "Validation Loss: {:.5f}".format(validation_loss),
                    "Validation Accuracy: {:.5f}".format(accuracy.item()))
                
                # Reset the running loss to zero
                running_loss = 0


def save_checkpoint(model, class_to_idx, path='checkpoint.pth', architecture='vgg16', hidden_units=100, dropout=0.5, learning_rate=0.001, epochs=10):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'architecture': model.__class__.__name__,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': learning_rate,
    }
    torch.save(checkpoint, path)

def load_checkpoint(checkpoint_path='checkpoint.pth', device='gpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    loaded_architecture = checkpoint['architecture']
    loaded_hidden_units = checkpoint['hidden_units']
    loaded_dropout = checkpoint['dropout']
    loaded_learning_rate = checkpoint['learning_rate']
    
    loaded_model, _, _ = create_model(loaded_architecture, loaded_dropout, loaded_hidden_units, loaded_learning_rate, device)
    loaded_model.class_to_idx = checkpoint['class_to_idx']
    loaded_model.load_state_dict(checkpoint['state_dict'])

    return loaded_model


def process_image(img_path):
    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(256),  # resize the image to 256x256
        transforms.CenterCrop(224),  # crop the image to size 224x224 at the center
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # normalize the image data
    ])
    image_tensor = transform(image)
    return image_tensor


def predict(img, model, topk=5, device='gpu'):
    # Check if GPU is available and the user requested to use it
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda:0') # move the model to the GPU
    img_torch = process_image(img).unsqueeze_(0).float() # process the image and convert to PyTorch tensor
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda()) # forward pass the image through the model on GPU
    else:
        with torch.no_grad():
            output=model.forward(img_torch) # forward pass the image through the model on CPU
    prob = F.softmax(output.data,dim=1) # apply softmax to get the probabilities
    return prob.topk(topk) # return the top k probabilities and their corresponding classes
