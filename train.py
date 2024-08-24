import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split



class QuickDrawDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    # __len__ is called when len() is used on the object.
    def __len__(self):
        return len(self.data)
    
    # __getitem__ is called when you use indexing ([]) on the object. Each time the DataLoader requests an item
    # (or a batch of items), it internally calls the __getitem__ method of the QuickDrawDataset class and the conversions take place.
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        # Convert image to a PyTorch tensor and reshape to [28, 28]
        image = torch.tensor(image, dtype=torch.float32).reshape(28, 28)  # Reshape from [784] to [28, 28]
        image = image.unsqueeze(0)  # Add channel dimension: [1, 28, 28]
        label = torch.tensor(label, dtype=torch.long)
        return image, label



def create_combined_dataloaders(directory, batch_size=64, train_size=0.7, val_size=0.2):
    """ Creates the data loaders. """
    all_data = []
    all_labels = []
    class_labels = {}

    # Assign a unique label for each class
    for idx, file_name in enumerate(os.listdir(directory)):
        if file_name.endswith('.npy'):
            # Get the class name
            class_name = file_name.split('full_numpy_bitmap_')[1].split('.')[0]

            # Dictionary containing the mappings between the class names (string values) and the integers assigned to them
            class_labels[class_name] = idx

            # Load the .npy file. This results in a 3D Tensor of stacked images of the same class along the depth axis
            data = np.load(os.path.join(directory, file_name))

            # Reshape the data to be 28x28 if needed
            data = np.array([np.reshape(img, (28, 28)) for img in data])

            # Create labels for the class
            labels = np.full(data.shape[0], idx)  

            # Append data and labels. 
            all_data.append(data)
            all_labels.append(labels)

    # Combine all data and labels. This is done because all_data for example is a list of arrays, but we want a single 3D Tensor
    # containing all images of every class and not multiple ones for each class as in the case of all_data. 
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)

    # Shuffle the data
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]

    # Create the dataset
    dataset = QuickDrawDataset(all_data, all_labels)

    # Calculate split sizes
    total_size = len(dataset)
    train_count = int(train_size * total_size)
    val_count = int(val_size * total_size)
    test_count = total_size - train_count - val_count

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_count, val_count, test_count])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(class_labels)

    return train_loader, val_loader, test_loader


# Create data loaders
directory = r'C:\Users\xthan\Desktop\portofolio_projects\SketchEdAI\drawings'  # Replace with the path to your folder
train_loader, val_loader, test_loader = create_combined_dataloaders(directory)


# Define the CNN model
class CNN(nn.Module):
    """ The CNN model. """
    def __init__(self):
        # Network layers
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 8)  # 8 classes

    def forward(self, x):
        # Flow of information
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x, device):
        # Predict method
        # Set model to evaluation mode
        self.eval()
        # Send model to device
        x = x.to(device)  
        # The following mappings are based on the class_labels variable of the create_combined_dataloaders functions
        class_labels = {0: 'Alarm Clock', 1: 'Apple', 2: 'Axe', 3: 'Banana', 4: 'Bed', 5: 'Bench', 6: 'Bicycle', 7: 'Book'}
        # Disable gradient calculation
        with torch.no_grad():
            # Network outputs
            outputs = self.forward(x)
            # Output of the highest probability
            _, predicted = torch.max(outputs, 1)
            # Predicted label
            predicted_label = class_labels[predicted.item()]  # Convert index to label
        return predicted_label




# Training / Validation function
def train_and_validate(model, trainloader, valloader, criterion, optimizer, num_epochs, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """ Train / Validation function. """
    print(f"Using device: {device}")
    # Send model to device
    model.to(device)
    # Best model in regards to validation accuracy
    best_model_wts = None
    # Best validation accuracy
    best_acc = 0.0
    # Trains, validation epoch losses
    train_losses, val_losses = [], []

    # Epoch iterations
    for epoch in range(num_epochs):
        # Set model to train mode
        model.train()
        # Initiallize train epoch loss
        running_loss = 0.0
        # Iterate through the train batches
        for inputs, labels in trainloader:
            # Send data, labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Initiallize gradients
            optimizer.zero_grad()
            # Predict outputs
            outputs = model(inputs)
            # Calculate loss based on the defined criterion 
            loss = criterion(outputs, labels)
            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            # Add total train loss for the batch to running_loss
            running_loss += loss.item() * inputs.size(0)
        # Calculate average epoch train loss
        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)

        # Set model to evaluation mode
        model.eval()
        # Inittiallize validation loss
        running_loss = 0.0
        # Initiallize number of correctly predicted cases and the total number of cases
        correct, total = 0, 0
        # Disable gradient calculations
        with torch.no_grad():
            # Iterate through the validation batches
            for inputs, labels in valloader:
                # Send inputs and labels to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Predict outputs
                outputs = model(inputs)
                # Calculate loss 
                loss = criterion(outputs, labels)
                # Add total validation loss for the batch to running_loss
                running_loss += loss.item() * inputs.size(0)
                # Get the outputs of highest probability
                _, predicted = torch.max(outputs, 1)
                # Accumulate the number of total cases in the batch 
                total += labels.size(0)
                # Accumulate number of correctly classified cases for the batch
                correct += (predicted == labels).sum().item()
        # Calculate average validation loss for the epoch
        val_loss = running_loss / len(valloader.dataset)
        val_losses.append(val_loss)
        # Calulate accuray on the validation set for this epoch
        val_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Save the best model (highest validation accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
    print("Best validation accuracy achieved :",best_acc )
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model_.pth')
    
    # Plotting the losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.show()

    return model

def evaluate_model(model, test_loader):
    """ Evaluating on the test set function. """

    # Evaluating the trained model on the test set

    # Initialize the total number of cases and the number of correctly classified cases
    correct = 0
    total = 0

    # Disable gradient calculations
    with torch.no_grad():
        # Iterate through test batches, make predictions and accumulate the correct number of classifications as well as the total
        # number of cases
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy on test set
    accuracy = correct / total
    return accuracy


# Model, loss function, optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

time1 = time()

# Train the model
trained_model = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

time2 = time()
# Calculate train time
print("Time spent training",(time2-time1)/60, "minutes")


# Evaluate the model on the test dataset
# Instantiate the model
model = CNN()

# Load the saved model weights
model.load_state_dict(torch.load('best_model.pth'))

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Calculate accuracy on test set
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')

