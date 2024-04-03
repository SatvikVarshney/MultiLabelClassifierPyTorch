import numpy as np
import random
import json
import sys

# PyTorch for neural networks
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import time

# Define the Neural Network for Digit Classification
class DigitClassifier(nn.Module):
    def __init__(self):   
        super(DigitClassifier, self).__init__()
        
        self.fc1 = nn.Linear(img_rows * img_cols, 128)  # Input to first hidden layer
        self.fc2 = nn.Linear(128, 64)  # First hidden to second hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Second hidden to output layer
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        
        # Activation and Dropout
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.optim = optim.Adam(self.parameters(), lr)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def train_model(self, X_train, Y_train):
        self.optim.zero_grad()
        output = self.forward(X_train)
        loss = self.criterion(output, Y_train)
        predictions = torch.argmax(output, dim=1)
        accuracy = (Y_train == predictions).float().sum()
        loss.backward()
        self.optim.step()
        return loss.item(), accuracy.item()

    def evaluate_model(self, dataloader):
        total_accuracy = 0
        total_loss = 0
        for X_batch, Y_batch in dataloader:
            with torch.no_grad():
                loss = self.criterion(self.forward(X_batch.float()), Y_batch.long())
                loss = loss.item()  # Get the loss number
                total_loss += loss
                
                predictions = torch.argmax(self.forward(X_batch.float()), dim=1)
                total_accuracy += (Y_batch == predictions).float().sum().item()

        avg_accuracy = total_accuracy / (len(dataloader) * batch_size)
        avg_loss = total_loss / len(dataloader)

        return avg_accuracy, avg_loss

    def predict(self, X):
        with torch.no_grad():
            output = self.forward(X)
            return torch.argmax(output, dim=1)

if __name__ == "__main__":
    
    if sys.argv[1] == "--help":
        print("Usage: python main.py config/parameters.json even_mnist.csv performance_report.html")
        print("Argument 1: Path to the JSON configuration file containing hyperparameters (e.g., learning rate, batch size)")
        print("  - Default: config/parameters.json")
        print("  - File Type: JSON")
        print("Argument 2: Path to the even MNIST dataset file")
        print("  - Default: even_mnist.csv")
        print("  - File Type: CSV")
        print("Argument 3: Path to the output HTML file summarizing the model's performance")
        print("  - Default: reports/performance_report.html")
        print("  - File Type: HTML")
    else:
        # Load arguments or set defaults
        param_file = sys.argv[1] if len(sys.argv) > 1 else "config/parameters.json"
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else "even_mnist.csv"
        output_html = sys.argv[3] if len(sys.argv) > 3 else "reports/performance_report.html"
        
        # Load dataset
        dataset = genfromtxt(dataset_path, delimiter=",")
        labels = dataset[:, -1]  # Extract labels
        features = np.delete(dataset, -1, axis=1)  # Extract features

        # Split dataset into training and test sets
        X_test, y_test = features[-3000:], labels[-3000:]
        X_train, y_train = features[:-3000], labels[:-3000]
        
        # Load hyperparameters from JSON file
        with open(param_file) as f:
            config = json.load(f)
        learning_rate = config['learning_rate']
        img_height = config['img_height']  # 14
        img_width = config['img_width']  # 14
        batch_size = config['batch_size']  # 128
        num_classes = config['num_classes']  # 5
        num_epochs = config['num_epochs']

        # Normalize feature data
        X_train /= 255
        X_test /= 255

        # Convert to PyTorch tensors
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
        tensor_y_train = torch.tensor(y_train, dtype=torch.long)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.long)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_y_train)
        test_dataset = torch.utils.data.TensorDataset(tensor_X_test, tensor_y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Prepare HTML output file
        with open(output_html, "w") as html_file:
            html_file.write("<html><head><title>Performance Report</title></head><body>")
            html_file.write("<h1>Model Performance Summary</h1>")

            # Instantiate and train the model
            classifier = DigitClassifier()
            training_losses, training_accuracies = [], []
            test_losses, test_accuracies = [], []

            for epoch in range(num_epochs):
                train_loss, train_acc = 0, 0
                # Training phase
                for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                    loss, acc = classifier.train_model(X_batch, y_batch)
                    train_loss += loss
                    train_acc += acc

                # Calculate average loss and accuracy
                avg_train_loss = train_loss / len(train_loader)
                avg_train_acc = train_acc / (len(train_loader) * batch_size)
                training_losses.append(avg_train_loss)
                training_accuracies.append(avg_train_acc)

                # Testing phase
                avg_test_acc, avg_test_loss = classifier.evaluate_model(test_loader)
                test_losses.append(avg_test_loss)
                test_accuracies.append(avg_test_acc)

                # Log to console
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {avg_train_acc:.4f}")
                print(f"Epoch {epoch+1}: Test Loss = {avg_test_loss:.4f}, Test Accuracy = {avg_test_acc:.4f}")

                # Write to HTML file
                html_file.write(f"<p>Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {avg_train_acc:.4f}</p>")
                html_file.write(f"<p>Epoch {epoch+1}: Test Loss = {avg_test_loss:.4f}, Test Accuracy = {avg_test_acc:.4f}</p>")

            html_file.write("</body></html>")

        plt.style.use('dark_background')
        plt.figure(figsize=(15, 6))
        
        # Plot training and test loss
        plt.subplot(121)
        plt.plot(range(num_epochs), training_losses, label="Training Loss", color="orange", linewidth=4, marker='o')
        plt.plot(range(num_epochs), test_losses, label="Test Loss", color="green", linewidth=4, linestyle='--', marker='o')
        plt.legend()
        plt.title("Loss Performance Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Plot training and test accuracy
        plt.subplot(122)
        plt.plot(range(num_epochs), training_accuracies, label="Training Accuracy", color="orange", linewidth=4, marker='o')
        plt.plot(range(num_epochs), test_accuracies, label="Test Accuracy", color="green", linewidth=4, linestyle='--', marker='o')
        plt.legend()
        plt.title("Accuracy Performance Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        # Save and display the plot
        plot_filename = output_html.replace(".html", "_graphs.png")
        plt.savefig(plot_filename)
        # plt.show(block=True) # Uncomment to display the plot
        
        # Embed the plot in the HTML output file
        plot_filename_short = plot_filename.split("/")[-1]  # Assuming a simple path structure
        with open(output_html, "a") as html_file:  # Appending the image to the existing HTML file
            html_file.write(f"<div><img src='{plot_filename_short}' alt='Training and Testing Performance Graphs'></div>")
            html_file.write("</body></html>")
        
        print(f"Report with embedded graphs saved to {output_html}")

        # Demonstrate predictions on a batch from the test set
        classifier = DigitClassifier()  # Assuming classifier is your trained model
        for X_batch, y_true in test_loader:
            y_pred = classifier.predict(X_batch.float())
            print("Predicting digits and evaluating model accuracy:")
            print("True labels: ")
            print(y_true)
            print("Predicted labels: ")
            print(y_pred)
            accuracy = (y_true == y_pred).float().mean()
            print(f"Batch accuracy: {accuracy.item() * 100:.2f}%")
            break  # Remove break to evaluate on all test batches

