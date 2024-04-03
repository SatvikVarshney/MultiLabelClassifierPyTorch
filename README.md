# PyTorch Multi-Label Classifier for Even MNIST dataset

## Overview
This project delves into the intricacies of multi-label classification, leveraging the powerful PyTorch framework to construct and train a neural network. The focus is on the Even MNIST dataset—a modified version of the famous MNIST dataset, comprising only the even digits. By reducing the image resolution for expedited training, this endeavor seeks not only to accurately classify these digits but also to serve as an explorative journey into the capabilities and utilization of neural networks within the domain of machine learning.

## Features
- **Neural Network Model:** Utilizes a fully connected deep neural network, designed and implemented in PyTorch, for the classification task.
- **Custom Dataset Handling:** Demonstrates the process of dataset preparation and loading within PyTorch, tailored for a specific classification problem.
- **Dynamic Parameter Adjustment:** Employs a JSON-based configuration system to easily tweak hyperparameters, allowing for flexible experimentation.
- **Performance Visualization:** Includes functionality to generate visual reports of the model's performance, detailing both loss and accuracy over time.

## Results

![Performance_graphs](https://github.com/SatvikVarshney/MultiLabelClassifierPyTorch/assets/114079530/b0c8d450-2329-4716-b123-4330efba54ef)


The performance of the neural network model is summarized through two key metrics: loss and accuracy. These metrics are visualized over the training epochs for both the training and test datasets.

### Loss Performance
The loss graph presents a sharp decline in the initial epoch, indicating a significant improvement in the model's learning capability at the beginning of the training process. Following this rapid descent, the curve flattens, suggesting that the model quickly reaches a point of diminishing returns with each subsequent epoch. Both the training and test loss exhibit a convergent behavior, with the test loss slightly higher than the training loss, a common and expected phenomenon due to the model being more familiar with the training data.

### Accuracy Performance
The accuracy graph showcases a swift ascent to high accuracy levels, reflecting the model's effectiveness at correctly classifying the even digits from the MNIST dataset. The training accuracy slightly outperforms the test accuracy, which may indicate a good fit to the data without significant overfitting. The model achieves near-peak accuracy rapidly, and further epochs bring marginal gains, indicating that the model efficiently captures the underlying patterns in the dataset.

Overall, the performance charts demonstrate a successful training phase, with the model achieving high levels of accuracy while maintaining a manageable difference between the training and test metrics, suggesting a balanced fit to the data. This balance between learning from the training data and generalizing to unseen test data is crucial for the robust performance of a machine learning model.


## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- numpy
- matplotlib
- tqdm

### Data Files
The project utilizes the `even_mnist.csv` dataset located in the `data` directory. This file contains grayscale images of even digits from the MNIST dataset, resized to 14x14 pixels for efficient training. Each row represents a flattened image followed by its label.

### Configuration Files
Model and training hyperparameters are defined in a JSON file located in the `param` directory. This allows for easy adjustments to parameters such as the learning rate, batch size, and the number of epochs without altering the main script.

Example `parameters.json`:
```json
{
	"learning rate": 0.001,
	"num iter": 10,
	"img_rows": 14,
	"img_cols": 14,
	"batch_size" : 128,
	"num_classes": 10	
}
```

## Installation

#### Clone this repository to your local machine:
``` bash
git clone https://github.com/YourUsername/MultiLabelClassifierPyTorch.git
```
#### Navigate to the project directory:
```bash
cd MultiLabelClassifierPyTorch
```
### Install the required dependencies:
``` bash
pip install -r requirements.txt
```

### Usage
To train the model and evaluate its performance, run:
```bash
python Scripts/main.py param/parameters.json
```
