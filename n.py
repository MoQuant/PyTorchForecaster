import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Neural network class for machine learning process made with PyTorch
class Model(nn.Module):

    def __init__(self, window, output):
        super(Model, self).__init__()

        # Define layers of the neural network with appropriate dimensions
        self.layer1 = nn.Linear(window, 250)
        self.layer2 = nn.Linear(250, 100)
        self.layer3 = nn.Linear(100, output)
        self.relu = nn.ReLU()
    
    # Forward propigation
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

# Compute the dataframe with the technical indicators for the stock loaded in it
def AddMetrics(x, days=50):
    dataset = []
    n = len(x)
    for i in range(days, n):
        # Breaks the dataset into a window
        box = x[i-days:i]

        # Compute the mean and standard deviation of the stock data
        ma = np.mean(box)
        sd = np.std(box)

        # Compute the bollinger bands
        low = ma - 2*sd
        up = ma + 2*sd

        # Store each iteration in dataset
        dataset.append([x[i], ma, low, up])

    # Return pandas dataframe
    return pd.DataFrame(dataset, columns=['Price','MA','LB','UB'])

# Splits the data into training and testing data
def TrainTest(x, prop=0.85):
    I = int(prop*len(x))
    train = x[:I]
    test = x[I:]
    return train, test

# Build PyTorch tensors containing the stock data 
def Inputs(dataset, window=100, output=30):
    n = len(dataset)
    training_data = []
    for w in range(window, n-output+1):
        # Pulls all columns from dataset within the input bounds
        a1, a2, a3, a4 = np.array(dataset[w-window:w]).T.tolist()

        # Pulls b1 which are the close prices within the output bounds
        b1, b2, b3, b4 = np.array(dataset[w:w+output]).T.tolist()

        # Add data to training set
        training_data.append([a1 + a2 + a3 + a4, b1])

    # Convert the lists storing inputs and outputs into PyTorch tensors
    IN = [torch.tensor(item[0], dtype=torch.float32) for item in training_data]
    OUT = [torch.tensor(item[1], dtype=torch.float32) for item in training_data]
    return torch.stack(IN), torch.stack(OUT)

# Build the PyTorch tensor for the values to be predicted based on input
def Outputs(dataset, window):
    a1, a2, a3, a4 = np.array(dataset[-window:]).T.tolist()
    X = torch.tensor(a1 + a2 + a3 + a4, dtype=torch.float32)
    return torch.stack((X,)), a1

# Load the stock data, in this case Apple
data = pd.read_csv('AAPL.csv')

# Reverse the stock data to get the oldest prices first and the newest prices last
close = data['adjClose'].values.tolist()[::-1]

# Declare parameters to train the model
epochs = 2000
window = 100
output = 30
learning_rate = 0.0001

# Load the dataframe containing technical indicators
df = AddMetrics(close)

# Initialize the neural network class into the model object
model = Model(int(window*4), int(output))

# Split the data for training and testing
train, test = TrainTest(df)

# Extract the PyTorch tensors based on the trainng data
X, Y = Inputs(train, window=window, output=output)

# Declare a loss function and Adam optimizer for the Neural Network
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Run training with the set number of epochs
for epoch in range(epochs):
    # Compute outputs from model and compare them to the provided output
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Set gradient and backpropigate the Neural Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Prints out how many epochs are left in training
    if (epoch + 1) % 100 == 0:
        print('Epochs left: ', epochs-epoch-1)

# Retrieve the testing torch tensor
XX, history = Outputs(test, window)

# Generate predictions
with torch.no_grad():
    test_outputs = model(XX)

predictions = test_outputs[-1].numpy().tolist()

# Declare plot figure and subplot
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

# Graph the historical stock data
xa = list(range(len(history)))
ax.plot(xa, history, color='red', label='Historical')

# Graph the predicted future stock prices
xb = list(range(len(history), len(history)+len(predictions)))
ax.plot(xb, predictions, color='green', label='Predictions')

plt.show()








    
