import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self, window, output):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(window, 250)
        self.layer2 = nn.Linear(250, 100)
        self.layer3 = nn.Linear(100, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

def AddMetrics(x, days=50):
    dataset = []
    n = len(x)
    for i in range(days, n):
        box = x[i-days:i]
        ma = np.mean(box)
        sd = np.std(box)
        low = ma - 2*sd
        up = ma + 2*sd
        dataset.append([x[i], ma, low, up])
    return pd.DataFrame(dataset, columns=['Price','MA','LB','UB'])

def TrainTest(x, prop=0.85):
    I = int(prop*len(x))
    train = x[:I]
    test = x[I:]
    return train, test
    
def Inputs(dataset, window=100, output=30):
    n = len(dataset)
    training_data = []
    for w in range(window, n-output+1):
        a1, a2, a3, a4 = np.array(dataset[w-window:w]).T.tolist()
        b1, b2, b3, b4 = np.array(dataset[w:w+output]).T.tolist()
        training_data.append([a1 + a2 + a3 + a4, b1])
    IN = [torch.tensor(item[0], dtype=torch.float32) for item in training_data]
    OUT = [torch.tensor(item[1], dtype=torch.float32) for item in training_data]
    return torch.stack(IN), torch.stack(OUT)

def Outputs(dataset, window):
    a1, a2, a3, a4 = np.array(dataset[-window:]).T.tolist()
    X = torch.tensor(a1 + a2 + a3 + a4, dtype=torch.float32)
    return torch.stack((X,)), a1


data = pd.read_csv('AAPL.csv')
close = data['adjClose'].values.tolist()[::-1]

epochs = 2000
window = 100
output = 30
learning_rate = 0.0001

df = AddMetrics(close)

model = Model(int(window*4), int(output))

train, test = TrainTest(df)


X, Y = Inputs(train, window=window, output=output)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epochs left: ', epochs-epoch-1)


XX, history = Outputs(test, window)

with torch.no_grad():
    test_outputs = model(XX)

predictions = test_outputs[-1].numpy().tolist()


fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

xa = list(range(len(history)))
ax.plot(xa, history, color='red', label='Historical')

xb = list(range(len(history), len(history)+len(predictions)))
ax.plot(xb, predictions, color='green', label='Predictions')

plt.show()








    
