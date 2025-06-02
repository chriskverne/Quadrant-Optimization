import pandas as pd
from helper.fetch_mnist import preprocess_image
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
df = pd.read_csv('./data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values
x = preprocess_image(x, 100)



x_tensor = torch.FloatTensor(x)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(100, 8, bias=False), nn.Linear(8, 4, bias=False))
optimizer = torch.optim.SGD(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {total_loss:.4f}, Acc {correct/total:.4f}')