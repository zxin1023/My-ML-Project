import torch
import torch.nn as nn
from torch.utils.data import DataLoader


model = Informer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(6):
    for batch in DataLoader(dataset, batch_size=32, shuffle=True):

        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
torch.save(model.state_dict(), 'models/trained/informer.pth')