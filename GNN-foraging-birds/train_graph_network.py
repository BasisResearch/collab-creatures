import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(4, 16, heads=2)  # Input features are x, y, dx, dy
        self.conv2 = GATConv(32, 2, heads=1)  # Output features are new_dx, new_dy

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):  # Loop over the dataset multiple times
    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         optimizer.zero_grad()  # Clear gradients.
         out = model(data)  # Perform a single forward pass.
         loss = F.mse_loss(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
