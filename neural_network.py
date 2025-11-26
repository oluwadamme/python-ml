import torch
import torch.nn as nn  # component for building neural networks
import torch.optim as optim  # component for training neural networks
import torch.nn.functional as F  # component for activation functions


distance = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [18.22], [22.33]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(1, 1)
)
# Time = W × Distance + B where W is the weight and B is the bias
loss_function = nn.MSELoss() # mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent - algorithm for adjusting the model's parameters to reduce the loss

for epoch in range(500): # number of iterations
    optimizer.zero_grad() # zero the gradients, clear the previous iteration's gradients
    outputs = model(distance) # forward pass
    loss = loss_function(outputs, times) # calculate the loss
    loss.backward() # back propagation - calculate the gradients, adjust the parameters
    optimizer.step() # update the parameters

with torch.no_grad():
    test_distance = torch.tensor([[5.0]], dtype=torch.float32)
    predicted_time = model(test_distance)
    print(f"Predicted time for distance 5km: {predicted_time.item():.1f}")
