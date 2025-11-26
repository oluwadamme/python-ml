import torch
import torch.nn as nn  # component for building neural networks
import torch.optim as optim  # component for training neural networks
import torch.nn.functional as F  # component for activation functions
import helper_utils

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [18.22], [22.33]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(1, 1)
)
# Time = W × Distance + B where W is the weight and B is the bias
loss_function = nn.MSELoss() # mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent - algorithm for adjusting the model's parameters to reduce the loss

for epoch in range(500): # number of iterations
    optimizer.zero_grad() # zero the gradients, clear the previous iteration's gradients
    outputs = model(distances)  # forward pass
    loss = loss_function(outputs, times) # calculate the loss
    loss.backward() # back propagation - calculate the gradients, adjust the parameters
    optimizer.step() # update the parameters
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

with torch.no_grad():
    test_distance = torch.tensor([[5.0]], dtype=torch.float32)
    predicted_time = model(test_distance)
    print(f"Predicted time for distance 5km: {predicted_time.item():.1f}")
    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        print("\nDecision: Take the job. You can make it!")
# helper_utils.plot_results(model, distances, times)

# Access the first (and only) layer in the sequential model
# layer = model[0]

# # Get weights and bias
# weights = layer.weight.data.numpy()
# bias = layer.bias.data.numpy()

# print(f"Weight: {weights}")
# print(f"Bias: {bias}")

with torch.no_grad():
    predictions = model(helper_utils.new_distances)
    print("\nPredicted delivery times for new distances:")
    new_loss = loss_function(predictions, helper_utils.new_times)
    print(f"Loss on new, combined data: {new_loss.item():.2f}")

# helper_utils.plot_nonlinear_comparison(
#     model, helper_utils.new_distances, helper_utils.new_times
# )
