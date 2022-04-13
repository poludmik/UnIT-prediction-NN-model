import matplotlib.pyplot as plt
import model
import random
import torch
import psycopg2
import create_database

directory = 'PG24A'

data = create_database.Data(directory)
# data.create_dataset()
# data.write_data_to_file()

data.read_dataset_from_file()

trained_model = model.NN()
#model_path = "weights.pth"
model_path = None

if model_path is not None:
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()

optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()

losses = []
iterations = []
counter = 0

sets = data.number_of_data
epoch_num = 10
for j in range(epoch_num):

    for i in range(sets):
        print("i:", counter, "epochs:", sets * epoch_num)

        # Get inputs
        inputs = data.dataset[i]
        targets = data.targets[i]

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = trained_model(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        losses.append(loss.item())
        iterations.append(counter)
        counter += 1

        # Perform optimization
        optimizer.step()

torch.save(trained_model.state_dict(), "weights.pth")


# plotting
x = iterations
y = losses

fig = plt.figure()

plt.plot(x, y)
plt.grid()

fig.suptitle('Forward pass MSElosses')
plt.xlabel('epoch')
plt.ylabel('MSEloss')
fig.savefig('test.jpg')

plt.show()



