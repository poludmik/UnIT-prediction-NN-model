import model
import torch

model_my = model.NN()
model_path = "weights.pth"


model_my.load_state_dict(torch.load(model_path))
model_my.eval()

with open('predict.txt', 'r') as f:
    lines = f.readlines()
    array = [float(t) for t in lines[-5:]]
    result = model_my(torch.Tensor(array))

with open('prediction.txt', 'w') as f:
    f.write(str(result.item()))
    f.write("\n")

