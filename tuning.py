import torch
import time

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(16 * 326 * 326, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Instantiate the Model
model = MyModel()

# Load the Checkpoint
checkpoint = torch.load("checkpoint.pth", map_location="cpu") # cpu check
model.load_state_dict(checkpoint["model_state_dict"])

# Optim
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move model to the device
model.to(device)

# pinned memory CPU → GPU
pin_memory = device == "cuda"

# Data Dummy
dummy_input = torch.randn(16, 3, 326, 326)
if pin_memory:
    dummy_input = dummy_input.pin_memory()  # Pinned memory

# (CPU → GPU)
torch.cuda.synchronize()
start_transfer = time.time()
dummy_input = dummy_input.to(device)
torch.cuda.synchronize()
transfer_time = time.time() - start_transfer

model.eval()
torch.cuda.synchronize()
start_infer = time.time()

with torch.cuda.amp.autocast():  # AMP optim
    with torch.no_grad():
        model(dummy_input)

torch.cuda.synchronize()
inference_time = time.time() - start_infer

# Output
print(f"Transfer Time: {transfer_time:.6f} sec")
print(f"Inference Time: {inference_time:.6f} sec")
