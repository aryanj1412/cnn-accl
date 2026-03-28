import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as quantization
import os
from tqdm import tqdm

# Hyper parameters
lr = 0.001
epochs = 4

# 1. MODEL ARCHITECTURE (Residual f(x) + x, No BatchNorm)
class SimpleResidualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(8192, 8192)
        self.fc2 = nn.Linear(8192, num_classes)


        # Quantization Stubs (Required for Fixed-Point math)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x) # Entry to fixed-point

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x) + x
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        
        return self.dequant(x) # Exit to float

# 2. DEFINE 16-BIT CUSTOM CONFIGURATION
# This tells the model to simulate 16-bit integer precision (-32768 to 32767)
# 2. DEFINE 16-BIT CUSTOM CONFIGURATION (Fixed for MinMaxObserver)
qconfig_16bit = quantization.QConfig(
    activation=quantization.FakeQuantize.with_args(
        observer=quantization.MinMaxObserver,
        quant_min=-32768,
        quant_max=32767,
        dtype=torch.int16,               # Actual 16-bit integer type
        qscheme=torch.per_tensor_symmetric
    ),
    weight=quantization.FakeQuantize.with_args(
        observer=quantization.MinMaxObserver,
        quant_min=-32768,
        quant_max=32767,
        dtype=torch.int16,               # Actual 16-bit integer type
        qscheme=torch.per_tensor_symmetric # CHANGED: MinMaxObserver requires per_tensor
    )
)


# 3. DATA LOADING
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 4. PREPARE FOR 16-BIT QAT
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SimpleResidualCNN().to(device)
model.train()

model.qconfig = qconfig_16bit
model_qat = quantization.prepare_qat(model)

# 5. TRAINING LOOP
optimizer = optim.Adam(model_qat.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print("Starting 16-bit Quantization Aware Training...")
for epoch in tqdm(range(epochs)):
    print("enter training")
    j = 1
    for i, (images, labels) in enumerate(trainloader):
        print(j)
        j=j+1
        optimizer.zero_grad()
        outputs = model_qat(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 200 == 199:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

# 6. CONVERT AND SAVE
model_qat.eval()
model_int16 = quantization.convert(model_qat)

save_path = "model_16bit_fixed_point.pth"
torch.save(model_int16.state_dict(), save_path)
print(f"Success! 16-bit fixed-point model saved to: {os.path.abspath(save_path)}")

# 7. Print the names and tensor values for all layers
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param.data}")