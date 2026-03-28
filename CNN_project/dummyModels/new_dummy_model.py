import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as quantization
import os

class SimpleResidualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d((2,2))
        self.fc = nn.Linear(16, num_classes)


        # Quantization Stubs (Required for Fixed-Point math)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x) # Entry to fixed-point

        x = self.conv1(x) = x
        x = self.relu(x)

        x = self.pool(x)
        
        x = self.conv2(x) + x
        x = self.relu(x)
        
        x = self.pool(x)

        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return self.dequant(out) # Exit to float

# 2. DATA LOADING (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 3. TRAINING SETUP
device = torch.device("cpu") # Quantization is best handled on CPU in PyTorch
model = SimpleResidualCNN().to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. TRAINING LOOP (Floating Point)
print("Starting Training...")
model.train()
for epoch in range(2):  # Small number of epochs for demonstration
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 200 == 199: 
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print("Training Finished.")

# 5. FIXED-POINT QUANTIZATION (PTQ)
print("Starting Quantization...")
model.eval()

# Configure the quantization backend (x86 or ARM)
model.qconfig = quantization.get_default_qconfig('fbgemm')

# Prepare model (adds observers to record data ranges)
model_prepared = quantization.prepare(model)

# Calibrate: Run one pass of data so the model knows the number ranges
with torch.no_grad():
    for images, _ in trainloader:
        model_prepared(images)
        break # One batch is usually enough for scale calibration

# Convert to actual INT8 Fixed-Point
model_int8 = quantization.convert(model_prepared)

# 6. STORAGE
save_path = "model_fixed_point_final.pth"
torch.save(model_int8.state_dict(), save_path)

print(f"Success! Fixed-point model saved to: {os.path.abspath(save_path)}")