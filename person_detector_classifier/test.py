import torch
import timm

device = "cpu"

# 1. Recreate model (MUST match training/export)
model = timm.create_model(
    "mobilenetv4_conv_small.e3600_r256_in1k",
    pretrained=False,
    num_classes=8
)

# 2. Load checkpoint
checkpoint = torch.load("./models/mobilenetv4_utkface_age_gender.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

# 3. Dummy inference
dummy = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    out = model(dummy)

print(out.shape)