import os
import time
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UTKDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = cv2.imread(row["path"])
            if img is None:
                return self.__getitem__((idx + 1) % len(self.df))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = row["label"]
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        except:
            return self.__getitem__((idx + 1) % len(self.df))


def age_to_class(age):
    if age <= 10: return 0
    elif age <= 20: return 1
    elif age <= 30: return 2
    elif age <= 40: return 3
    elif age <= 60: return 4
    else: return 5


def save_model_and_onnx(model, save_path="mobilenetv4_utkface"):
    model.eval()
    torch.save({"model_state_dict": model.state_dict()}, f"{save_path}.pth")
    print(f"[INFO] Saved PyTorch checkpoint: {save_path}.pth")
    
    dummy_input = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
    
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=True
    )
    print(f"[INFO] Saved ONNX model: {save_path}.onnx")

if __name__ == '__main__':
    target_dir = r"D:\UFK\UTK_Final"

    files = []
    ages = []
    for f in os.listdir(target_dir):
        try:
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            age = int(f.split("_")[0])
            path = os.path.join(target_dir, f)
            if os.path.getsize(path) < 1000:
                continue
            files.append(path)
            ages.append(age)
        except:
            continue

    df = pd.DataFrame({"path": files, "age": ages})
    df["label"] = df["age"].apply(age_to_class)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = UTKDataset(train_df, train_tfms)
    val_ds = UTKDataset(val_df, val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "mobilenetv4_conv_small.e3600_r256_in1k",
        pretrained=True,
        num_classes=6
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(35):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        train_loop = tqdm(train_loader, desc=f"🚀 Epoch {epoch+1:02d}/35 [Train]", leave=False)

        for imgs, labels in train_loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            current_lr = optimizer.param_groups[0]["lr"]
            train_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(100 * train_correct / train_total):.2f}%",
                lr=f"{current_lr:.1e}",
            )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_loop = tqdm(val_loader, desc=f"🧪 Epoch {epoch+1:02d}/35 [Val]", leave=False)

        with torch.no_grad():
            for imgs, labels in val_loop:
                imgs, labels = imgs.to(device), labels.to(device)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(100 * val_correct / val_total):.2f}%",
                )

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - start_time

        print(
            f"📊 Epoch {epoch+1:02d}/35 Metrics Summary\n"
            f"   ⏱️  Duration: {epoch_time:.1f}s | 🔄 LR: {current_lr:.1e}\n"
            f"   📉 Loss: [Train: {epoch_train_loss:.4f}] • [Val: {epoch_val_loss:.4f}]\n"
            f"   🎯 Acc:  [Train: {epoch_train_acc:.2f}%] • [Val: {epoch_val_acc:.2f}%]\n"
            f"{'-'*55}"
        )

    save_model_and_onnx(model)