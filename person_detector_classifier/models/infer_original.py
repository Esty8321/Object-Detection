import os
import time
import random
import cv2
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

def run_inference(image_folder, num_images=10, model_path="mobilenetv4_utkface.pth", delay_ms=4000):
    device = "cpu"
    
    # Load Model
    model = timm.create_model("mobilenetv4_conv_small.e3600_r256_in1k", pretrained=False, num_classes=6)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessing Pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Category Mappings
    class_mapping = {
        0: "0-10 Years (Child)",
        1: "11-20 Years (Teen)",
        2: "21-30 Years (Youth)",
        3: "31-40 Years (Adult)",
        4: "41-60 Years (Middle Age)",
        5: "60+ Years (Senior)"
    }
    race_mapping = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
    gender_mapping = {0: "Male", 1: "Female"}

    # Fetch and Randomly Sample Dataset
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"[Error] No valid images found in {image_folder}")
        return

    image_files = random.sample(image_files, min(num_images, len(image_files)))

    total_inference_time = 0.0
    processed_count = 0

    cv2.namedWindow("MobileNetV4 Premium Inference (CPU)", cv2.WINDOW_AUTOSIZE)

    for img_path in image_files:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue

        # --- Parse Ground Truth Attributes from Filename ---
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        true_age = "Unknown"
        true_gender = "Unknown"
        true_race = "Unknown"
        
        if len(parts) >= 3:
            try:
                true_age = f"{parts[0]} Years"
                true_gender = gender_mapping.get(int(parts[1]), "Unknown")
                true_race = race_mapping.get(int(parts[2]), "Unknown")
            except (ValueError, KeyError):
                pass

        # --- Visual Layout and HUD Settings ---
        display_img = cv2.resize(orig_img, (640, 640))
        overlay = display_img.copy()
        
        # Draw dynamic translucent display console panel
        cv2.rectangle(overlay, (0, 460), (640, 640), (24, 24, 28), -1)
        cv2.addWeighted(overlay, 0.88, display_img, 0.12, 0, display_img)
        # Top boundary accent line
        cv2.rectangle(display_img, (0, 458), (640, 460), (235, 156, 42), -1)

        # --- Execute Core Model Inference ---
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
        latency = time.perf_counter() - start_time
        
        total_inference_time += latency
        processed_count += 1

        # --- Text Generation & Overlay Rendering ---
        gt_text = f"Target: {true_age}    {true_gender}    {true_race}"
        pred_text = f"Prediction: {class_mapping[pred.item()]} ({conf.item()*100:.1f}%)"
        metrics_text = f"Latency: {latency*1000:.1f}ms      Device: CPU      [Q] to Quit"

        # Row 1: Ground Truth Metadata
        cv2.putText(display_img, gt_text, (25, 500), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        # Row 2: Model Output Group Classification (Bold / Golden Accent)
        cv2.putText(display_img, pred_text, (25, 545), cv2.FONT_HERSHEY_DUPLEX, 0.70, (235, 156, 42), 1, cv2.LINE_AA)
        # Row 3: Pipeline Hardware Metrics
        cv2.putText(display_img, metrics_text, (25, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 143, 150), 1, cv2.LINE_AA)

        cv2.imshow("MobileNetV4 Premium Inference (CPU)", display_img)
        
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            print("\n[INFO] Execution interrupted by user.")
            break

    cv2.destroyAllWindows()

    # --- Print Benchmark Summary ---
    if processed_count > 0:
        avg_fps = processed_count / total_inference_time
        avg_latency = (total_inference_time / processed_count) * 1000
        print(
            f"\n📊 Inference Execution Summary\n"
            f"   ⏱️  Total Processed : {processed_count} frames\n"
            f"   ⚡ Pure Core Performance: {avg_fps:.2f} FPS\n"
            f"   🎯 Latency Profile  : {avg_latency:.1f} ms\n"
            f"{'-'*55}"
        )

if __name__ == "__main__":
    # Adjust target path and defaults here if needed
    run_inference(r"../Aviv_2026/pinterest_500_clean/man", num_images=10)