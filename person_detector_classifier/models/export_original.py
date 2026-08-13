import torch
import timm

def export_pth_to_onnx(pth_path, onnx_path="mobilenetv4_utkface.onnx"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = timm.create_model(
        "mobilenetv4_conv_small.e3600_r256_in1k", 
        pretrained=False, 
        num_classes=6
    )
    
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Successfully exported {pth_path} to {onnx_path}")

if __name__ == "__main__":
    export_pth_to_onnx("mobilenetv4_utkface.pth")