# -*- coding: utf-8 -*-
"""
ONNX Export CLI for MobileNetV4 Models
======================================
Usage:
    python scripts/export_onnx.py --model-weights models/trained/mobilenetv4_kenya_finetuned.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch
import timm

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch MobileNetV4 to ONNX")
    parser.add_argument("--model-weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-onnx", default=None, help="Output .onnx path")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of target classes")
    args = parser.parse_args()

    if not os.path.exists(args.model_weights):
        print(f"Error: Weights not found at {args.model_weights}")
        sys.exit(1)

    out_onnx = args.output_onnx or args.model_weights.replace(".pt", ".onnx")

    print(f"Loading MobileNetV4 Conv Small from {args.model_weights}...")
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_weights, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to ONNX format (opset 18) -> {out_onnx}...")
    torch.onnx.export(
        model,
        dummy_input,
        out_onnx,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    size_mb = round(os.path.getsize(out_onnx) / (1024 * 1024), 2)
    print(f"Export successful! ONNX model saved to {out_onnx} ({size_mb} MB)")

if __name__ == "__main__":
    main()
