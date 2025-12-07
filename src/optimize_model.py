import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

# 1. Configuration
# We use DistilBERT because it's small and fast—perfect for this demo.
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_PATH = Path("onnx_model")

def convert_and_quantize():
    print(f"🚀 Starting optimization for: {MODEL_ID}")
    
    # 2. Load and Export to ONNX
    # The 'export=True' flag tells Optimum to run the conversion automatically.
    print("⏳ Loading model and converting to ONNX (this might take a minute)...")
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Save the base ONNX model (still 32-bit float)
    model.save_pretrained(ONNX_PATH)
    tokenizer.save_pretrained(ONNX_PATH)
    print("✅ Base ONNX model saved.")

    # 3. Quantize (Float32 -> Int8)
    # This reduces the model size by ~3x-4x with minimal accuracy loss.
    print("⏳ Quantizing model to Int8...")
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(model)
    
    quantizer.quantize(
        save_dir=ONNX_PATH,
        quantization_config=qconfig,
    )
    print(f"🎉 Optimized model saved to: {ONNX_PATH}")
    
    # Optional: Print file size comparison
    base_size = os.path.getsize(ONNX_PATH / "model.onnx") / (1024 * 1024)
    quant_size = os.path.getsize(ONNX_PATH / "model_quantized.onnx") / (1024 * 1024)
    print(f"\n📊 Stats:")
    print(f"Original ONNX Size: {base_size:.2f} MB")
    print(f"Quantized ONNX Size: {quant_size:.2f} MB")
    print(f"Compression Ratio: {base_size / quant_size:.2f}x")

if __name__ == "__main__":
    convert_and_quantize()