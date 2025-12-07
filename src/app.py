import os
import gradio as gr
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
import time
import wandb

# --- 1. Initialize Observability ---
# (Wrap in try-except in case secret is missing, to prevent crash)
try:
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="financial-sentiment-monitor", name="production-serving")
except Exception as e:
    print(f"⚠️ WandB Init failed (continuing without it): {e}")

# --- 2. Load Model & Debugging ---
# This path logic works for: root/src/app.py -> root/onnx_model
model_path = Path(__file__).parent.parent / "onnx_model"

print(f"🔍 DEBUG: Current working directory: {Path.cwd()}")
print(f"🔍 DEBUG: Looking for model at: {model_path}")

# Check if folder exists
if not model_path.exists():
    print("❌ ERROR: The 'onnx_model' folder does NOT exist on the server.")
    print(f"📂 Contents of parent folder ({model_path.parent}):")
    print(sorted([p.name for p in model_path.parent.glob("*")]))
    raise FileNotFoundError(f"Model folder missing at {model_path}")
else:
    print("✅ Folder found. Listing contents:")
    # List all files in the folder to verify 'model_quantized.onnx' is there
    files = [f.name for f in model_path.iterdir()]
    print(files)
    
    if "model_quantized.onnx" not in files:
        raise FileNotFoundError("❌ 'model_quantized.onnx' is MISSING from the folder!")

# Load with explicit local forcing
try:
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path, 
        file_name="model_quantized.onnx",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print("✅ Model & Pipeline Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR loading model: {e}")
    raise e

def predict(text):
    start_time = time.time()
    try:
        result = pipe(text)
        end_time = time.time()
        
        label = result[0]['label']
        score = result[0]['score']
        latency_ms = (end_time - start_time) * 1000
        
        # Log to WandB if active
        if wandb.run:
            wandb.log({
                "input_text": text,
                "prediction": label,
                "confidence": score,
                "latency_ms": latency_ms,
                "timestamp": time.time()
            })
            
        return label, score, f"{latency_ms:.2f} ms"
    except Exception as e:
        return "Error", 0.0, str(e)

# --- 3. Interface ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter Financial News", lines=2),
    outputs=[
        gr.Label(label="Sentiment"),
        gr.Number(label="Confidence"),
        gr.Textbox(label="Inference Latency (CPU)")
    ],
    title="⚡ ONNX Optimized Financial Sentiment",
    description="Running a Quantized DistilBERT model on CPU.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()