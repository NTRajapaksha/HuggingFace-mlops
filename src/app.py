import gradio as gr
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from pathlib import Path
import time

# 1. Global Setup
# We load the model once (at startup) so we don't reload it for every user request.
# This is a standard production optimization.
model_path = Path(__file__).parent.parent / "onnx_model"

print(f"⏳ Loading Quantized Model from: {model_path}")

# Verify the file exists (helps debugging)
if not model_path.exists():
    raise FileNotFoundError(f"❌ Model folder not found at {model_path}")

model = ORTModelForSequenceClassification.from_pretrained(
    model_path, 
    file_name="model_quantized.onnx",
    local_files_only=True  # <--- FORCE it to look locally
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("✅ Model Loaded!")

def predict(text):
    """
    1. Records start time
    2. Runs prediction using ONNX Runtime (CPU)
    3. Calculates latency
    4. Returns results
    """
    start_time = time.time()
    
    # Run inference
    result = pipe(text)
    
    end_time = time.time()
    
    # Parse results
    label = result[0]['label']
    score = result[0]['score']
    
    # Calculate latency in milliseconds
    latency_ms = (end_time - start_time) * 1000
    latency_str = f"{latency_ms:.2f} ms"
    
    return label, score, latency_str

# 2. Define the Interface
# We use Gradio to create a clean web UI.
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter Financial News", 
        placeholder="e.g., The company reported a 20% increase in revenue.",
        lines=2
    ),
    outputs=[
        gr.Label(label="Sentiment Analysis"),
        gr.Number(label="Confidence Score"),
        gr.Textbox(label="Inference Latency (CPU)")
    ],
    title="⚡ ONNX Optimized Financial Sentiment",
    description=f"Running a **Quantized DistilBERT** model ({model_path}).\nNotice how fast the latency is even on a standard CPU!",
    examples=[
        ["The company reported record profits this quarter."],
        ["Stocks plummeted due to unexpected inflation reports."],
        ["The merger is expected to complete by Q4."]
    ]
)

if __name__ == "__main__":
    iface.launch()