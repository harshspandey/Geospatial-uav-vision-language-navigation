from llmcompressor import compress_model
from transformers import AutoTokenizer

model_path = "/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT/model_weight"
output_path = "/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT/model_quantized"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantization config (AWQ-style)
quant_config = {
    "algorithm": "awq",
    "wbits": 4,
    "group_size": 128
}

# Run compression
compress_model(
    model=model_path,
    output_dir=output_path,
    quantization_config=quant_config,
    tokenizer=tokenizer
)

print("✅ Quantization complete")