import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Testing MAPPO Critic Backbone on {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for Mac and Force CPU
    is_mac = not torch.cuda.is_available() and torch.backends.mps.is_available()
    device = "cpu" if is_mac else "auto"

    if is_mac:
        print("[*] Detected macOS. Forcing CPU for Critic verification...")

    # Load the model with a built-in Value Head (num_labels=1)
    # Note: It will print a warning about "uninitialized weights". This is 100% normal
    # because we are creating a brand new Value Head from scratch!
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=1, 
        device_map=device,
        torch_dtype=torch.float32 if is_mac else torch.float16
    )

    inputs = tokenizer("Testing the critic network.", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        values = outputs.logits # The output is our Critic Value (batch, seq_len, 1)

    print(f"[SUCCESS] Input Shape: {inputs['input_ids'].shape}")
    print(f"[SUCCESS] Critic Value Shape: {values.shape}")
    print("[SUCCESS] The Value Head is successfully attached!")
    print("[SUCCESS] Ready for MAPPO integration in Review 2!")

if __name__ == "__main__":
    main()