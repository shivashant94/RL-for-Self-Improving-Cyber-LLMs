import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    print(f"[*] Starting Infrastructure SFT Test on {args.model_name}")

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Check for Mac and Force CPU
    is_mac = not torch.cuda.is_available() and torch.backends.mps.is_available()
    
    device = "cpu" if is_mac else "auto"
    if is_mac:
        print("[*] Detected macOS. Forcing CPU to bypass Apple Silicon gradient bugs...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map=device,
        torch_dtype=torch.float32 # Forces standard 32-bit math
    )

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # 4. Load Dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,           
        per_device_train_batch_size=1,
        report_to="none",
        use_cpu=is_mac # Explicitly tells the Trainer to avoid the Mac GPU
    )

    # 6. Train 
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer 
    )

    trainer.train()
    print("[SUCCESS] Infrastructure test passed! Model saved.")

if __name__ == "__main__":
    main()