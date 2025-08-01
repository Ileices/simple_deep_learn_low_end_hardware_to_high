import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


def load_jsonl(path: Path):
    texts = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj.get('prompt') or obj.get('content') or ''
            response = obj.get('response') or ''
            texts.append({'text': prompt + '\n' + response})
    return texts


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with LoRA')
    parser.add_argument('--data', type=Path, required=True, help='JSONL dataset path')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1', help='Base model')
    parser.add_argument('--output', type=Path, default=Path('lora-output'), help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to train on, e.g. "cpu"')
    parser.add_argument(
        '--lora-target',
        type=str,
        default='q_proj,v_proj',
        help='Comma separated target modules for LoRA',
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.device == 'auto':
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.to(args.device)

    dataset = load_dataset('json', data_files=str(args.data))['train']
    target_modules = [m.strip() for m in args.lora_target.split(',') if m.strip()]
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules)
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)

    tokenized = dataset.map(tokenize, batched=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        no_cuda=args.device == 'cpu',
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(str(args.output))


if __name__ == '__main__':
    main()
