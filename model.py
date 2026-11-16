"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth — assembled scaffold.
This updates live as you solve each step.
"""
from unsloth import FastLanguageModel
import numpy as np
import bitsandbytes as bnb
from datasets import Dataset
import torch
from transformers import TrainingArguments


# ── Step 001  load_base_model_and_tokenizer ──
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # auto-pick float16 / bfloat16 for the GPU
        load_in_4bit=True,   # load 4-bit weights so the model fits on a small GPU
    )
    return model, tokenizer


def count_total_parameters(model):
      """Return the total number of parameters in `model` as a Python int."""
      return int(sum(p.numel() for p in model.parameters))

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    return any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())

def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    from unsloth import FastLanguageModel

    if target_modules is None:
        target_modules = get_lora_target_modules()   # reuse step 5

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,                           # no dropout: fastest path in Unsloth
        bias="none",                              # don't train bias terms
        use_gradient_checkpointing="unsloth",     # saves GPU memory during training
        random_state=3407,                        # reproducible adapter initialization
    )
    return model

def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def trainable_fraction(trainable_count, total_count):
    """Return the fraction of parameters that are trainable."""
    return float(trainable_count) / float(total_count)


def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "Say hello in one short sentence.",
            "response": "Hello! Nice to meet you.",
        },
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "Translate 'thank you' into German.",
            "response": "'Thank you' in German is 'Danke'.",
        },
        {
            "instruction": "Give me one tip for staying focused while studying.",
            "response": "Put your phone in another room so it can't distract you.",
        },
        {
            "instruction": "What is 7 times 8?",
            "response": "7 times 8 is 56.",
        },
    ]

def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(ex) for ex in examples]

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": list(texts)})

def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    encoded = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)
    return list(encoded["input_ids"])

def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return int(len(input_ids))

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
  
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        optim="adamw_8bit",
        report_to="none",
    )

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    import inspect
    from trl import SFTTrainer, SFTConfig

    # 1) Turn the TrainingArguments from step 15 into an SFTConfig
    #    and add the SFT-specific settings.
    cfg = training_args.to_dict()
    cfg["hub_token"] = training_args.hub_token   # to_dict() masks this, so copy the real value
    cfg["dataset_text_field"] = "text"           # train on the 'text' column
    cfg["packing"] = False                       # one example = one training row

    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in cfg_params:           # older TRL name
        cfg["max_seq_length"] = max_seq_length
    else:                                        # newer TRL name
        cfg["max_length"] = max_seq_length
    cfg = {k: v for k, v in cfg.items() if k in cfg_params}
    sft_config = SFTConfig(**cfg)

    # 2) Build the trainer; the tokenizer argument name also depends on the version.
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kw = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        **{tok_kw: tokenizer},
    )

def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    train_output = trainer.train()

    # Loss of the last logged step (logging_steps=1 means every step is logged)
    step_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    if step_losses:
        return float(step_losses[-1])

    # Fallback: the average loss over the whole run
    return float(train_output.training_loss)


def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""

    FastLanguageModel.for_inference(model)
    model.eval()          # make sure training mode is off (model.training == False)
    return model

def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,              # return a string, not token ids
        add_generation_prompt=True,  # end with the assistant-turn opener
    )

def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    import torch

    # 1) Tokenize the prompt and move it to the same device as the model (the GPU)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    # 2) Greedy decoding: always pick the most likely next token
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 3) Keep only the new tokens (cut off the prompt) and decode them
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Scaffold (runner) ──
"""Scaffold: LoRA fine-tune a tiny 4-bit Qwen2.5 chat model with Unsloth."""
import torch

from solution import (
    load_base_model_and_tokenizer,
    count_total_parameters,
    is_model_4bit_quantized,
    ensure_pad_token,
    get_lora_target_modules,
    attach_lora_adapters,
    count_trainable_parameters,
    trainable_fraction,
    build_instruction_examples,
    format_instruction_example,
    format_all_examples,
    build_text_dataset,
    tokenize_text,
    count_tokens,
    build_training_arguments,
    build_sft_trainer,
    run_sft_training,
    switch_to_inference_mode,
    build_chat_prompt,
    generate_reply,
)


def main():
    torch.manual_seed(0)

    # 1) Load 4-bit base model + tokenizer.
    model, tokenizer = load_base_model_and_tokenizer(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=256,
    )
    total_params = count_total_parameters(model)
    quantized = is_model_4bit_quantized(model)
    print(f"[base] total_params={total_params:,} 4bit={quantized}")

    # 2) Make sure tokenizer has a pad token.
    tokenizer = ensure_pad_token(tokenizer)
    print(f"[tokenizer] pad_token={tokenizer.pad_token!r}")

    # 3) Attach LoRA adapters to attention projections.
    target_modules = get_lora_target_modules()
    print(f"[lora] target_modules={target_modules}")
    model = attach_lora_adapters(
        model, r=8, lora_alpha=16, target_modules=target_modules
    )
    trainable = count_trainable_parameters(model)
    frac = trainable_fraction(trainable, total_params)
    print(f"[lora] trainable={trainable:,} fraction={frac:.6f}")

    # 4) Build a tiny in-code instruction dataset.
    examples = build_instruction_examples()
    print(f"[data] num_examples={len(examples)}")
    first_text = format_instruction_example(examples[0])
    print(f"[data] formatted[0]={first_text!r}")
    texts = format_all_examples(examples)
    dataset = build_text_dataset(texts)
    print(f"[data] dataset_columns={dataset.column_names} size={len(dataset)}")

    # 5) Peek at tokenization of one example.
    ids = tokenize_text(tokenizer, texts[0])
    print(f"[data] tokens[0]={count_tokens(ids)}")

    # 6) Featherweight SFT: 5 steps, batch size 1.
    training_args = build_training_arguments(
        output_dir="./sft_out", max_steps=5, learning_rate=2e-4
    )
    trainer = build_sft_trainer(
        model, tokenizer, dataset, training_args, max_seq_length=256
    )
    final_loss = run_sft_training(trainer)
    print(f"[train] final_loss={final_loss:.4f}")

    # 7) Switch to fast inference and generate a reply.
    switch_to_inference_mode(model)
    prompt = build_chat_prompt(tokenizer, "Say hello in one short sentence.")
    reply = generate_reply(model, tokenizer, prompt, max_new_tokens=32)
    print(f"[gen] reply={reply!r}")

    passed = (
        total_params > 0
        and trainable > 0
        and trainable < total_params
        and 0.0 < frac < 0.1
        and isinstance(reply, str)
        and len(reply) > 0
        and final_loss == final_loss  # finite (not NaN)
    )
    print({"passed": bool(passed)})
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
