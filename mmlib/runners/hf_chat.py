#!/usr/bin/env python3
"""Interactive chat session using Hugging Face transformers."""

import argparse
import sys
from pathlib import Path

# Remove runner directory from sys.path to avoid shadowing third-party packages
runners_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if str(Path(p).resolve()) != runners_dir]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    parser = argparse.ArgumentParser(description="Chat with a Hugging Face model")
    parser.add_argument("--model", required=True, help="Model path or HF repo ID")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (
            torch.float16
            if (
                torch.cuda.is_available()
                or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            )
            else torch.float32
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if device_map is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            model = model.to("cpu")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("Model loaded. Type your message (Ctrl+C or 'exit' to quit):\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"User: {user_input}\nAssistant: "

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("\nAssistant: ", end="", flush=True)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": args.max_tokens,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
            }
            if args.temp > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temp
                gen_kwargs["top_p"] = args.top_p
                if args.top_k > 0:
                    gen_kwargs["top_k"] = args.top_k
            else:
                gen_kwargs["do_sample"] = False

            output_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
