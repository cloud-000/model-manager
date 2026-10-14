#!/usr/bin/env python3
"""OpenAI-compatible HTTP server using Hugging Face transformers."""

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = None
TOKENIZER = None
MODEL_ID = "default"
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 512


class OpenAIChatHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            return

        if self.path == "/v1/models":
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        self.send_response(404)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}}).encode("utf-8")
            )
            return

        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        temp = float(body.get("temperature", DEFAULT_TEMP))
        top_p = float(body.get("top_p", DEFAULT_TOP_P))
        top_k = int(body.get("top_k", 50))
        max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))

        try:
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant: "

        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "pad_token_id": TOKENIZER.eos_token_id or TOKENIZER.pad_token_id,
        }
        if temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False

        if stream:
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True)
            gen_kwargs["streamer"] = streamer

            thread = threading.Thread(target=MODEL.generate, kwargs=gen_kwargs)
            thread.start()

            # First chunk: role
            first_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

            try:
                for chunk_text in streamer:
                    if not chunk_text:
                        continue
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                thread.join()

            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            try:
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            with torch.no_grad():
                output_ids = MODEL.generate(**gen_kwargs)
            new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
            reply = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(inputs.input_ids.shape[1]),
                    "completion_tokens": int(len(new_tokens)),
                    "total_tokens": int(inputs.input_ids.shape[1] + len(new_tokens)),
                },
            }
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))


def main():
    global MODEL, TOKENIZER, MODEL_ID, DEFAULT_TEMP, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS

    parser = argparse.ArgumentParser(description="OpenAI-compatible server for HF models")
    parser.add_argument("--model", required=True, help="Model path or HF repo ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--temp", type=float, default=0.7, help="Default temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Default top-p")
    parser.add_argument("--max-tokens", type=int, default=512, help="Default max tokens")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    args, _ = parser.parse_known_args()

    MODEL_ID = args.model
    DEFAULT_TEMP = args.temp
    DEFAULT_TOP_P = args.top_p
    DEFAULT_MAX_TOKENS = args.max_tokens

    print(f"Loading {args.model}...", file=sys.stderr)
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

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

    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if device_map is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            MODEL = MODEL.to("mps")
        else:
            MODEL = MODEL.to("cpu")

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, OpenAIChatHandler)
    print(f"Serving OpenAI-compatible API on http://{args.host}:{args.port}/v1", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", file=sys.stderr)
        httpd.server_close()


if __name__ == "__main__":
    main()
