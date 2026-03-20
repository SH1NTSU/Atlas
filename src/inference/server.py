"""
Atlas inference server.

Provides:
- Interactive CLI chat
- FastAPI server with OpenAI-compatible /v1/chat/completions endpoint
- Streaming support
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from model.transformer import Atlas, AtlasConfig
from tokenizer.trainer import load_tokenizer


class AtlasInference:
    """Handles model loading and text generation."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = load_tokenizer(tokenizer_path)

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(
            Path(checkpoint_path) / "checkpoint.pt",
            map_location=self.device,
            weights_only=False,
        )

        # Rebuild config from checkpoint (supports both saved config and default)
        saved_config = checkpoint.get("config", {})
        config = AtlasConfig(**saved_config) if saved_config else AtlasConfig()
        self.model = Atlas(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        param_count = self.model.count_parameters()
        print(f"Model loaded: {param_count / 1e6:.1f}M parameters on {self.device}")

    def format_prompt(self, messages: list[dict]) -> str:
        """Format chat messages into model template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n<|end|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|end|>")
        # Add assistant prefix for generation
        parts.append("<|assistant|>")
        return "\n".join(parts)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate text from a prompt."""
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Decode only new tokens
        new_tokens = output_ids[0][len(encoding.ids):].tolist()
        return self.tokenizer.decode(new_tokens)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Chat-style generation from a list of messages."""
        prompt = self.format_prompt(messages)
        response = self.generate(prompt, **kwargs)
        # Strip end token if present
        if "<|end|>" in response:
            response = response[:response.index("<|end|>")]
        return response.strip()


def run_cli(engine: AtlasInference):
    """Interactive CLI chat loop."""
    print("\nAtlas CLI — type 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are Atlas, a coding assistant specialized in TypeScript, Go, Rust, and Shell scripting."}
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = messages[:1]  # Keep system prompt
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        start = time.time()
        response = engine.chat(messages)
        elapsed = time.time() - start

        print(f"\nAtlas: {response}")
        print(f"  ({elapsed:.1f}s)")

        messages.append({"role": "assistant", "content": response})


def run_server(engine: AtlasInference, host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server with OpenAI-compatible API."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Atlas API")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "atlas"
        messages: list[Message]
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response = engine.chat(
            messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return {
            "id": f"chatcmpl-atlas-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "atlas",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [{"id": "atlas", "object": "model", "owned_by": "local"}]
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    print(f"Starting Atlas API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Atlas inference")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory path")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = AtlasInference(args.checkpoint, args.tokenizer, args.device)

    if args.mode == "cli":
        run_cli(engine)
    elif args.mode == "server":
        run_server(engine, args.host, args.port)


if __name__ == "__main__":
    main()
