# mm

Local LLM inventory, configuration, and runner.

`mm` is for the machine where you actually run inference. It reads what is
already on disk, classifies checkpoints by format and role, translates your
settings into whatever flags the underlying runtime actually expects, and execs
into it.

- **Zero dependencies of its own.** Python standard library only.
- **Reads what's there.** Hugging Face cache, llama.cpp cache, local weights
  directories, or any directory tree you tell it to scan.
- **Run or serve by handle or path.** Point `mm run` or `mm serve` at a model handle,
  an HF repo ID (`hf/org/name`), or any local folder on disk (`--path /path/to/model`).
- **Cross-platform.** Automatically routes to Apple Silicon MLX on macOS and
  universal PyTorch `transformers` on Linux and Windows.
- **Knows your runtimes.** If you have multiple MLX virtualenvs (e.g. stock
  `mlx-lm` plus a fork like deepgrove for a newer architecture), `mm` matches
  each model to a venv that can actually build it, and runs the command from
  there.
- **Settings cascade with per-tool flag translation.** Set `temp=0.7` once and
  it reaches `mlx_lm` as `--temp 0.7`, `mlx_vlm` as `--temperature 0.7`, and
  `llama.cpp` as `--temp 0.7`. Flag names never collide because they translate
  at launch time.

---

## Install

Clone the repo anywhere and put `mm` on your `PATH`:

```sh
git clone https://github.com/cloud-000/model-manager.git
mkdir -p ~/.local/bin
ln -s "$PWD/model-manager/mm" ~/.local/bin/mm
```

`mm` is a launcher that resolves its own symlink to find the `mmlib/` package
beside it, so the link can live anywhere as long as the two stay together.

## Commands

| | |
|---|---|
| `mm ls [--all] [--names]` | inventory; `--all` adds draft models, `--names` feeds fzf |
| `mm path <handle\|path> [--path P]` | print the resolved runnable path |
| `mm run <handle\|path> [--path P] [-- RUNNER_ARGS…]` | interactive chat |
| `mm serve <handle\|path> [--path P] [--port N] [-- RUNNER_ARGS…]` | HTTP server (default port 8080) |
| `mm set <scope> k=v …` | write settings; scope is a handle or `@defaults`/`@mlx`/`@llama`/`@transformers` |
| `mm config <handle>` | show resolved settings and where each came from |
| `mm note <handle> [text]` | why you downloaded it |
| `mm alias <name> <handle>` | pin a short name |
| `mm rm <handle>` | delete a cached repo (asks you to type the repo name) |
| `mm link` | refresh `~/models/<format>/` symlinks |
| `mm root add\|rm\|ls <dir>` | scan a directory outside the hub cache |
| `mm runtime add\|rm\|ls` | register another venv to launch MLX models from |
| `mm runtime flag <rt> k=--flag` | teach a runtime a setting only it understands |

### Passing runner flags

Anything after a bare `--` is forwarded to the selected runner untouched. This
is how to use a runner option that `mm` does not define, including options added
by a newer version or a custom fork:

```sh
mm run minicpm5-1b -- --max-tokens 256
```

---

## How models are named

A handle is `format/name:variant`:

```
mlx/lfm2.5-2.6b:4bit
gguf/lfm2.5-2.6b:Q4_K_M
hf/lfm2.5-350m
litert/gemma-3-1b-it:cpu-int4
```

- **Format prefix is required.** `lfm2.5-2.6b` is not a model; it is an
  architecture that might exist in two formats on the same machine.
- **Variant is omitted** when the repo has only one: `mlx/minicpm5-1b`,
  `litert/gemma-3-1b-it`.
- **Slugs drop the redundant format.** `LFM2.5-2.6B-MLX` becomes `lfm2.5-2.6b`,
  not `lfm2.5-2.6b-mlx`.
- **Slugs add the org only on collision.** If you have both `LiquidAI/LFM2.5-2.6B-MLX`
  and someone else's fork, they become `mlx/liquidai-lfm2.5-2.6b` and
  `mlx/other-lfm2.5-2.6b`.

When a command takes a model argument, you can pass:
1. **The full handle:** `mlx/minicpm5-1b`
2. **A direct folder or weights path:** `/path/to/model-folder` or `--path ./my-model`
3. **A remote Hugging Face repo ID:** `hf/LiquidAI/LFM2.5-350M`
4. **An alias:** `mm alias minicpm mlx/minicpm5-1b`
5. **Any substring that matches unambiguously:** `mini` finds `mlx/minicpm5-1b`.

If a substring matches more than one model and they aren't all non-runnable, `mm` lists
the candidates instead of guessing:

```
$ mm path lfm2.5-2.6b
mm: 'lfm2.5-2.6b' is ambiguous:
      gguf/lfm2.5-2.6b:Q4_K_M
      mlx/lfm2.5-2.6b:4bit
```

Add the prefix to settle it (`mm run mlx/lfm2.5-2.6b`), or pin a shortcut with
`mm alias lfm mlx/lfm2.5-2.6b`.

`ROLE` in `mm ls` is separate from format, because it's an orthogonal question:
`chat`, `vision`, or `draft``.

## What runs what

The format picks the **runner** — the program that gets handed the model. Each
one lives in `mmlib/runners/` and declares the flags it understands next to the
code that builds its argv, so they can't drift apart.

**MLX** (`mlx/`, `hf/` on macOS) → `mlx_lm.chat` / `mlx_lm.server` from the model's
*runtime*, which is `~/.venvs/mlx` unless you say otherwise. Models with a
`vision_config` route to `mlx_vlm` instead; force either with `--vlm` / `--lm`.

> `mlx_lm.server` routes each request on its **`model` field**, not on the
> `--model` it was started with — passing an unknown name makes it try to
> download that name from Hugging Face. Use the resolved path:
> `curl … -d "{\"model\": \"$(mm path minicpm5-1b)\", …}"`

**Transformers** (`hf/` on Linux/Windows or fallback) → Universal Hugging Face
runner backed by `~/.venvs/transformers` (or `$MM_TRANSFORMERS_VENV`, or any
active virtualenv). Runs interactive chat and serves an OpenAI-compatible HTTP
server (`/v1/chat/completions` with streaming SSE chunks and `/v1/models`).

**GGUF** (`gguf/`) → delegated to llama.cpp's unified `llama` CLI via `-hf`, e.g.
`llama cli -hf deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M`. This is deliberate:
llama.cpp reads the same Hugging Face cache and already resolves quants and
attaches `mmproj` vision sidecars on its own. `mm` contributes only the short
name and inherits the rest. If there's no `repo:quant` to hand over, it falls
back to `-m` with an explicit `--mmproj`.

**LiteRT** (`litert/`) → `litert-lm run <path>`, where the model is a positional
argument rather than a flag. `mm serve` on one of these is a special case worth
knowing about — see below.

---

## Direct Folder Paths and On-Demand Repos

You can point `mm run` and `mm serve` directly at any directory or weights file:

```sh
# Run or serve an arbitrary directory directly
mm run /path/to/my-weights
mm serve --path /path/to/my-weights --port 8080

# Run or serve a Hugging Face repo ID on demand (downloads automatically)
mm run hf/LiquidAI/LFM2.5-350M
mm serve hf/LiquidAI/LFM2.5-350M --port 8080
```

---

## The settings cascade

`mm` has three layers of settings:

```
defaults  →  runner / runtime  →  model
(global)     (@mlx, @llama,       (mlx/minicpm5-1b)
              @transformers,
              @deepgrove)
```

Settings are written with `mm set <scope> key=val`:

```sh
# Global defaults
mm set @defaults temp=0.7 top_p=0.95

# Per-runner or per-runtime defaults
mm set @mlx      temp=0.8
mm set @llama    repeat_penalty=1.1
mm set @transformers temp=0.7

# Per-model override
mm set mlx/minicpm5-1b temp=0.2 max_tokens=1024
```

To see how `mm` resolves settings for a model and which layer each value came
from:

```sh
mm config mlx/minicpm5-1b
```

---

## Running on Linux or Windows

On Linux or Windows, `mm` automatically detects that MLX is unavailable and routes
standard Hugging Face models (`hf/`) to the universal `transformers` runner.
Ensure you have a Python virtualenv with `torch` and `transformers`:

```sh
python3 -m venv ~/.venvs/transformers
~/.venvs/transformers/bin/pip install torch transformers
```

You can now run and serve any Hugging Face model or folder directly:

```sh
mm run hf/LiquidAI/LFM2.5-350M
mm serve hf/LiquidAI/LFM2.5-350M --port 8080
```
