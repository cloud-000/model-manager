# mm

A resolver and launcher for models already sitting on your disk.

The Hugging Face cache is *managed* fine — `hf` downloads them, `hf cache ls`
lists them. What nothing does is turn a short name into an exact runnable path
and hand it to the runner that can actually load it. That's all `mm` is.

```
$ mm ls
HANDLE                         ROLE    RUNTIME    QUANT    SIZE    CTX   LAST  NOTE
mlx/gemma-4-12b-it-qat-4bit    vision  mlx        4bit     10.3G   256k  -
mlx/maple-2bit                 chat    deepgrove  2bit     5.0G    128k  1h    ternary MoE
mlx/nanbeige4.2-3b-optiq-4bit  chat    mlx        4bit     3.1G    256k  -
mlx/lfm2.5-2.6b:4bit           chat    mlx        4bit     1.5G    128k  2h    daily driver
gguf/ornith-1.0-9b:Q4_K_M      chat    -          Q4_K_M   5.2G    -     1d
litert/gemma-4-e2b-it-litert-lm chat   -          -        2.4G    -     -

not runnable here (2):
  hf/docling-layout-heron          163.7M  rt_detr_v2 - no tokenizer, not a text model
  ct2/faster-whisper-base          141.0M  CTranslate2 weights - needs faster-whisper

14 runnable, 46.8G on disk
  hub      41.8G  /Users/cloud/.cache/huggingface/hub
  local     5.0G  /Users/cloud/CodeProjects/AI/mlx-lm-deepgrove/maple-2bit-mlx
```

The `RUNTIME` column only appears once you have registered more than one.

## Install

Stdlib-only Python 3, no dependencies and no install step:

```sh
ln -s "$PWD/mm" ~/.local/bin/mm
```

`mm` is a launcher that resolves its own symlink to find the `mmlib/` package
beside it, so the link can live anywhere as long as the two stay together.

## Commands

| | |
|---|---|
| `mm ls [--all] [--names]` | inventory; `--all` adds draft models, `--names` feeds fzf |
| `mm path <handle>` | print the resolved runnable path |
| `mm run <handle>` | interactive chat |
| `mm serve <handle> [--port N]` | HTTP server (default port 8080) |
| `mm set <scope> k=v …` | write settings; scope is a handle or `@defaults`/`@mlx`/`@llama` |
| `mm config <handle>` | show resolved settings and where each came from |
| `mm note <handle> [text]` | why you downloaded it |
| `mm alias <name> <handle>` | pin a short name |
| `mm rm <handle>` | delete a cached repo (asks you to type the repo name) |
| `mm link` | refresh `~/models/<format>/` symlinks |
| `mm root add\|rm\|ls <dir>` | scan a directory outside the hub cache |
| `mm runtime add\|rm\|ls` | register another venv to launch MLX models from |
| `mm runtime flag <rt> k=--flag` | teach a runtime a setting only it understands |

Anything after a bare `--` is forwarded to the runner untouched:

```sh
mm run vibe -- --max-tokens 512 --temp 0.7
```

`--dry-run` on `run`/`serve` prints the command without executing it.

## Handles

Handles read `format/name:variant`, the same shape as a container image:

```
LiquidAI/LFM2.5-2.6B-MLX             → mlx/lfm2.5-2.6b:4bit
LiquidAI/LFM2.5-2.6B-GGUF            → gguf/lfm2.5-2.6b:Q4_K_M
deepreinforce-ai/Ornith-1.0-9B-GGUF  → gguf/ornith-1.0-9b:Q4_K_M
```

The format prefix is not decoration. Quantisation labels are chosen by whoever
uploaded the repo, and nothing stops an MLX repo from having a subdirectory
called `Q4_0` while a GGUF sibling ships a file called `…-Q4_0.gguf`. Without
the namespace those two produce **the same handle**, and `mm run` silently picks
whichever it happened to scan first. With it they are `mlx/weird-3b:Q4_0` and
`gguf/weird-3b:Q4_0`, and they can never be confused.

It also means the answer to "what even is this thing" is the first word you
read, rather than something you infer from a quant string:

| prefix | meaning | runner |
|---|---|---|
| `mlx/` | MLX safetensors — a `quantization` block in `config.json`, or an MLX-named repo | mlx-lm / mlx-vlm |
| `gguf/` | llama.cpp GGUF | llama.cpp |
| `hf/` | plain safetensors, not MLX-quantised | mlx-lm |
| `litert/` | LiteRT `.litertlm` bundle | litert-lm |
| `ct2/` | CTranslate2 (faster-whisper and friends) | none yet |

Because the namespace carries the format, a redundant `-MLX`/`-GGUF` is dropped
from the name itself, and slugs only need to be unique *within* a format. Two
repos colliding inside one namespace fall back to org-qualified names.

Matching is still fuzzy, so you rarely type a prefix: `mm run ornith` works. An
exact handle beats a substring, so `minicpm5-1b` resolves to itself rather than
the longer `minicpm5-1b-claude-opus-fable5-v2-thinking`. Ambiguous input lists
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
`chat`, `vision`, or `draft`.

## What runs what

The format picks the **runner** — the program that gets handed the model. Each
one lives in `mmlib/runners/` and declares the flags it understands next to the
code that builds its argv, so they can't drift apart.

**MLX** (`mlx/`, `hf/`) → `mlx_lm.chat` / `mlx_lm.server` from the model's
*runtime*, which is `~/.venvs/mlx` unless you say otherwise. Models with a
`vision_config` route to `mlx_vlm` instead; force either with `--vlm` / `--lm`.

> `mlx_lm.server` routes each request on its **`model` field**, not on the
> `--model` it was started with — passing an unknown name makes it try to
> download that name from Hugging Face. Use the resolved path:
> `curl … -d "{\"model\": \"$(mm path minicpm5-1b)\", …}"`

**GGUF** (`gguf/`) → delegated to llama.cpp's unified `llama` CLI via `-hf`, e.g.
`llama cli -hf deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M`. This is deliberate:
llama.cpp reads the same Hugging Face cache and already resolves quants and
attaches `mmproj` vision sidecars on its own. `mm` contributes only the short
name and inherits the rest. If there's no `repo:quant` to hand over, it falls
back to `-m` with an explicit `--mmproj`.

**LiteRT** (`litert/`) → `litert-lm run <path>`, where the model is a positional
argument rather than a flag. `mm serve` on one of these is a special case worth
knowing about — see below.

**Nothing** → listed under "not runnable here" with the reason, so a model you
downloaded once and can't load stops being a mystery. That includes a runner
that isn't installed: `litert/gemma-4-e2b-it-litert-lm` is runnable the moment
`litert-lm` is on your PATH and says `litert-lm not on PATH` when it isn't.

That last list also catches architectures nothing installed can build. mlx-lm
resolves a model by importing `mlx_lm.models.<model_type>`, so `mm` reads the
module names out of each runtime and checks against them before promising you
a launch — `mlx_vlm.models.*` for vision models, which is a different set
entirely:

```
mlx/something-new-4bit  6.1G  qwen4_moe - no installed mlx-lm implements this architecture
```

A checkpoint that carries its own architecture file (`model_file` in
`config.json`) is exempt: it brings the code with it and any runtime can load it.

## Models outside the cache

`hf download --local-dir <path>` puts the weights wherever you asked and leaves
only a `refs/` stub in the hub cache, which is how a project that ships a model
alongside its own runtime tends to distribute it. `mm` says so rather than
silently omitting the repo:

```
hf/maple-2bit  -  no snapshot in cache - downloaded with --local-dir? (try: mm root add <dir>)
```

Register the directory and it becomes an ordinary handle:

```sh
mm root add ~/CodeProjects/AI/mlx-lm-deepgrove
```

A root is scanned one level deep, so pointing at a project directory finds the
model inside it without walking the whole tree. The handle grammar doesn't
change — where the bytes live is not what a `mlx/` or `gguf/` prefix is for — so
the result is `mlx/maple-2bit` next to everything else, and the cache stub it
supersedes stops being listed.

The org is recovered rather than guessed: the commit sha recorded in the local
directory is matched against the hub's `refs/main`, so the model is still
`deepgrove/maple-2bit-mlx` for the purposes of `mm rm` and collision fallback.

### Serving a LiteRT model

`litert-lm serve` takes no model argument at all — it serves everything in its
own store at `~/.litert-lm/models` and routes each request by name, the way
`mlx_lm.server` routes on the request's `model` field. `mm serve` says so rather
than pretending the handle was used:

```
$ mm serve litert/gemma
mm: warning: litert-lm serve ignores litert/gemma-4-e2b-it-litert-lm and serves
    its own store (litert-lm list); requests route by model name
mm: warning: temp is not supported by litert-lm.serve, ignored
$ litert-lm serve --port 8080
```

Both warnings are load-bearing. `litert-lm serve` accepts `--host`, `--port` and
`--verbose`, and nothing else — every sampling setting genuinely goes nowhere.
Use `litert-lm import` to put a model in the store first.

### Adding a runner

One file in `mmlib/runners/`, about thirty lines:

```python
NAME    = "litert"
FORMATS = ("litert",)
FLAGS   = {"litert-lm.run": {"temp": "--temperature", …},
           "litert-lm.serve": {}}

def available():  return None if shutil.which("litert-lm") else "litert-lm not on PATH"
def tool_for(entry, serve, args):  return f"litert-lm.{'serve' if serve else 'run'}"
def command(entry, serve, args, runtime):  …
```

`FLAG_MAP` assembles itself from these, `@litert` becomes a settings scope, and
`mm ls` starts reporting the model's real status. Adding one is not a config
change, and deliberately so: the differences between runners are structural
rather than parametric. mlx-lm takes `--model <path>`, llama.cpp takes
`-hf <repo>:<quant>` against its own cache, and `litert-lm serve` takes no model
whatsoever. A template language in JSON that could express all three would be a
worse thing to write than three small Python files.

## Runtimes

A *runner* is a different program, chosen by format. A **runtime** is one of
several interchangeable installations of the *same* runner, chosen per model —
so far only MLX has more than one, which is why it's the only runner with a
registry.

There is more than one mlx-lm in the world. A fork adds architectures and flags
upstream doesn't have, and a checkpoint built against one may need it. So the
venv is not a global constant — it is a named runtime, and a model can be
pinned to one:

```sh
mm runtime add deepgrove ~/CodeProjects/AI/mlx-lm-deepgrove/.venv
mm set mlx/maple-2bit runtime=deepgrove
```

```
$ mm runtime
deepgrove    /Users/cloud/CodeProjects/AI/mlx-lm-deepgrove/.venv  (ok)
               110 architectures, 2 not in mlx: maple, mellum
               flash_head -> --flash-head (bool, mlx_lm.chat, mlx_lm.server)
mlx          /Users/cloud/.venvs/mlx  (ok)
               108 architectures
```

Selection is explicit, not inferred — `runtime=` is an ordinary setting, so
`mm config` tells you which venv a model will launch from and why. The
architecture scan is only a backstop: if the default runtime can't build a
model and another one can, `mm` uses the one that can.

Editable installs (`pip install -e`, which is how you'd have a fork checked out)
put nothing in `site-packages` but a finder shim, so `mm` reads the shim to find
the source tree. Nothing here starts a Python interpreter; it is all filesystem.

### Flags a fork adds

A runtime declares the settings only it understands, so `mm set` can still
reject a typo without `mm` carrying a table of every downstream mlx-lm:

```sh
mm runtime flag deepgrove flash_head=--flash-head --bool --negatable \
  --tools mlx_lm.chat,mlx_lm.server
mm set mlx/maple-2bit flash_head=true
```

`--bool` means the flag takes no value; `--negatable` means setting it `false`
emits `--no-flash-head` rather than nothing. Ask for it on a runtime that has
never heard of it and you get the same warning as any other unsupported key:

```
$ mm run vibe
mm: warning: flash_head is not supported by mlx_lm.chat (runtime mlx), ignored
```

## Settings

Sampling settings layer in a cascade. Lowest to highest:

**`defaults` → `runner` → `model` → explicit `--` passthrough.** Last writer wins.

```sh
mm set @defaults temp=0.7 top_p=0.95
mm set @mlx      min_p=0.05 top_k=40
mm set @llama    repeat_penalty=1.1
mm set vibe      temp=0.6 max_tokens=4096
```

`mm config <handle>` shows the merged result and which layer each value came
from, which is usually the question you actually have:

```
$ mm config vibe
mlx/vibethinker-3b-8bit  ->  mlx_lm.chat
  max_tokens  4096  model
  min_p       0.05  runner:mlx  ignored - mlx_lm.chat has no min_p
  temp        0.6   model
  top_p       0.95  defaults
```

Add `--serve` to resolve against the server tool instead, which supports more.

### Why a normalised vocabulary

The runners disagree about names, and about what exists at all:

| | `mlx_lm.chat` | `mlx_lm.server` | `mlx_vlm.chat` | `llama` |
|---|---|---|---|---|
| `temp` | `--temp` | `--temp` | `--temperature` | `--temp` |
| `top_p` | `--top-p` | `--top-p` | — | `--top-p` |
| `top_k` | — | `--top-k` | — | `--top-k` |
| `min_p` | — | `--min-p` | — | `--min-p` |
| `max_tokens` | `--max-tokens` | `--max-tokens` | `--max-tokens` | `-n` |
| `seed` | `--seed` | — | — | `--seed` |
| `repeat_penalty` | — | — | — | `--repeat-penalty` |

So you write `temp` once and it becomes `--temp` or `--temperature` as needed.
A key the chosen tool can't express is **reported on stderr and dropped**, never
silently ignored:

```
$ mm run vibe
mm: warning: top_k is not supported by mlx_lm.chat, ignored (try mm serve)
```

That warning is load-bearing: `top_k` genuinely does not exist in the
`mlx_lm.chat` REPL. For top-k or min-p on an MLX model, use `mm serve` and a
client, or `mlx_lm.generate` for one-shots.

Note also that **mlx-lm defaults to `temp 0.0`** — pure greedy. llama.cpp already
defaults to `temp 0.8, top_k 40, top_p 0.95`, so `@defaults temp=…` mainly
matters for MLX.

### Escape hatch

Anything outside the vocabulary goes through `--raw`, appended verbatim:

```sh
mm set vibe --raw '--xtc-probability 0.1 --xtc-threshold 0.1'
mm set vibe --raw ''          # drop them
```

A lone flag with no argument needs `--raw=`, because argparse reads a bare
`-…` as the next option unless it contains a space:

```sh
mm set @deepgrove --raw=--trust-remote-code
```

Unlike normal keys, `raw` from every layer accumulates rather than overriding,
since these are extra arguments rather than competing values.

### Overriding at the prompt

`--` still wins over everything. Where it collides with a cascade value, `mm`
drops its own copy rather than passing the flag twice:

```sh
mm run vibe -- --temp 0.9
mm set vibe --clear temp      # or remove it for good
```

## Draft models

A repo whose `model_type` ends in `_assistant` is a speculative-decoding draft
model, not something to launch. It's hidden from `mm ls` and paired with its
base automatically:

```sh
mm serve gem --draft
# → mlx_lm.server --model .../gemma-4-12B-it-qat-4bit \
#                 --draft-model .../gemma-4-12B-it-qat-assistant-4bit
```

`mlx_lm.chat` has no `--draft-model`, so `mm run --draft` tells you to use
`serve` instead rather than failing obscurely.

## The symlink farm

`mm link` maintains short symlinks in `~/models`, so tools that never heard of
`mm` still get readable paths:

```
~/models/mlx/lfm2.5-2.6b-4bit  -> …/models--LiquidAI--LFM2.5-2.6B-MLX/snapshots/58e239…/4bit
~/models/gguf/ornith-1.0-9b-Q4_K_M.gguf -> …/ornith-1.0-9b-Q4_K_M.gguf
```

Links are grouped into a subdirectory per format, mirroring the handles.

It only removes links that point into the hub cache, so anything else you keep
in `~/models` is left alone. Re-run it after downloading something new.

## Config

`~/.config/mm/config.json` holds aliases, notes, last-used timestamps, settings,
and the roots and runtimes you have registered. Nothing *about the models* is
cached — the scan reads the filesystem every time, which costs milliseconds and
can never go stale.

Overridable via `HF_HUB_CACHE`, `MM_MLX_VENV`, and `MM_LINK_DIR`. `MM_MLX_VENV`
sets the default `mlx` runtime, which always exists and can't be removed.

## Worked example: Maple on a forked mlx-lm

[deepgrove-ai/mlx-lm-deepgrove][dg] is a fork that ships Maple, a 2-bit ternary
MoE, and downloads the checkpoint into the repo with `--local-dir`. It exercises
every part of the above at once:

[dg]: https://github.com/deepgrove-ai/mlx-lm-deepgrove

```sh
mm root add ~/CodeProjects/AI/mlx-lm-deepgrove
mm runtime add deepgrove ~/CodeProjects/AI/mlx-lm-deepgrove/.venv
mm runtime flag deepgrove flash_head=--flash-head --bool --negatable \
  --tools mlx_lm.chat,mlx_lm.server

mm set @deepgrove --raw=--trust-remote-code
mm set mlx/maple-2bit runtime=deepgrove flash_head=true \
  temp=1.0 top_p=0.95 max_tokens=-1
mm alias maple mlx/maple-2bit
```

```
$ mm run maple --dry-run
$ …/mlx-lm-deepgrove/.venv/bin/mlx_lm.chat --model …/maple-2bit-mlx \
    --flash-head --max-tokens -1 --temp 1.0 --top-p 0.95 --trust-remote-code
```

Worth knowing: the checkpoint bundles its own `maple.py` and names it in
`config.json`, and mlx-lm executes that file in preference to anything it ships,
so stock `~/.venvs/mlx` runs this model perfectly well. The fork buys one thing
— the `--flash-head` plumbing — which measured 276 tok/s against 206 on the same
prompt here, at 0.6 GB less peak memory. Drop the `runtime=` pin and it still
runs, just slower. That is the sort of question `mm config` exists to answer.

## Tests

```sh
python3 test_mm.py
```

Stdlib `unittest`, no fixtures on disk and nothing downloaded — every test
builds a fake cache or a fake root out of empty files, because what is worth
testing here is layout and naming rather than weights.

## Not included

Downloading (`hf download` and `llama download` both work), benchmarking, and
process supervision. Tracking running servers and their ports is the obvious
next addition, but it needs real state, which v1 deliberately avoids.
