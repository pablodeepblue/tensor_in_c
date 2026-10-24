# Tensor in C

Learn how neural networks and transformers actually work by building them from scratch, first in plain Python, then in C, with **zero external libraries**.

This is both a personal exercise and a tutorial. Each stage re implements the previous one at a lower level, so you can see exactly what frameworks like PyTorch are doing under the hood.

## Why this exists

Most ML tutorials start with `import torch`. This repo starts with nothing: no NumPy, no frameworks, only the language itself. The goal is an easy-to-follow path from "what is a neuron?" to "how does a transformer work?", with every step small enough to read in one sitting.

## The learning path

| Stage | What you build | Language | Status |
|-------|----------------|----------|--------|
| 1 | A neural net from scratch | Python | [x] |
| 2 | Batching | Python | [] |
| 3 | Attention | Python | [] |
| 4 | A transformer | Python | [] |
| 5 | Port the tensor library to C and call it from Python (ctypes) | Python + C | [] |
| 5 | TBD | NA | [] |

## Getting started

**Prerequisites:** Python (see `.python-version`) and [uv](https://docs.astral.sh/uv/). A C compiler (gcc or clang) is needed from stage 5.

```bash
git clone https://github.com/pablodeepblue/tensor_in_c.git
cd tensor_in_c
uv sync
```

<!-- Add: how to run the first example, e.g. `uv run python src/tensor_in_c/<file>.py` -->

## Project structure
