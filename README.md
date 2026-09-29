# GPT-2 Inference with Julia

I built a GPT-2 inference engine from scratch in Julia.

![(Cherrypicked) demo](https://raw.githubusercontent.com/krswm/asset/main/slope-jl/demo.gif)

I also built:

- [An inference engine for GPT-2 in Rust](https://github.com/krswm/slope-rs)
- [An inference engine for Stable Diffusion in Julia](https://github.com/krswm/diff-jl)
- [An inference engine for Stable Diffusion in Rust](https://github.com/krswm/diff-rs)

## Quickstart

I made this project just for **educational purpose**. Use at your own risk.

It is assumed that you have cURL, Git, and Julia installed on your machine.

**Download a pretrained GPT-2 model from Hugging Face.**

```
curl --progress-bar --location --remote-name --output-dir model --create-dirs 'https://huggingface.co/openai-community/gpt2/resolve/main/{config.json,vocab.json,merges.txt,model.safetensors}'
```

**Clone this repository.**

```
git clone https://github.com/krswm/slope-jl.git
cd slope-jl
```

**Install packages.**

```
julia --project --eval 'using Pkg; Pkg.instantiate()'
```

**Start generating text.**
The GPT-2 model is not for chat conversation, but for text continuation.

```
julia --project src/main.jl ../model 0.75 'Julia is a programming language. It is fun to code in Julia.'
```

The second parameter (`0.75` here) is sampling temperature, which controls the randomness of the generated text.
Set it to `0.0` to make the generated text deterministic.
Increase it to make the generated text more *creative*.

## Supported Models

This program only supports models that are build on the GPT-2 architecture.

This program only supports models that have the following files in the model repository.

- `config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`

I have verified that this program works with the following models.

- [GPT-2](https://huggingface.co/openai-community/gpt2)
- [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)
- [GPT-2 Large](https://huggingface.co/openai-community/gpt2-large)
- [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl)

## Source Files

- [`src/model.jl`](src/model.jl) builds a `struct` holding the parameters of the model.
- [`src/tokenizer.jl`](src/tokenizer.jl) converts your prompt into numbers that the model understands (tokens) with the BPE algorithm.
- [`src/transformer.jl`](src/transformer.jl) is the heart of the GPT-2 inferenece. It receives tokens (your prompt + already generated text) and predicts the next token.
- [`src/main.jl`](src/main.jl) loads files from the GPT-2 repository and generates text.

## Credits

- [GPT-2](https://huggingface.co/openai-community/gpt2) for devising an influental LLM architecture.
- [*GPT in 60 Lines of NumPy*](https://jaykmody.com/blog/gpt-from-scratch/) (a blog post) for teaching me how to implement a GPT-2 inference engine from scratch.
- [*Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (a blog post) for teaching me how to implement a BPE tokenizer from scratch.
- [Julia](https://github.com/JuliaLang/julia) for providing me an amazing programming language.

## Development

This is a hobby project of mine I started from scratch.

- 2026-07-03: I started this project.
- 2026-07-20: I finished implementing an GPT-2 inference engine in Julia.
- 2026-08-04: I finished rewriting the transformer to use KV-cache.

I used open source LLM inference engines (Ollama, etc.) and open weight LLM models (TinyLlama, GPT-2, etc.) only for the purpose to observe their behavior as LLM architecture.
Except for this, I did **not** use generative AI for this project at all.
