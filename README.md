# My Memo

I want to implement the KV cache.
Then I wondered why it's not QKV cache and I researched and I found an interesting Stack Exchange post.

<https://ai.stackexchange.com/questions/48185/why-not-cache-the-q-query-matrix>

I don't have to get all Q every time at first place!

Yes I already employed a similar tecknique on the Projection part of the transformer
but the similar thing can be applied more broadly!

I need this optimization before going to kv cache implementation.

I verified in kv-cache.jl and confirmed all QKV is same except for the last row.

# GPT-2 Inference with Julia

I built a GPT-2 inference engine from scratch in Julia.

![Demo](asset/demo.gif)

I also built [a Rust counterpart](https://github.com/krswm/slope-rs).

## Quickstart

I made this project just for educational purpose. Use at your own risk.

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
julia --project --handle-signals=no main.jl ../model 'Natural language processing is a branch of computer science. We study'
```

Press `Control+C` to stop generating text.

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

## Credits

- [GPT-2](https://huggingface.co/openai-community/gpt2) for devising an influental LLM architecture.
- [*GPT in 60 Lines of NumPy*](https://jaykmody.com/blog/gpt-from-scratch/) (a blog post) for teaching me how to implement a GPT-2 inference engine from scratch.
- [*Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (a blog post) for teaching me how to implement a BPE tokenizer from scratch.
- [Julia](https://github.com/JuliaLang/julia) for providing me an amazing programming language.

## Development

This is a hobby project of mine I started from scratch.

I started this project on 2026-07-03 and finished my first implementation on 2026-07-20.

I used open source LLM inference engines (Ollama, etc.) and open source LLM models (TinyLlama, GPT-2, etc.) only for the purpose to observe their behavior as LLM architecture.
Except for that, I did **not** use generative AI for this project at all.
