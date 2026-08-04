# GPT-2 inference with Julia
# Copyright (C) 2026  Kurosawa Mutsumi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

using LinearAlgebra
using Statistics

using JSON
using SafeTensors

#### Tokenizer ####

# Tokenize an input with the BPE algorithm.
function tokenize(token_to_id, ranks, input)
    raw_tokens = String[]
    for (i_line, line) ∈ enumerate(split(input, "\n"))
        if i_line ≥ 2
            push!(raw_tokens, "\n")
        end

        for (i_word, word) in enumerate(split(line, " "))
            if i_word == 1 && word ≠ ""
                push!(raw_tokens, word)
            elseif i_word ≥ 2
                push!(raw_tokens, " $word")
            end
        end
    end

    tokens = [encode_unique_encoding(raw_token) for raw_token ∈ raw_tokens]

    # Token IDs
    ids = []
    for token ∈ tokens
        if haskey(token_to_id, token)
            push!(ids, token_to_id[token])
        else
            #### Merge ####

            symbols = string.(collect(token))

            while length(symbols) ≥ 2
                pairs = [
                    (token0, token1) for
                    (token0, token1) ∈ zip(symbols[1:(end-1)], symbols[2:end])
                ]

                best_rank = typemax(Int)
                best_i_pair = 0
                for (i_pair, pair) ∈ enumerate(pairs)
                    if haskey(ranks, pair) && ranks[pair] < best_rank
                        best_rank = ranks[pair]
                        best_i_pair = i_pair
                    end
                end
                if best_i_pair == 0
                    break
                end

                symbols[best_i_pair] *= symbols[best_i_pair+1]
                deleteat!(symbols, best_i_pair + 1)
            end

            for symbol in symbols
                push!(ids, token_to_id[symbol])
            end
        end
    end
    ids
end

# GPT-2 has a unique encoding.
# e.g.: 'Ġ' (U+0120) → 0x20

function encode_unique_encoding(text)
    encoded = [
        if byte ∈ 0x00:0x20
            UInt32(byte + 0x0100)
        elseif byte ∈ 0x21:0x7E
            UInt32(byte)
        elseif byte ∈ 0x7F:0xA0
            UInt32(byte + 0x00A2)
        elseif byte ∈ 0xA1:0xAC
            UInt32(byte)
        elseif byte == 0xAD
            0x00000143
        elseif byte ∈ 0xAE:0xFF
            UInt32(byte)
        end for byte ∈ transcode(UInt8, text)
    ]
    transcode(String, encoded)
end

function decode_unique_encoding(buffer, encoded)
    bytes = [
        if codepoint ∈ 0x0100:0x0120
            UInt8(codepoint - 0x0100)
        elseif codepoint ∈ 0x0021:0x007E
            UInt8(codepoint)
        elseif codepoint ∈ 0x0121:0x0142
            UInt8(codepoint - 0x00A2)
        elseif codepoint ∈ 0x00A1:0x00AC
            UInt8(codepoint)
        elseif codepoint == 0x0143
            0xAD
        elseif codepoint ∈ 0x00AE:0x00FF
            UInt8(codepoint)
        end for codepoint ∈ transcode(UInt32, encoded)
    ]

    bytes = vcat(buffer, bytes)

    # A token may contain only a part of UTF-8 sequence.
    # Decode it incrementally.

    # Unfinished valid UTF-8 sequences:
    #
    # Case A1: 110xxxxx
    #
    # Case B1: 1110xxxx
    # Case B2: 1110xxxx 10xxxxxx
    #
    # Case C1: 11110xxx
    # Case C2: 11110xxx 10xxxxxx
    # Case C3: 11110xxx 10xxxxxx 10xxxxxx

    if length(bytes) ≥ 3 &&
       bytes[end-2] ∈ 0xC0:0xDF &&
       bytes[end-1] ∈ 0x80:0xBF &&
       bytes[end] ∈ 0x80:0xBF
        # Case C3
        decoded = bytes[1:(end-3)]
        buffer = bytes[(end-2):end]
    elseif length(bytes) ≥ 2 && bytes[end-1] ∈ 0xC0:0xEF && bytes[end] ∈ 0x80:0xBF
        # Case B2 and Case C2
        decoded = bytes[1:(end-2)]
        buffer = bytes[(end-1):end]
    elseif length(bytes) ≥ 1 && bytes[end] ∈ 0xC0:0xF7
        # Case A1, Case B1, and Case C1
        decoded = bytes[1:(end-1)]
        buffer = bytes[end:end]
    else
        # No unfinished sequence at the end
        decoded = bytes
        buffer = UInt8[]
    end

    decoded = transcode(String, decoded)
    decoded = string(
        (
            (valid ? char : '�') for
            (char, valid) ∈ zip(decoded, isvalid.(collect(decoded)))
        )...,
    )

    (buffer, decoded)
end

#### Transformer ####

# The paper that introduced layer norm uses uncorrelated variance.
# https://arxiv.org/abs/1607.06450
layer_norm(x, g, t, e) = (x .- mean(x)) ./ √(var(x, corrected = false) + e) .* g .+ t

# The transformer for the GPT-2 architecture.
function transform!(k_caches, v_caches, model, id, pos)
    #### Embedding ####

    # ids are 0-based. Julia is 1-based.
    x = model.wte[:, id+1] + model.wpe[:, pos]

    for (layer, k_cache, v_cache) ∈ zip(model.layers, k_caches, v_caches)
        #### Masked Multi-Head Attention ####

        y = layer_norm(x, layer.g1, layer.t1, model.e)

        y = layer.w11 * y + layer.b11

        q_heads, k_heads, v_heads = (
            Iterators.partition(chunk, model.head_size) for
            chunk ∈ Iterators.partition(y, model.n_embd)
        )

        k_cache[:] = [hcat(cache, head) for (cache, head) in zip(k_cache, k_heads)]
        v_cache[:] = [hcat(cache, head) for (cache, head) in zip(v_cache, v_heads)]

        y = (
            begin
                z = k' * q ./ √Float32(model.head_size)
                z = exp.(z .- maximum(z))
                v * z ./ sum(z)
            end for (q, k, v) ∈ zip(q_heads, k_cache, v_cache)
        )
        y = vcat(y...)

        y = layer.w12 * y + layer.b12

        x += y

        #### Feed Forward ####

        y = layer_norm(x, layer.g2, layer.t2, model.e)

        y = layer.w21 * y + layer.b21

        y = (tanh.((y .^ 3 * 0.044715f0 + y) * √(2.0f0 / π)) .+ 1.0f0) .* y * 0.5f0

        y = layer.w22 * y + layer.b22

        x += y
    end

    #### Projection ####

    x = layer_norm(x, model.gf, model.tf, model.e)

    model.wte' * x
end

#### Main ####

struct Layer
    g1::Array{Float32,1}
    t1::Array{Float32,1}
    w11::Array{Float32,2}
    b11::Array{Float32,1}
    w12::Array{Float32,2}
    b12::Array{Float32,1}
    g2::Array{Float32,1}
    t2::Array{Float32,1}
    w21::Array{Float32,2}
    b21::Array{Float32,1}
    w22::Array{Float32,2}
    b22::Array{Float32,1}
end

struct Model
    n_ctx::Int
    n_embd::Int
    n_head::Int
    n_layer::Int
    vocab_size::Int
    head_size::Int
    e::Float32
    wte::Array{Float32,2}
    wpe::Array{Float32,2}
    layers::Array{Layer}
    gf::Array{Float32,1}
    tf::Array{Float32,1}
end

function get_model(tensors, config)
    n_ctx = config["n_ctx"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    vocab_size = config["vocab_size"]
    head_size = n_embd ÷ n_head
    e = Float32(config["layer_norm_epsilon"])

    function validate_size(tensor, expected)
        if size(tensor) ≠ expected
            error("size of tensor $(size(tensor)) differs from expected $expected")
        end
    end

    wte = permutedims(tensors["wte.weight"])
    validate_size(wte, (n_embd, vocab_size))

    wpe = permutedims(tensors["wpe.weight"])
    validate_size(wpe, (n_embd, n_ctx))

    layers = Layer[]
    for i = 0:(n_layer-1)
        g1 = tensors["h.$i.ln_1.weight"]
        validate_size(g1, (n_embd,))

        t1 = tensors["h.$i.ln_1.bias"]
        validate_size(t1, (n_embd,))

        w11 = permutedims(tensors["h.$i.attn.c_attn.weight"])
        validate_size(w11, (n_embd * 3, n_embd))

        b11 = tensors["h.$i.attn.c_attn.bias"]
        validate_size(b11, (n_embd * 3,))

        w12 = permutedims(tensors["h.$i.attn.c_proj.weight"])
        validate_size(w12, (n_embd, n_embd))

        b12 = tensors["h.$i.attn.c_proj.bias"]
        validate_size(b12, (n_embd,))

        g2 = tensors["h.$i.ln_2.weight"]
        validate_size(g2, (n_embd,))

        t2 = tensors["h.$i.ln_2.bias"]
        validate_size(t2, (n_embd,))

        w21 = permutedims(tensors["h.$i.mlp.c_fc.weight"])
        validate_size(w21, (n_embd * 4, n_embd))

        b21 = tensors["h.$i.mlp.c_fc.bias"]
        validate_size(b21, (n_embd * 4,))

        w22 = permutedims(tensors["h.$i.mlp.c_proj.weight"])
        validate_size(w22, (n_embd, n_embd * 4))

        b22 = tensors["h.$i.mlp.c_proj.bias"]
        validate_size(b22, (n_embd,))

        layer = Layer(g1, t1, w11, b11, w12, b12, g2, t2, w21, b21, w22, b22)
        push!(layers, layer)
    end

    gf = tensors["ln_f.weight"]
    validate_size(gf, (n_embd,))

    tf = tensors["ln_f.bias"]
    validate_size(tf, (n_embd,))

    Model(
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        vocab_size,
        head_size,
        e,
        wte,
        wpe,
        layers,
        gf,
        tf,
    )
end

function main()
    if length(ARGS) ≠ 2
        println("GPT-2 Inference with Julia")
        print("Usage: ")
        printstyled("julia main.jl <path to model repository> <your prompt>", bold = true)
        println()
        println("You may have to enclose 'your prompt' with quotes.")
        exit()
    end

    #### Loading Files ####

    config = JSON.parsefile("$(ARGS[1])/config.json")

    ranks = begin
        ranks = Dict{Tuple{String,String},Int}()
        rank = 0
        for line ∈ readlines("$(ARGS[1])/merges.txt")
            # Skip a comment line.
            if startswith(line, "#")
                continue
            end

            token0, token1 = split(line, " ")
            ranks[(token0, token1)] = rank
            rank += 1
        end
        ranks
    end

    token_to_id = JSON.parsefile("$(ARGS[1])/vocab.json")
    id_to_token = Dict(id => token for (token, id) ∈ token_to_id)

    #### Tokenization ####

    ids = tokenize(token_to_id, ranks, ARGS[2])
    if length(ids) == 0
        println("Your prompt should not be empty.")
        exit()
    elseif length(ids) > config["n_ctx"]
        println("Your prompt exceeds the context length. Try shorter prompt.")
        exit()
    end

    #### Loading Tensors ####

    tensors = load_safetensors("$(ARGS[1])/model.safetensors")

    #### Inference ####

    model = get_model(tensors, config)

    k_caches = [
        [Array{Float32}(undef, model.head_size, 0) for _ = 1:model.n_head] for
        _ = 1:config["n_layer"]
    ]
    v_caches = [
        [Array{Float32}(undef, model.head_size, 0) for _ = 1:model.n_head] for
        _ = 1:config["n_layer"]
    ]

    buffer = UInt8[]
    for (pos, id) in enumerate(ids[1:(end-1)])
        (buffer, decoded) = decode_unique_encoding(buffer, id_to_token[id])
        printstyled(decoded, bold = true, color = :light_black)
        transform!(k_caches, v_caches, model, id, pos)
    end
    id = ids[end]
    (buffer, decoded) = decode_unique_encoding(buffer, id_to_token[id])
    printstyled(decoded, bold = true, color = :light_black)
    for pos = (length(ids)+1):config["n_ctx"]
        logits = transform!(k_caches, v_caches, model, id, pos)
        id = argmax(logits) - 1
        (buffer, decoded) = decode_unique_encoding(buffer, id_to_token[id])
        printstyled(decoded, bold = true)
    end
end

main()
