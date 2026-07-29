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

function mshow(matrix) 
    show(IOContext(stdout, :limit => true), "text/plain", matrix)
    println()
end

# The transformer for the GPT-2 architecture.
function transform!(tensors, config, id, pos, cached_k, cached_v)
    #### Embedding ####

    # ids are 0-based. Julia is 1-based.
    x = tensors["wte.weight"][id + 1, :] + tensors["wpe.weight"][pos, :]

    ε = Float32(config["layer_norm_epsilon"])

    for i_layer = 0:(config["n_layer"]-1)
        #### Masked Multi-Head Attention ####

        y =
            (x .- mean(x)) ./
            √(var(x, corrected = false) + ε) .*
            tensors["h.$i_layer.ln_1.weight"] .+
            tensors["h.$i_layer.ln_1.bias"]

        y =
            permutedims(tensors["h.$i_layer.attn.c_attn.weight"]) * y +
            tensors["h.$i_layer.attn.c_attn.bias"]

        q, k, v = [y[(config["n_embd"]*(i-1)+1):(config["n_embd"]*i)] for i = 1:3]

        size_of_head = config["n_embd"] ÷ config["n_head"]

        q_heads = [q[(size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        k_heads = [k[(size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        v_heads = [v[(size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]

        # Here I'll use the KV-cache!
        k_heads_full = [vcat(cached, permutedims(head)) for (cached, head) = zip(cached_k[i_layer + 1], k_heads)]
        v_heads_full = [vcat(cached, permutedims(head)) for (cached, head) = zip(cached_v[i_layer + 1], v_heads)]
        # TODO: Can I use broadcasting `.` here?

        cached_k[i_layer + 1] = k_heads_full
        cached_v[i_layer + 1] = v_heads_full

        #=
        y = cat(
            (
                begin
                    z = (
                        tril(q * transpose(k) ./ √Float32(pos)) +
                        triu(fill(-Inf32, (length(q), length(q))), 1)
                    )
                    z = exp.(z .- maximum(z))
                    z ./ sum(z) * v
                end for (q, k, v) ∈ zip(q_heads, k_heads, v_heads)
            )...,
            dims = 1
        )
        =#

        # I haven't realized until now that
        # I don't need the causal mask anymore
        # if I use the KV cache!!!
        y = [begin
            z = (
                permutedims(q) * transpose(k) ./ √Float32(length(q))
            )
            z = exp.(z .- maximum(z))
            z ./ sum(z) * v
        end for (q, k, v) ∈ zip(q_heads, k_heads_full, v_heads_full)]

        y = hcat(y...)

        y =
            permutedims(tensors["h.$i_layer.attn.c_proj.weight"]) * vec(y) +
            tensors["h.$i_layer.attn.c_proj.bias"]

        x += y

        #### Feed Forward ####

        y =
            (x .- mean(x)) ./
            √(var(x, corrected = false) + ε) .*
            tensors["h.$i_layer.ln_2.weight"] .+
            tensors["h.$i_layer.ln_2.bias"]

        y =
            permutedims(tensors["h.$i_layer.mlp.c_fc.weight"]) * y +
            tensors["h.$i_layer.mlp.c_fc.bias"]

        y = (tanh.((y .^ 3 * 0.044715f0 + y) * √(2.0f0 / π)) .+ 1.0f0) .* y * 0.5f0

        y =
            permutedims(tensors["h.$i_layer.mlp.c_proj.weight"]) * y +
            tensors["h.$i_layer.mlp.c_proj.bias"]

        x += y

    end

    #### Projection ####

    x =
        (x .- mean(x)) ./ .√(var(x, corrected = false) + ε) .* tensors["ln_f.weight"] +
        tensors["ln_f.bias"]

    tensors["wte.weight"] * x
end

#### Main ####

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

    printstyled(ARGS[2], bold = true, color = :light_black)

    size_of_head = config["n_embd"] ÷ config["n_head"]

    cached_k = [[Array{Float32}(undef, 0, size_of_head) for _ = 1:config["n_head"]] for _ = 1:config["n_layer"]]
    cached_v = [[Array{Float32}(undef, 0, size_of_head) for _ = 1:config["n_head"]] for _ = 1:config["n_layer"]]

    pos = 1

    x = undef

    for id = ids
        x = transform!(tensors, config, id, pos, cached_k, cached_v)
        pos += 1
    end

    buffer = UInt8[]

    next_id = argmax(x) - 1
    (buffer, decoded) = decode_unique_encoding(buffer, id_to_token[next_id])
    printstyled(decoded, bold = true)

    while pos ≤ config["n_ctx"]
        x = transform!(tensors, config, next_id, pos, cached_k, cached_v)

        # ids are 0-based. Julia is 1-based.
        next_id = argmax(x) - 1
        (buffer, decoded) = decode_unique_encoding(buffer, id_to_token[next_id])
        printstyled(decoded, bold = true)

        pos += 1

        #=
        if length(ids) == config["n_ctx"]
            popfirst!(ids)
        end
        push!(ids, next_id)
        =#
    end
end

main()

# KV Cache done!
# faster!
