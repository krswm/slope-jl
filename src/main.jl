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

include("tokenizer.jl")
include("transformer.jl")

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
