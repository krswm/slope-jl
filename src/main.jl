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
using Printf
using Random
using Statistics

using JSON
using SafeTensors

include("tokenizer.jl")
include("transformer.jl")

function main()::Nothing
    if length(ARGS) ≠ 2
        println("GPT-2 Inference with Julia")
        print("Usage: ")
        printstyled("julia main.jl <path to model repository> <your prompt>", bold = true)
        println()
        println("You may have to enclose 'your prompt' with quotes.")
        exit()
    end

    #### Loading Files ####

    token_to_id, id_to_token = begin
        vocab = JSON.parsefile("$(ARGS[1])/vocab.json")
        token_to_id = Dict(token => id for (token, id) ∈ vocab)
        id_to_token = Dict(id => token for (token, id) ∈ vocab)
        token_to_id, id_to_token
    end

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

    model = begin
        tensors = load_safetensors("$(ARGS[1])/model.safetensors")
        config = JSON.parsefile("$(ARGS[1])/config.json")
        get_model(tensors, config)
    end

    #### Tokenization ####

    ids = tokenize(token_to_id, ranks, ARGS[2])
    if length(ids) == 0
        println("Your prompt should not be empty.")
        exit()
    elseif length(ids) > model.n_ctx
        println("Your prompt exceeds the context length. Try shorter prompt.")
        exit()
    end

    #### Inference ####

    buffer = UInt8[]
    cached_k = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    cached_v = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]

    begin_time = time_ns()
    for (pos, id) ∈ enumerate(ids[1:(end-1)])
        decoded = decode_unique_encoding!(buffer, id_to_token[id])
        printstyled(decoded, bold = true, color = :light_black)
        transform!(cached_k, cached_v, model, id, pos)
    end
    id = ids[end]
    decoded = decode_unique_encoding!(buffer, id_to_token[id])
    printstyled(decoded, bold = true, color = :light_black)
    for pos = length(ids):config["n_ctx"]
        logits = transform!(cached_k, cached_v, model, id, pos)

        # Temperature! Physics! Statistical mechanics <3
        function tempsoftmax(x, T)
            x = exp.((x .- maximum(x)) ./ T)
            x / sum(x)
        end
        
        logits = tempsoftmax(logits, 10.0f0)

        @assert sum(logits) ≈ 1.0f0

        random = rand(Float32)
        sum_prob = 0.0f0
        for (i, prob) ∈ enumerate(logits)
            sum_prob += prob
            if random < sum_prob
                id = i - 1
                break
            end
        end

           
        decoded = decode_unique_encoding!(buffer, id_to_token[id])
        printstyled(decoded, bold = true)
    end
    end_time = time_ns()

    process_time = end_time - begin_time
    sec = process_time * 1e-9
    println()
    printstyled("Transformer processed $(model.n_ctx) tokens", color = :light_black)
    println()
    printstyled(
        (@sprintf "Took %.3f s | %.3f tokens/s" sec (model.n_ctx / sec)),
        color = :light_black,
    )
    println()
    printstyled(
        "$(length(ids)) tokens prompted | ",
        "$(model.n_ctx - length(ids) + 1) tokens generated",  # There is an extra token.
        color = :light_black,
    )
    println()
end

main()
