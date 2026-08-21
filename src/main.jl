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

using Printf
using Random

using JSON
using SafeTensors

include("model.jl")
using .Model
include("tokenizer.jl")
using .Tokenizer
include("transformer.jl")
using .Transformer

function main()::Nothing
    if length(ARGS) ≠ 3
        println("GPT-2 Inference with Julia")
        print("Usage: ")
        printstyled(
            "julia --project $PROGRAM_FILE <path to model repository> <sampling temperature> <your prompt>",
            bold = true,
        )
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

    #### Temperature ####

    temperature = parse(Float32, ARGS[2])
    if temperature < 0.0f0
        println("Temperature must be ≥ 0.0.")
        exit()
    end

    #### Tokenization ####

    ids = tokenize(token_to_id, ranks, ARGS[3])
    if length(ids) == 0
        println("Your prompt should not be empty.")
        exit()
    elseif length(ids) > model.n_ctx
        println("Your prompt exceeds the context length. Try shorter prompt.")
        exit()
    end

    #### Inference ####

    num_prompted_tokens = 0
    num_processed_tokens = 0
    num_generated_tokens = 0
    id = 0
    utf8_buffer = UInt8[]
    k_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    v_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]

    performance_timer = time_ns()
    for pos = 1:model.n_ctx
        if pos ≤ length(ids)
            id = ids[pos]
            decoded = decode_unique_encoding!(utf8_buffer, id_to_token[id])
            printstyled(decoded, bold = true, color = :light_black)
            num_prompted_tokens += 1
        end

        logits = transform!(k_caches, v_caches, model, id, pos)
        num_processed_tokens += 1

        if pos ≥ length(ids)
            if temperature == 0.0f32
                # id is 1-based. Julia is 0-based.
                id = argmax(logits) - 1
            else
                x = exp.((logits .- maximum(logits)) ./ temperature)
                x = x / sum(x)
                rand_prob = rand(Float32)
                total_prob = 0.0f0
                for (i, prob) ∈ enumerate(x)
                    total_prob += prob
                    if rand_prob < total_prob
                        # id is 1-based. Julia is 0-based.
                        id = i - 1
                        break
                    end
                end
            end
            decoded = decode_unique_encoding!(utf8_buffer, id_to_token[id])
            printstyled(decoded, bold = true)
            num_generated_tokens += 1
        end
    end
    println()
    performance_time = (time_ns() - performance_timer) * 1e-9

    printstyled("Took $(@sprintf "%.3f" performance_time) s", color = :light_black)
    println()
    printstyled(
        "$num_prompted_tokens $(num_prompted_tokens == 1 ? "token" : "tokens") prompted",
        color = :light_black,
    )
    println()
    printstyled(
        "$num_processed_tokens $(num_processed_tokens == 1 ? "token" : "tokens") processed by the transformer ",
        "($(@sprintf "%.3f" (num_processed_tokens / performance_time)) tok/s)",
        color = :light_black,
    )
    println()
    printstyled(
        "$num_generated_tokens $(num_generated_tokens == 1 ? "token" : "tokens") generated",
        color = :light_black,
    )
    println()
end

main()
