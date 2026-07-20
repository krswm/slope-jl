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

function load_ranks()
    ranks = Dict()
    rank = 0
    for line ∈ readlines("$(ARGS[1])/merges.txt")
        # Skip a comment line.
        if startswith(line, "#") continue end

        token0, token1 = split(line, " ")
        ranks[(token0, token1)] = rank
        rank += 1
    end
    ranks
end

ranks = load_ranks()

function tokenize(token_to_id, ranks, input)
    tokens = String[]
    for (i_line, line) ∈ enumerate(split(input, "\n"))
        if i_line ≥ 2 push!(tokens, "\n") end

        for (i_word, word) in enumerate(split(line, " "))
            if i_word == 1 && word ≠ ""
                push!(tokens, word)
            elseif i_word ≥ 2
                push!(tokens, " $word")
            end
        end
    end

    tokens = [encode_unique_encoding(token) for token ∈ tokens]

    # Token IDs
    ids = []
    for token ∈ tokens
        if haskey(token_to_id, token)
            push!(ids, token_to_id[token])
        else
            # Merge

            symbols = string.(collect(token))

            while length(symbols) ≥ 2
                pairs = [(token0, token1) for (token0, token1) ∈ zip(symbols[begin:end-1], symbols[begin+1:end])]

                best_rank = typemax(Int32)
                best_i_pair = typemax(Int32)
                for (i_pair, pair) ∈ enumerate(pairs)
                    if haskey(ranks, pair) && ranks[pair] < best_rank
                        best_rank = ranks[pair]
                        best_i_pair = i_pair
                    end
                end
                if best_i_pair == typemax(Int32) break end

                symbols[best_i_pair] = symbols[best_i_pair] * symbols[best_i_pair + 1]
                deleteat!(symbols, best_i_pair + 1)
            end

            for symbol in symbols
                push!(ids, token_to_id[symbol])
            end
        end
    end
    ids
end

using JSON

token_to_id = JSON.parsefile("$(ARGS[1])/vocab.json")

println(tokenize(token_to_id, ranks, "I love coding."))
println(tokenize(token_to_id, ranks, "Schrödinger"))
println(tokenize(token_to_id, ranks, "あ"))
