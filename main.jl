using LinearAlgebra
using Statistics

using JSON
using SafeTensors

function transform(tensors, config, ids)
    #### Embedding ####

    # ids are 0-based. Julia is 1-based.
    x = tensors["wte.weight"][ids .+ 1, :] .+ tensors["wpe.weight"][1:length(ids), :]

    for i_layer = 0:(config["n_layer"]-1)
        #### Masked Multi-Head Attention ####

        y =
            (x .- mean(x, dims = 2)) ./ .√(var(x, corrected = false, dims = 2) .+ 1.0f-5) .*
            permutedims(tensors["h.$i_layer.ln_1.weight"]) .+
            permutedims(tensors["h.$i_layer.ln_1.bias"])

        y =
            y * tensors["h.$i_layer.attn.c_attn.weight"] .+
            permutedims(tensors["h.$i_layer.attn.c_attn.bias"])

        q, k, v = [y[:, (config["n_embd"]*(i-1)+1):(config["n_embd"]*i)] for i = 1:3]

        size_of_head = config["n_embd"] ÷ config["n_head"]

        q_heads = [q[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        k_heads = [k[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        v_heads = [v[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]

        y = cat(
            (
                begin
                    z = (
                        tril(q * transpose(k) ./ √Float32(size(q, 2))) +
                        triu(fill(-Inf32, (size(q, 1), size(q, 1))), 1)
                    )
                    z = exp.(z .- maximum(z, dims = 2))
                    z ./ sum(z, dims = 2) * v
                end for (q, k, v) ∈ zip(q_heads, k_heads, v_heads)
            )...,
            dims = 2,
        )

        y =
            y * tensors["h.$i_layer.attn.c_proj.weight"] .+
            permutedims(tensors["h.$i_layer.attn.c_proj.bias"])

        x += y

        #### Feed Forward ####

        y =
            (x .- mean(x, dims = 2)) ./ .√(var(x, corrected = false, dims = 2) .+ 1.0f-5) .*
            permutedims(tensors["h.$i_layer.ln_2.weight"]) .+
            permutedims(tensors["h.$i_layer.ln_2.bias"])

        y =
            y * tensors["h.$i_layer.mlp.c_fc.weight"] .+
            permutedims(tensors["h.$i_layer.mlp.c_fc.bias"])

        y = 0.5f0 .* y .* (1.0f0 .+ tanh.(√(2.0f0 / π) .* (y .+ 0.044715f0 .* (y .^ 3))))

        y =
            y * tensors["h.$i_layer.mlp.c_proj.weight"] .+
            permutedims(tensors["h.$i_layer.mlp.c_proj.bias"])

        x += y
    end

    #### Projection ####

    # I only need the last row to predict the next token ID.
    x = x[end, :]

    x =
        (x .- mean(x)) ./ .√(var(x, corrected = false) + 1.0f-5) .* tensors["ln_f.weight"] + tensors["ln_f.bias"]

    tensors["wte.weight"] * x
end

tensors = load_safetensors("$(ARGS[1])/model.safetensors")
config = JSON.parsefile("$(ARGS[1])/config.json")

token_to_id = JSON.parsefile("$(ARGS[1])/vocab.json")
id_to_token = Dict(id => token for (token, id) ∈ token_to_id)

ids = [parse(Int64, arg) for arg ∈ ARGS[2:end]]
if length(ids) == 0
    println("Your prompt should not be empty.")
    exit()
elseif length(ids) > config["n_ctx"]
    println("Your prompt exceeds the context length. Try shorter prompt.")
    exit()
end

function decode_unique_encoding!(str, buffer)
    decoded = [
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
        end for codepoint ∈ transcode(UInt32, str)
    ]

    decoded = vcat(buffer, decoded)

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

    # Case C3
    if length(decoded) ≥ 3 &&
       decoded[end-2] ∈ 0xC0:0xDF &&
       decoded[end-1] ∈ 0x80:0xBF &&
       decoded[end] ∈ 0x80:0xBF
        a = decoded[begin:(end-2)]
        b = decoded[(end-2):end]

        # Case B2 and Case C2
    elseif length(decoded) ≥ 2 && decoded[end-1] ∈ 0xC0:0xEF && decoded[end] ∈ 0x80:0xBF
        a = decoded[begin:(end-1)]
        b = decoded[(end-1):end]

        # Case A1, Case B1, and Case C1
    elseif length(decoded) ≥ 1 && decoded[end] ∈ 0xC0:0xF7
        a = decoded[begin:end]
        b = decoded[end:end]

        # No unfinished sequence at the end
    else
        a = decoded
        b = UInt8[]

    end

    c = transcode(String, a)
    d = string(((valid ? s : '�') for (s, valid) ∈ zip(c, isvalid.(collect(c))))...)

    d
end

buffer = UInt8[]
while true
    x = transform(tensors, config, ids)

    # ids are 0-based. Julia is 1-based.
    next_id = argmax(x) - 1
    printstyled(decode_unique_encoding!(id_to_token[next_id], buffer), bold = true)

    if length(ids) == config["n_ctx"]
        popfirst!(ids)
    end
    push!(ids, next_id)
end
