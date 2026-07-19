using LinearAlgebra
using Statistics

using JSON
using SafeTensors

# The transformer for the GPT-2 architecture.
function transform(tensors, config, ids)
    size_of_head = config["n_embd"] ÷ config["n_head"]

    #### Embedding ####

    # ids are 0-based. Julia is 1-based.
    x = tensors["wte.weight"][ids .+ 1, :] .+ tensors["wpe.weight"][1:length(ids), :]

    for i_layer = 0:(config["n_layer"]-1)
        #### Masked Multi-Head Attention ####

        y =
            (x .- mean(x, dims = 2)) ./
            sqrt.(var(x, corrected = false, dims = 2) .+ 1.0f-5) .*
            transpose(tensors["h.$i_layer.ln_1.weight"]) .+
            transpose(tensors["h.$i_layer.ln_1.bias"])

        y =
            y * tensors["h.$i_layer.attn.c_attn.weight"] .+
            transpose(tensors["h.$i_layer.attn.c_attn.bias"])

        q, k, v = [y[:, (config["n_embd"]*(i-1)+1):(config["n_embd"]*i)] for i = 1:3]

        q_heads = [q[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        k_heads = [k[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]
        v_heads = [v[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:config["n_head"]]

        y = cat(
            [
                begin
                    z = (
                        tril(q * transpose(k) ./ sqrt(Float32(size(q, 2)))) +
                        triu(fill(-Inf32, (size(q, 1), size(q, 1))), 1)
                    )

                    z = exp.(
                        z .- reshape([maximum(row) for row in eachrow(z)], (size(z, 1), 1)),
                    )

                    (z ./ reshape([sum(row) for row in eachrow(z)], (size(z, 1), 1))) * v
                end for (q, k, v) in zip(q_heads, k_heads, v_heads)
            ]...,
            dims = 2,
        )

        y =
            y * tensors["h.$i_layer.attn.c_proj.weight"] .+
            transpose(tensors["h.$i_layer.attn.c_proj.bias"])

        x += y

        #### Feed Forward ####

        y =
            (x .- mean(x, dims = 2)) ./
            sqrt.(var(x, corrected = false, dims = 2) .+ 1.0f-5) .*
            transpose(tensors["h.$i_layer.ln_2.weight"]) .+
            transpose(tensors["h.$i_layer.ln_2.bias"])

        y =
            y * tensors["h.$i_layer.mlp.c_fc.weight"] .+
            transpose(tensors["h.$i_layer.mlp.c_fc.bias"])

        y = 0.5f0 .* y .* (1.0f0 .+ tanh.(sqrt(2.0f0 / π) .* (y .+ 0.044715f0 .* (y .^ 3))))

        y =
            y * tensors["h.$i_layer.mlp.c_proj.weight"] .+
            transpose(tensors["h.$i_layer.mlp.c_proj.bias"])

        x += y
    end

    #### Projection ####

    x =
        (x .- mean(x, dims = 2)) ./ sqrt.(var(x, corrected = false, dims = 2) .+ 1.0f-5) .*
        transpose(tensors["ln_f.weight"]) .+ transpose(tensors["ln_f.bias"])

    tensors["wte.weight"] * x[end, :]
end

tensors = load_safetensors("$(ARGS[1])/model.safetensors")
config = JSON.parsefile("$(ARGS[1])/config.json")

ids = [parse(Int64, arg) for arg in ARGS[2:end]]
if length(ids) == 0
    println("Your prompt should not be empty.")
    exit()
elseif length(ids) > config["n_ctx"]
    println("Your prompt exceeds the context length. Try shorter prompt.")
    exit()
end

while true
    x = transform(tensors, config, ids)

    # ids are 0-based. Julia is 1-based.
    next_id = argmax(x) - 1
    println(next_id)

    if length(ids) == config["n_ctx"]
        popfirst!(ids)
    end
    push!(ids, next_id)
end
