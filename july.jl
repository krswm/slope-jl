using LinearAlgebra
using SafeTensors
using Statistics

tensors = load_safetensors("$(ARGS[1])/model.safetensors")
ids = [parse(Int64, arg) for arg in ARGS[2:end]]

# ids are 0-based. Julia is 1-based.
x = tensors["wte.weight"][ids .+ 1, :] .+ tensors["wpe.weight"][1:length(ids), :]

# TODO: Do not hardcode them.
n_embd = 768
n_head = 12
n_layer = 12

size_of_head = n_embd ÷ n_head

function attention(q, k, v)
    x = (
        tril(q * transpose(k) ./ sqrt(size(q, 2))) +
        triu(ones(size(q, 1), size(q, 1)) * -1e12, 1)
    )

    x = exp.(x .- reshape([maximum(row) for row in eachrow(x)], (size(x, 1), 1)))

    return (x ./ reshape([sum(row) for row in eachrow(x)], (size(x, 1), 1))) * v
end

for i_layer = 0:(n_layer-1)
    global x

    y =
        (x .- mean(x, dims = 2)) ./ sqrt.(var(x, dims = 2, corrected = false) .+ 1e-5) .*
        transpose(tensors["h.$i_layer.ln_1.weight"]) .+
        transpose(tensors["h.$i_layer.ln_1.bias"])

    y =
        y * tensors["h.$i_layer.attn.c_attn.weight"] .+
        transpose(tensors["h.$i_layer.attn.c_attn.bias"])

    q, k, v = [y[:, (n_embd*(i-1)+1):(n_embd*i)] for i = 1:3]

    q_heads = [q[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:n_head]
    k_heads = [k[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:n_head]
    v_heads = [v[:, (size_of_head*(i-1)+1):(size_of_head*i)] for i = 1:n_head]

    y = cat(
        [attention(q, k, v) for (q, k, v) in zip(q_heads, k_heads, v_heads)]...,
        dims = 2,
    )

    y =
        y * tensors["h.$i_layer.attn.c_proj.weight"] .+
        transpose(tensors["h.$i_layer.attn.c_proj.bias"])

    x = x + y

    O =
        (x .- mean(x, dims = 2)) ./ sqrt.(var(x, dims = 2, corrected = false) .+ 1e-5) .*
        transpose(tensors["h.$i_layer.ln_2.weight"]) .+
        transpose(tensors["h.$i_layer.ln_2.bias"])

    Q =
        O * tensors["h.$i_layer.mlp.c_fc.weight"] .+
        transpose(tensors["h.$i_layer.mlp.c_fc.bias"])

    R = 0.5 .* Q .* (1.0 .+ tanh.(sqrt(2.0 / pi) .* (Q .+ 0.044715 .* (Q .^ 3))))

    S =
        R * tensors["h.$i_layer.mlp.c_proj.weight"] .+
        transpose(tensors["h.$i_layer.mlp.c_proj.bias"])

    x = x + S
end

U =
    (x .- mean(x, dims = 2)) ./ sqrt.(var(x, dims = 2, corrected = false) .+ 1e-5) .*
    transpose(tensors["ln_f.weight"]) .+ transpose(tensors["ln_f.bias"])
V = U * transpose(tensors["wte.weight"])

Base.print_matrix(IOContext(Base.stdout, :limit => true), V)
