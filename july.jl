# Julia!

using SafeTensors

path_to_model = ARGS[1]
ids = [parse(Int64, arg) for arg in ARGS[2:end]]

println(path_to_model)
println(ids)
tensors = load_safetensors("$path_to_model/model.safetensors")

function disp(matrix) 
    println(Base.stdout, summary(matrix), ":")
    Base.print_matrix(IOContext(Base.stdout, :limit => true), matrix)
    println()
end

# A = tensors["wte.weight"][ids .+ 1, :]  # The token IDs are 0-based but Julia is 1-based!
# disp(A)

B = tensors["wte.weight"][ids .+ 1, :] .+ tensors["wpe.weight"][1:length(ids), :]
# disp(B)

# return weight * (y - y.mean(-1, keepdim=True)) / (y.var(-1, keepdim=True, correction=0.0) + 1e-5).sqrt() + bias

using Statistics

# disp(tensors["h.0.ln_1.weight"])
# disp(tensors["h.0.ln_1.bias"])

C = (B .- mean(B, dims=2)) ./ sqrt.(var(B, dims=2, corrected=false) .+ 1e-5) .* transpose(tensors["h.0.ln_1.weight"]) .+ transpose(tensors["h.0.ln_1.bias"])
# disp(C)

E = C * tensors["h.0.attn.c_attn.weight"] .+ transpose(tensors["h.0.attn.c_attn.bias"])  # np/pt "@" -> jl "*", np/pt "*" -> jl ".*"
# disp(E)

n_embd = 768  # don't hardcode!

q, k, v = [E[:, (n_embd * (i - 1) + 1):(n_embd * i)] for i in 1:3]
# disp(q)
# disp(k)
# disp(v)

n_head = 12  # don't hardcode!
N = n_embd ÷ n_head

q_heads = [q[:, (N * (i - 1) + 1):(N * i)] for i in 1:n_head]
k_heads = [k[:, (N * (i - 1) + 1):(N * i)] for i in 1:n_head]
v_heads = [v[:, (N * (i - 1) + 1):(N * i)] for i in 1:n_head]

# disp(v_heads[6])

using LinearAlgebra

function attention(q, k, v)
    y = (tril(q * transpose(k) ./ sqrt(size(q, 2))) + triu(ones(size(q, 1), size(q, 1)) * -1e12, 1))

    e = exp.(y .- reshape([maximum(row) for row in eachrow(y)], (size(y, 1), 1)))
    # disp(e)

    return (e ./ reshape([sum(row) for row in eachrow(e)], (size(e, 1), 1))) * v
end


heads =[
    attention(q, k, v) for (q, k, v) in zip(q_heads, k_heads, v_heads)
]

# show(heads[8])

stacked = cat(heads..., dims=2)

M = stacked * tensors["h.0.attn.c_proj.weight"] .+ transpose(tensors["h.0.attn.c_proj.bias"])
N = B + M

O = (N .- mean(N, dims=2)) ./ sqrt.(var(N, dims=2, corrected=false) .+ 1e-5) .* transpose(tensors["h.0.ln_2.weight"]) .+ transpose(tensors["h.0.ln_2.bias"])

Q = O * tensors["h.0.mlp.c_fc.weight"] .+ transpose(tensors["h.0.mlp.c_fc.bias"])

# gelu
R = 0.5 .* Q .* (1.0 .+ tanh.(sqrt(2.0 / pi) .* (Q .+ 0.044715 .* (Q .^ 3))))

S = R * tensors["h.0.mlp.c_proj.weight"] .+ transpose(tensors["h.0.mlp.c_proj.bias"])

B = N + S
disp(B)
