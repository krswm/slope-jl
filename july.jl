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

q = E[:, 1:n_embd]
k = E[:, (n_embd + 1):(2 * n_embd)]
v = E[:, (2 * n_embd + 1):(3 * n_embd)]
disp(q)
disp(k)
disp(v)
