# Julia!

using SafeTensors

path_to_model = ARGS[1]
ids = [parse(Int64, arg) for arg in ARGS[2:end]]

println(path_to_model)
println(ids)
tensors = load_safetensors("../../Downloads/gpt2/model.safetensors")
