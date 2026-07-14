# Julia!

using SafeTensors

path_to_model = ARGS[1]
ids = [parse(Int64, arg) for arg in ARGS[2:end]]

println(path_to_model)
println(ids)
tensors = load_safetensors("$path_to_model/model.safetensors")

disp(matrix) = Base.print_matrix(IOContext(Base.stdout, :limit => true), matrix)

disp(tensors["wte.weight"])
