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

struct Layer
    g1::Vector{Float32}
    t1::Vector{Float32}
    w11::Matrix{Float32}
    b11::Vector{Float32}
    w12::Matrix{Float32}
    b12::Vector{Float32}
    g2::Vector{Float32}
    t2::Vector{Float32}
    w21::Matrix{Float32}
    b21::Vector{Float32}
    w22::Matrix{Float32}
    b22::Vector{Float32}
end

struct Model
    n_ctx::Int
    n_embd::Int
    n_head::Int
    n_layer::Int
    vocab_size::Int
    e::Float32
    wte::Matrix{Float32}
    wpe::Matrix{Float32}
    layers::Array{Layer}
    gf::Vector{Float32}
    tf::Vector{Float32}
end

function get_model(tensors::Dict{String,Array}, config::JSON.Object)::Model
    n_ctx = config["n_ctx"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    vocab_size = config["vocab_size"]
    e = Float32(config["layer_norm_epsilon"])

    # It feels more natural for me
    # to perform "matrix * vector -> vector"
    # than to perform "row vector * matrix -> row vector."
    # Therefore, I apply `permutedims` to the all matrices.

    function validate_size(tensor, expected)
        if size(tensor) ≠ expected
            error("size of tensor $(size(tensor)) differs from expected $expected")
        end
    end

    wte = permutedims(tensors["wte.weight"])
    validate_size(wte, (n_embd, vocab_size))

    wpe = permutedims(tensors["wpe.weight"])
    validate_size(wpe, (n_embd, n_ctx))

    layers = Layer[]
    for i = 0:(n_layer-1)
        g1 = tensors["h.$i.ln_1.weight"]
        validate_size(g1, (n_embd,))

        t1 = tensors["h.$i.ln_1.bias"]
        validate_size(t1, (n_embd,))

        w11 = permutedims(tensors["h.$i.attn.c_attn.weight"])
        validate_size(w11, (n_embd * 3, n_embd))

        b11 = tensors["h.$i.attn.c_attn.bias"]
        validate_size(b11, (n_embd * 3,))

        w12 = permutedims(tensors["h.$i.attn.c_proj.weight"])
        validate_size(w12, (n_embd, n_embd))

        b12 = tensors["h.$i.attn.c_proj.bias"]
        validate_size(b12, (n_embd,))

        g2 = tensors["h.$i.ln_2.weight"]
        validate_size(g2, (n_embd,))

        t2 = tensors["h.$i.ln_2.bias"]
        validate_size(t2, (n_embd,))

        w21 = permutedims(tensors["h.$i.mlp.c_fc.weight"])
        validate_size(w21, (n_embd * 4, n_embd))

        b21 = tensors["h.$i.mlp.c_fc.bias"]
        validate_size(b21, (n_embd * 4,))

        w22 = permutedims(tensors["h.$i.mlp.c_proj.weight"])
        validate_size(w22, (n_embd, n_embd * 4))

        b22 = tensors["h.$i.mlp.c_proj.bias"]
        validate_size(b22, (n_embd,))

        layer = Layer(g1, t1, w11, b11, w12, b12, g2, t2, w21, b21, w22, b22)
        push!(layers, layer)
    end

    gf = tensors["ln_f.weight"]
    validate_size(gf, (n_embd,))

    tf = tensors["ln_f.bias"]
    validate_size(tf, (n_embd,))

    Model(n_ctx, n_embd, n_head, n_layer, vocab_size, e, wte, wpe, layers, gf, tf)
end

# The paper that introduced layer norm uses uncorrected variance.
# https://arxiv.org/abs/1607.06450
layer_norm(x::Vector{Float32}, g::Vector{Float32}, t::Vector{Float32}, e::Float32) =
    g .* (x .- mean(x)) ./ √(var(x, corrected = false) + e) + t

# The transformer of the GPT-2 architecture, the heart of the inference engine.
function transform!(
    cached_k::Vector{Vector{Matrix{Float32}}},
    cached_v::Vector{Vector{Matrix{Float32}}},
    model::Model,
    id::Int,
    pos::Int,
)::Vector{Float32}
    #### Embedding ####

    # ids are 0-based. Julia is 1-based.
    x = model.wte[:, id+1] + model.wpe[:, pos]

    for (layer, k_matrices, v_matrices) ∈ zip(model.layers, cached_k, cached_v)
        #### Masked Multi-Head Attention ####

        y = layer_norm(x, layer.g1, layer.t1, model.e)

        y = layer.w11 * y + layer.b11

        q_vectors, k_vectors, v_vectors = (
            Iterators.partition(chunk, model.n_embd ÷ model.n_head) for
            chunk ∈ Iterators.partition(y, model.n_embd)
        )
        k_matrices[:] = hcat.(k_matrices, k_vectors)
        v_matrices[:] = hcat.(v_matrices, v_vectors)
        y = (
            begin
                z = k' * q ./ √Float32(model.n_embd ÷ model.n_head)
                z = exp.(z .- maximum(z))
                v * z ./ sum(z)
            end for (q, k, v) ∈ zip(q_vectors, k_matrices, v_matrices)
        )
        y = vcat(y...)

        y = layer.w12 * y + layer.b12

        x += y

        #### Feed Forward ####

        y = layer_norm(x, layer.g2, layer.t2, model.e)

        y = layer.w21 * y + layer.b21

        # This formula is based on the paper that introduced GELU.
        # https://arxiv.org/abs/1606.08415
        y = (tanh.((y .^ 3 * 0.044715f0 + y) * √(2.0f0 / π)) .+ 1.0f0) .* y * 0.5f0

        y = layer.w22 * y + layer.b22

        x += y
    end

    #### Projection ####

    x = layer_norm(x, model.gf, model.tf, model.e)

    model.wte' * x
end
