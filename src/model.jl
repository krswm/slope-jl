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

module Model

export Layer, Model, get_model

using JSON

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

function validate_size(tensor, expected)
    if size(tensor) ≠ expected
        error("size of tensor $(size(tensor)) differs from expected $expected")
    end
end

function get_model(tensors::Dict{String,Array}, config::JSON.Object)::Model
    n_ctx = config["n_ctx"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    vocab_size = config["vocab_size"]
    e = Float32(config["layer_norm_epsilon"])

    # Regarding a product between a matrix and a vector,
    # it feels more natural for me to perform:
    #               ┏━━━┓
    # ┏━━━┯━━━┯━━━┓ ┃ a ┃   ┏━━━━━━━━━━┓
    # ┃ d │ f │ h ┃ ┠───┨   ┃ ad+bf+ch ┃
    # ┠───┼───┼───┨ ┃ b ┃ → ┠──────────┨
    # ┃ e │ g │ i ┃ ┠───┨   ┃ ae+bg+ci ┃
    # ┗━━━┷━━━┷━━━┛ ┃ c ┃   ┗━━━━━━━━━━┛
    #               ┗━━━┛
    # than to perform:
    #               ┏━━━┯━━━┓
    #               ┃ d │ e ┃
    # ┏━━━┯━━━┯━━━┓ ┠───┼───┨   ┏━━━━━━━━━━┯━━━━━━━━━━┓
    # ┃ a │ b │ c ┃ ┃ f │ g ┃ → ┃ ab+bf+ch │ ae+bg+ci ┃
    # ┗━━━┷━━━┷━━━┛ ┠───┼───┨   ┗━━━━━━━━━━┷━━━━━━━━━━┛
    #               ┃ h │ i ┃
    #               ┗━━━┷━━━┛
    # Therefore, I apply `permutedims` to 2D tensors here.
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

end
