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

module Transformer

export transformer!

using Statistics

using ..Model

function layer_norm(
    x::Vector{Float32},
    g::Vector{Float32},
    t::Vector{Float32},
    model::Model,
)::Vector{Float32}
    # The paper that introduced layer norm uses uncorrected variance.
    # https://arxiv.org/abs/1607.06450
    g .* (x .- mean(x)) ./ √(var(x, corrected = false) + model.e) + t
end

# Numerically stable softmax
function softmax(x::Vector{Float32})::Vector{Float32}
    x = exp.(x .- maximum(x))
    x / sum(x)
end

function multi_head_attention!(
    x::Vector{Float32},
    layer::Layer,
    k::Vector{Matrix{Float32}},
    v::Vector{Matrix{Float32}},
    pos::Int,
    model::Model,
)::Vector{Float32}
    x = layer.w11 * x + layer.b11
    chunks = Iterators.partition.(
        Iterators.partition(x, model.n_embd),
        model.n_embd ÷ model.n_head,
    )
    q = popfirst!(chunks)
    k[:] = hcat.(k, popfirst!(chunks))
    v[:] = hcat.(v, popfirst!(chunks))
    # Scaled dot-product attention
    a = v .* softmax.(transpose.(k) .* q ./ √Float32(model.n_embd ÷ model.n_head))
    x = vcat(a...)
    x = layer.w12 * x + layer.b12
    x
end

function feed_forward(x::Vector{Float32}, layer::Layer, model::Model)::Vector{Float32}
    x = layer.w21 * x + layer.b21
    # This formula is based on the paper that introduced GELU.
    # https://arxiv.org/abs/1606.08415
    x = (tanh.((x .^ 3 * 0.044715f0 + x) * √(2.0f0 / π)) .+ 1.0f0) .* x * 0.5f0
    x = layer.w22 * x + layer.b22
    x
end

# The transformer of the GPT-2 architecture.
function transformer!(
    id::Int,
    pos::Int,
    model::Model,
    k_caches::Vector{Vector{Matrix{Float32}}},
    v_caches::Vector{Vector{Matrix{Float32}}},
)::Vector{Float32}
    # ids are 0-based. Julia is 1-based.
    x = model.wte[:, id+1] + model.wpe[:, pos]
    for (layer, k, v) ∈ zip(model.layers, k_caches, v_caches)
        y = layer_norm(x, layer.g1, layer.t1, model)
        y = multi_head_attention!(y, layer, k, v, pos, model)
        x += y
        y = layer_norm(x, layer.g2, layer.t2, model)
        y = feed_forward(y, layer, model)
        x += y
    end
    x = layer_norm(x, model.gf, model.tf, model)
    x = transpose(model.wte) * x
    x
end

end
