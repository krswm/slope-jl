function encode_unique_encoding(text)
    encoded = [
        if byte ∈ 0x00:0x20
            UInt32(byte + 0x0100)
        elseif byte ∈ 0x21:0x7E
            UInt32(byte)
        elseif byte ∈ 0x7F:0xA0
            UInt32(byte + 0x00A2)
        elseif byte ∈ 0xA1:0xAC
            UInt32(byte)
        elseif byte == 0xAD
            0x00000143
        elseif byte ∈ 0xAE:0xFF
            UInt32(byte)
        end for byte ∈ transcode(UInt8, text)
    ]
    transcode(String, encoded)
end

function load_ranks()
    ranks = Dict()
    rank = 0
    for line ∈ readlines(ARGS[1])
        # Skip a comment line.
        if startswith(line, "#") continue end

        token0, token1 = split(line, " ")
        ranks[(token0, token1)] = rank
        rank += 1
    end
    ranks
end

ranks = load_ranks()
print(ranks)
