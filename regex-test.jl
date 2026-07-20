function decode_unique_encoding(str)
    # TODO: This function currently cannot handle an invalid UTF-8 sequence.
    decoded = [
            if 0x0100 <= codepoint <= 0x0120
                UInt8(codepoint - 0x0100)
            elseif 0x0021 <= codepoint <= 0x007E
                UInt8(codepoint)
            elseif 0x0121 <= codepoint <= 0x0142
                UInt8(codepoint - 0x00A2)
            elseif 0x00A1 <= codepoint <= 0x00AC
                UInt8(codepoint)
            elseif codepoint == 0x0143
                0xAD
            elseif 0x00AE <= codepoint <= 0x00FF
                UInt8(codepoint)
            end for codepoint in transcode(UInt32, str)
        ]

    # UTF-8 sequences that is valid if they are at the end of the string
    # and the string is decoded incrementally but not valid itself
    # 110xxxxx                   0xC0..=0xDF
    # 1110xxxx                   0xE0..=0xEF
    # 1110xxxx 10xxxxxx          0xE0..=0xEF 0x80..=0xBF
    # 11110xxx                   0xF0..=0xF7
    # 11110xxx 10xxxxxx          0xF0..=0xF7 0x80..=0xBF
    # 11110xxx 10xxxxxx 10xxxxxx 0xF0..=0xF7 0x80..=0xBF 0x80..=0xBF
    # I can regex it.

    println(repr(decoded))

    # match(
    #     # r"^([\x00-\xff]*?)([\xc0-\xdf]|[\xe0-\xef][\x80-\xbf]?|[\xf0-\xf7][\x80-\xbf]{,2})$",
    #     # r"^([\x00-\xff])",
    #     r"^([\x00-\xff]*?)(|[\xc0-\xdf]|[\xe0-\xef][\x80-\xbf]?|[\xf0-\xf7][\x80-\xbf]{,2})$",
    #     decoded,
    # )

    if length(decoded) >= 3 && 0xC0 <= decoded[end-1] <= 0xDF && 0x80 <= decoded[end] <= 0xBF && 0x80 <= decoded[end] <= 0xBF
        a = decoded[begin:end-2]
        b = decoded[end-2:end]
    elseif length(decoded) >= 2 && 0xC0 <= decoded[end-1] <= 0xEF && 0x80 <= decoded[end] <= 0xBF
        a = decoded[begin:end-1]
        b = decoded[end-1:end]
    elseif length(decoded) >= 1 && 0xC0 <= decoded[end] <= 0xF7
        a = decoded[begin:end]
        b = decoded[end:end]
    else
        a = decoded
        b = ""
    end

    c = transcode(String, a)
    d = string(((valid ? s : '�') for (s, valid) in zip(c, isvalid.(collect(c))))...)

    (d, transcode(String, b))
end

# I may need unit test later (?)

println(decode_unique_encoding("a"))
println(decode_unique_encoding("abc"))
println(decode_unique_encoding("Ġ"))
println(decode_unique_encoding("ÃĢ"))
println(decode_unique_encoding("Ã"))
println(decode_unique_encoding("ÃÃ"))
# hmm...
# Python has binary regex but Julia doesn't?
