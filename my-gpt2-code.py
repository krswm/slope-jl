# I learned a lot about the GPT-2 architecture from this blog post from January 2023.
# https://jaykmody.com/blog/gpt-from-scratch/

# I did not use generative AI for this file.

# 2026-07-07


"""
class GPT:
    def __init__(self):
        # Imagine we load the GPT-2 model here.
        self.tensors = get_tensors()

        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.GELU.html
        self.gelu = torch.nn.GELU()

        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Softmax.html
        self.softmax = torch.nn.Softmax(2)

    def gpt(self, ids):
        x = self.tensors["wte"][ids]
        x += self.tensors["wpe"][:len(ids), :]

        # Is num_blocks stored on metadata? I have to check later.
        for i in range(self.num_blocks):
            # Self attention
            y = self.layer_norm(
                x, self.tensors[f"h{i}/ln_1/b"], self.tensors[f"h{i}/ln_1/g"]
            )
            y @= self.tensors[f"h{i}/attn/c_attn/w"]
            y += self.tensors[f"h{i}/attn/c_attn/b"]
            # query, key, and value
            q, k, v = torch.split(y, 3)  # I'm not sure whether this syntax is correct...
            z = q @ k.transpose(0, 1)
            z /= torch.sqrt(self.num_embedding)
            z = self.softmax(z)
            z @= v
            z @= self.tensors[f"h{i}/mlp/c_proj/w"]
            z += self.tensors[f"h{i}/mlp/c_proj/b"]
            x += z

            # Feed forward
            y = self.layer_norm(
                x, self.tensors[f"h{i}/ln_2/b"], self.tensors[f"h{i}/ln_2/g"]
            )
            y @= self.tensors[f"h{i}/mlp/c_fc/w"]
            y += self.tensors[f"h{i}/mlp/c_fc/b"]
            y = self.gelu(y)
            y @= self.tensors[f"h{i}/mlp/c_proj/w"]
            y += self.tensors[f"h{i}/mlp/c_proj/b"]
            x += y

        x = self.LayerNorm(x, self.tensors["ln_f/b"], self.tensors["ln_f/g"])
        x @= self.tensors["wte"].transpose(0, 1)

        return x

        # It's not super complicated than I imagined before

    def LayerNorm(self, x, b, g):
"""

import codecs
import json
import math
import pprint

import safetensors
import torch


DEBUG_PRINT = False


def tprint2(checkpoint_name: str, tensor: torch.Tensor):
    """Tensor print"""

    print(
        f"\x1b[36m{checkpoint_name} {tensor.shape} "
        f"min:{tensor.min():f} max:{tensor.max():f}\x1b[39m"
    )
    print(tensor)


def tprint(checkpoint_name: str, tensor: torch.Tensor):
    """Tensor print"""

    if not DEBUG_PRINT:
        return

    tprint2(checkpoint_name, tensor)


class MyGPT2:
    def __init__(self, model_path: str) -> None:
        safetensors_path = f"{model_path}/model.safetensors"
        config_path = f"{model_path}/config.json"
        vocab_path = f"{model_path}/vocab.json"

        # "pt" stands for PyTorch.
        with safetensors.safe_open(safetensors_path, framework="pt") as file:
            self._tensors = {key: file.get_tensor(key) for key in file.keys()}

        if DEBUG_PRINT:
            print(f"\x1b[32mTensors loaded!\x1b[39m")
            for key, tensor in self._tensors.items():
                print(f"{key:32}{tensor.shape}")

        with open(config_path) as file:
            self._config = json.load(file)

        if DEBUG_PRINT:
            print(f"\x1b[32mConfig loaded!\x1b[39m")
            pprint.pprint(self._config)

        # Maybe this is causal mask?
        # print(self._tensors["h.0.attn.bias"])
        # Yes as I suspected!
        # But this matrix is 1 for unmasked and 0 for masked
        # not 0 for unmasked and -inf for masked
        # Maybe I have to use to multiply instead of adding?
        # At least I can ignore h.i.attn.bias
        # because it is just a simple triangular matrix
        # that doesn't contain any model information.

        with open(vocab_path) as file:
            self._id_to_token = {
                id: token for token, id in json.load(file).items()
            }

    def gpt(self, ids: list[int]) -> torch.Tensor:
        if DEBUG_PRINT:
            print(f"\x1b[32mInput\x1b[39m {ids=}")

        #### Input Embedding ####

        if DEBUG_PRINT:
            print(f"\x1b[32mInput embedding\x1b[39m")

        x = self._tensors["wte.weight"][ids]
        # tprint2("A", x)

        """
        assert(
            repr(self._tensors["wpe.weight"][:len(ids), :])
            == 
            repr(self._tensors["wpe.weight"][list(range(len(ids)))])
        )
        """

        x += self._tensors["wpe.weight"][: len(ids), :]
        # tprint2("B", x)

        #### Layers ####

        for i in range(self._config["n_layer"]):
            if DEBUG_PRINT:
                print(f"\x1b[32mLayer #{i}\x1b[39m")

            #### Attention ####

            y = self.LayerNorm(
                x, self._tensors[f"h.{i}.ln_1.weight"], self._tensors[f"h.{i}.ln_1.bias"]
            )
            # tprint2("C", y)

            y @= self._tensors[f"h.{i}.attn.c_attn.weight"]
            tprint("D", y)

            y += self._tensors[f"h.{i}.attn.c_attn.bias"]
            # tprint2("E", y)

            """
            print(type(y.split(3)))
            print([z.shape for z in y.split(3)])
            # print([z.shape for z in y.split(3, dim=1)])
            print([z.shape for z in y.split(self._config["n_embd"], dim=1)])
            """

            # Here, y [len_ids x 3n_embd] looks like
            #
            # (q11 ... q1N) ... (qH1 ... qHN) (k11 ... k1N) ... (kH1 ... kHN) (v11 ... v1N) ... (vH1 ... vHN)
            #
            # where H = n_head, N = n_embd / n_head

            # q, k, v = y.split(self._config["n_embd"], dim=1)
            q, k, v = y.chunk(3, dim=1)
            # tprint2("q", q)
            # tprint2("k", k)
            # tprint2("v", v)

            # Here, q has (q11 ... q1N) ... (qH1 ... qHN) etc

            """
            q_heads = q.split(self._config["n_head"], dim=1)
            k_heads = k.split(self._config["n_head"], dim=1)
            v_heads = v.split(self._config["n_head"], dim=1)
            """
            q_heads = q.chunk(self._config["n_head"], dim=1)
            k_heads = k.chunk(self._config["n_head"], dim=1)
            v_heads = v.chunk(self._config["n_head"], dim=1)
            """
            if i == 0:
                print("v_heads", v_heads[5])
            """

            # Here, q_heads has [(q11 ... q1N), ..., (qH1 ... qHN)]etc

            heads = [
                self._attention(q_, k_, v_, len(ids), tprint=i==0)
                for q_, k_, v_ in zip(q_heads, k_heads, v_heads)
            ]

            # tprint2("heads[7]", heads[7])

            # Now we have heads=[Attention1, Attention2, ..., AttentionH]
            
            '''
            N = self._config["n_embd"] // self._config["n_head"]
            heads = [
            	self._attention(
                    # Why did I wrote `i` instead of `j` here!?!?!?
            	    y[:, 0 * self._config["n_embd"] + N * j:0 * self._config["n_embd"] + N * j + N],  # q,
            	    y[:, 1 * self._config["n_embd"] + N * j:1 * self._config["n_embd"] + N * j + N],  # k,
            	    y[:, 2 * self._config["n_embd"] + N * j:2 * self._config["n_embd"] + N * j + N],  # v,
            	    len(ids),
            	    tprint=(j == 0 and i == 0),
            	) for j in range(self._config["n_head"])
            ]
            '''

            y = torch.hstack(heads)

            y @= self._tensors[f"h.{i}.attn.c_proj.weight"]
            # tprint2("L", y)

            y += self._tensors[f"h.{i}.attn.c_proj.bias"]

            x += y

            #### Feed Forward ####

            y = self.LayerNorm(
                x, self._tensors[f"h.{i}.ln_2.weight"], self._tensors[f"h.{i}.ln_2.bias"]
            )

            y @= self._tensors[f"h.{i}.mlp.c_fc.weight"]

            y += self._tensors[f"h.{i}.mlp.c_fc.bias"]

            """
            gelu = torch.nn.GELU()
            y = gelu(y)
            """
            y = 0.5 * y * (1.0 + ((2.0 / 3.141592)**0.5 * (y + 0.044715 * y**3)).tanh())

            y @= self._tensors[f"h.{i}.mlp.c_proj.weight"]
            # tprint2("S", y)

            y += self._tensors[f"h.{i}.mlp.c_proj.bias"]
            # tprint2("T", y)

            x += y
            if i == 0:
                tprint2("B", x)

        #### Output embedding ####
        # tprint2("xb after decoder stack", x)

        if DEBUG_PRINT:
            print(f"\x1b[32mOutput embedding\x1b[39m")

        x = self.LayerNorm(
            x, self._tensors["ln_f.weight"], self._tensors["ln_f.bias"]
        )
        # tprint2("U", x)

        x @= self._tensors["wte.weight"].transpose(0, 1)
        # tprint2("V", x)

        return x

        # Done. Kind of.

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, len_ids: int, *, tprint=False
    ) -> torch.Tensor:
        y = q
        # tprint("G", y)

        y @= k.T
        # tprint("H", y)

        y /= q.shape[-1]**0.5
        # tprint("I", y)
        ## Yes! Now my code generates the same as the original!

        y += (torch.ones(len_ids, len_ids) * -1e12).triu(diagonal=1)
        # tprint("II", y)

        # y = y.softmax(1)
        # softmax with the max trick
        e = (y - torch.amax(y, -1, keepdims=True)).exp()
        '''
        if tprint:
            tprint2("e", e)
        '''
        
        y = e / e.sum(-1, keepdims=True)

        # tprint("J", y)

        y @= v
        # tprint("K", y)

        return y

    def LayerNorm(self, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, tprint2=False) -> torch.Tensor:
        """
        ln = torch.nn.LayerNorm(self._config["n_embd"])
        ln.weight = torch.nn.Parameter(weight)
        ln.bias = torch.nn.Parameter(bias)
        if DEBUG_PRINT:
            print("ln", ln)
        return ln(y)
        """

        if tprint2:
            print("xb_mean")
            print(y.mean(-1, keepdim=True))
            print("xb_diff")
            print(y - y.mean(-1, keepdim=True))
            print("xb_fluct")
            print((y - y.mean(-1, keepdim=True)) ** 2.0)
            print("xb_var")
            print(y.var(-1, keepdim=True, correction=0.0))
            print("xb_purt")
            print(y.var(-1, keepdim=True, correction=0.0) + 1e-5)
            print("xb_division")
            print((y - y.mean(-1, keepdim=True)) / (y.var(-1, keepdim=True, correction=0.0) + 1e-5).sqrt())


        return weight * (y - y.mean(-1, keepdim=True)) / (y.var(-1, keepdim=True, correction=0.0) + 1e-5).sqrt() + bias

    def generate(self, ids: list[int]) -> list[int]:
        print(
            "\x1b[1m" + (
                b"".join(self.replace_characters(self._id_to_token[id]) for id in ids).decode()
            ) + "\x1b[22m",
            end="",
            flush=True,
        )

        decoder = codecs.getincrementaldecoder("utf-8")()
    
        for _ in range(100):
            a = self.gpt(ids)
            # print([(a[i].argmax(), self._id_to_token[int(a[i].argmax())]) for i in range(len(ids))])
            next_id = int(a[-1].argmax())
            # print(next_id, type(next_id))
            ids.append(next_id)

            """
            print(
                "".join(
                    f"{id}\x1b[7m{self._id_to_token[id]}\x1b[27m"
                for id in ids)
            )
            """
            # Wow! At least something generated!!!

            replaced = self.replace_characters(self._id_to_token[next_id])
            decoded = decoder.decode(input=replaced)

            print(
                f"\x1b[1;35m{decoded}\x1b[22;39m",
                end="",
                flush=True,
            )

            print(f"\x1b[2m({next_id})\x1b[22m", end="", flush=True)

        """
        return "".join(
            self.replace_characters(self._id_to_token[id]) for id in ids
        )
        """

    def replace_characters(self, text: str) -> bytes:
        # return text.replace("Ġ", " ").replace("Ċ", "\n")

        # hex(ord("Ġ")) == 0x120
        # hex(ord(" ")) == 0x20

        # hex(ord("Ċ")) == 0x10a
        # hex(ord("\n")) == 0xa

        # plus 0x100?

        # print(f"\x1b[31m{text}\x1b[39m")

        """
        return "".join(
            (chr(ord(char) - 0x100) if ord(char) >= 0x100 else char) for char in text
        )
        """

        # "あ" -> "ãģĤ"

        # ã = U+00E3
        # ģ = U+0123
        # Ĥ = U+0124

        # あ = U+3042
        # あ = 0xE3 0x81 0x82 in UTF-8

        # ã = U+00E3 - 0 = 0xE3
        # ģ = U+0123 - 0x0a2 = 0x81
        # Ĥ = U+0124 - 0x0a2 = 0x82

        raw = bytes(
            [
                ord(char) - 0xa2 if 0x122 <= ord(char) <= 0x142  # The range is just my guess!
                else ord(char) - 0x100 if ord(char) >= 0x100
                else ord(char)
                for char in text
            ]
        )

        return raw


# print((torch.ones(4, 4) * -1e12).triu(diagonal=1))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("ids", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    my_gpt2 = MyGPT2(args.model_path)
    if args.test:
        my_gpt2.gpt([40, 1842, 19617, 13])
    else:
        text = my_gpt2.generate([int(id) for id in args.ids.split()])
        # print(f"\x1b[1;35m{text}\x1b[22;39m")
