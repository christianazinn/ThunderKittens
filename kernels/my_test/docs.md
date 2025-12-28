ops:
- fused pre layernorm + x_* vector weight prep
- full RWKVAG computations (? - maybe split)
- actual rwkv kernel
- GN and O proj + residual
- fused pre mlp layernorm + first stage mlp
- mlp downproj + residual
- out layernorm + embedding

