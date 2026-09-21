import torch
import json
H=8
W=8
L=10
cube=torch.rand(H,W,L)
cube_mean=cube.mean(dim=(0,1))
print(cube_mean)

config=({
    "H":8,
    "W":8,
    "L":10
})

with open("config06.json","w",encoding="utf-8") as f:
    json.dump(config,f)

with open("config06.json","r",encoding="utf-8") as f:
    config_f=json.load(f)

assert config_f["H"]==H and config_f["W"]==W and config_f["L"]==L
print("形状对应")