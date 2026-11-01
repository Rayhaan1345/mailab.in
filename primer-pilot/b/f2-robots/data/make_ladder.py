"""Offline step for F2: parameter and multiply-add counts for the model ladder. Run with python -I (torch, torchvision).
Counts multiply-adds (MACs) of convolution and linear layers for one forward pass; random weights are enough because only the architecture matters."""
import json, os
import torch, torch.nn as nn, torchvision
HERE = os.path.dirname(os.path.abspath(__file__))
def count(model, x):
    macs = [0]; hooks = []
    def conv_hook(m, i, o): macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def lin_hook(m, i, o): macs[0] += o.numel() // o.shape[0] * m.in_features
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
        if isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    return sum(p.numel() for p in model.parameters()), macs[0]
class Tag(nn.Module):                      # the C2 network
    def __init__(s):
        super().__init__(); s.conv1 = nn.Conv2d(1, 6, 3); s.conv2 = nn.Conv2d(6, 12, 3); s.fc = nn.Linear(300, 10)
    def forward(s, x):
        x = nn.functional.max_pool2d(torch.relu(s.conv1(x)), 2); x = nn.functional.max_pool2d(torch.relu(s.conv2(x)), 2); return s.fc(x.flatten(1))
out = {}
out["c2"] = count(Tag(), torch.zeros(1, 1, 28, 28))
out["mobilenet"] = count(torchvision.models.mobilenet_v2(), torch.zeros(1, 3, 224, 224))
out["resnet18"] = count(torchvision.models.resnet18(), torch.zeros(1, 3, 224, 224))
print(out)
# GPT-2 small from the D5 chapter: 12 blocks of 7,087,872 weights, token table 50,257 x 768 (reused as the output layer), positions 1,024 x 768, final norm 1,536
blocks = 12 * 7087872; table = 50257 * 768
out["gpt2"] = [blocks + table + 1024 * 768 + 1536, blocks + table]    # multiply-adds per token: every block weight once, plus the output layer (attention over the context not counted)
print("gpt2", out["gpt2"])
json.dump({k: {"params": v[0], "macs": v[1]} for k, v in out.items()}, open(f"{HERE}/counts.json", "w"), indent=1)
