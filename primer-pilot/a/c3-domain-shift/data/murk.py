"""Torch version of depth.py's model (colour loss, darkening with auto exposure, haze), vectorised for batches of RGB images in [0,1]."""
import torch, numpy as np
A_WATER = torch.tensor([0.34, 0.0565, 0.0092]); K_RGB = torch.tensor([0.54, 0.256, 0.209])   # per metre, as in depth.py (rounded to 3 decimals there)
BETA = 0.06; VEIL = torch.tensor([0.08, 0.50, 0.55]); EXPOSURE = 0.6
def s2l(x): return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
def l2s(x): x = x.clamp(0, 1); return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
def degrade(x, z, k=K_RGB, beta=BETA, veil=VEIL, exposure=EXPOSURE):
    """x: (N,3,H,W) in [0,1]; z: float or (N,) depths in metres; k, beta, exposure may be (N,...) tensors."""
    n = x.shape[0]
    z = torch.as_tensor(z, dtype=torch.float32).expand(n).view(n, 1, 1, 1) if not torch.is_tensor(z) or z.dim() == 0 else z.view(n, 1, 1, 1)
    k = k.view(1, 3, 1, 1) if k.dim() == 1 else k.view(n, 3, 1, 1)
    beta = torch.as_tensor(beta, dtype=torch.float32); beta = beta.view(-1, 1, 1, 1) if beta.dim() > 0 else beta
    exposure = torch.as_tensor(exposure, dtype=torch.float32); exposure = exposure.view(-1, 1, 1, 1) if exposure.dim() > 0 else exposure
    lin = s2l(x)
    v = 1 - torch.exp(-beta * z)
    mixed = lin * (1 - v) + veil.view(1, 3, 1, 1) * v
    kmin = k.min(1, keepdim=True).values; kmean = k.mean(1, keepdim=True)
    colour_t = torch.exp(-(k - kmin) * z); dark_t = torch.exp(-kmin * z); gain = torch.exp(exposure * kmean * z)
    return l2s(mixed * colour_t * dark_t * gain)
if __name__ == "__main__":
    import importlib.util
    from PIL import Image
    R = "../../data/sharks/"   # relative to this folder
    spec = importlib.util.spec_from_file_location("depth", R + "depth.py"); depth = importlib.util.module_from_spec(spec); spec.loader.exec_module(depth)
    img = Image.open(R + "photos/milk-2.jpg").convert("RGB")
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    for z in (0, 5, 12.5, 25):
        a = np.asarray(depth.degrade(img, z), dtype=np.int32)
        b = (degrade(x, z)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.int32)
        print(z, "max abs diff", np.abs(a - b).max(), "mean", np.abs(a - b).mean().round(3))
