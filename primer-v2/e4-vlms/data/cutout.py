"""Cuts the spadenose shark out of spadenose-1 (a local-background threshold) for the counting composites of make_vlm2.py. Run: python -I cutout.py"""
import numpy as np
from PIL import Image, ImageFilter
R="../../data/sharks/photos/"
im=Image.open(R+"spadenose-1.jpg").convert("RGB")
g=im.convert("L")
bgl=g.filter(ImageFilter.MaxFilter(41)).filter(ImageFilter.GaussianBlur(14))
d=np.asarray(bgl).astype(float)-np.asarray(g).astype(float)
m=(d>22)
box=(70,135,585,300)
mm=np.zeros_like(m); x0,y0,x1,y1=box; mm[y0:y1,x0:x1]=m[y0:y1,x0:x1]
mk=Image.fromarray((mm*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(3))
mk=np.asarray(mk)>128

# cutout with soft edge
rgba=np.dstack([np.asarray(im), (np.asarray(Image.fromarray(mk.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(1.2)))).astype(np.uint8)])
ys,xs=np.where(mk); print(xs.min(),xs.max(),ys.min(),ys.max())
c=Image.fromarray(rgba,"RGBA").crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
c.save("cutout.png"); print(c.size)

