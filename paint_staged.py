"""Experimental staged Hunyuan3D paint; lower-memory test, not production."""
import os, sys, json, gc, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python/paint'))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
RAM_GB=float(os.getenv('HY3D_RAM_GB','0'))
if RAM_GB:
 mx.set_memory_limit(int(RAM_GB * 0.9 * 1024**3))
mx.set_cache_limit(64 * 1024 * 1024 if RAM_GB else 256 * 1024 * 1024)
import trimesh, xatlas
from PIL import Image
from safetensors import safe_open
from hy3dpaint_mlx.vae import AutoencoderKL
from hy3dpaint_mlx.unet2p5d import UNet2p5DConditionModel
from hy3dpaint_mlx.scheduler import UniPCScheduler
from hy3dpaint_mlx.mesh_render import MeshRender
import hy3dpaint_mlx.mesh_render as mesh_render
from hy3dpaint_mlx.convert import load_torch_weights
from hy3dpaint_mlx.attention import Attention, FeedForward
from mlx.utils import tree_unflatten

W=ROOT/'models/paint-small'
DIR=Path(os.getenv('PAINT_WORK',str(ROOT/'work/paint-experiment'))).resolve()
DIR.mkdir(parents=True,exist_ok=True)
MESH=Path(os.getenv('PAINT_MESH',''))
IMAGE=Path(os.getenv('PAINT_IMAGE',''))
R=int(os.getenv('PAINT_RES','256')); STEPS=int(os.getenv('PAINT_STEPS','15'))
AZIMS=[0,90,180,270,0,180]; ELEVS=[0,0,0,0,90,-90]; SF=0.18215
TEX=int(os.getenv('PAINT_TEX','1024'))
TOKEN_CHUNK=int(os.getenv('PAINT_TOKEN_CHUNK','256'))
VIEW_CHUNK=int(os.getenv('PAINT_VIEW_CHUNK','1'))
SHADOW_BIAS=float(os.getenv('PAINT_SHADOW_BIAS','0'))
SR_SCALE=int(os.getenv('PAINT_SR_SCALE','1'))
SR_TILE=int(os.getenv('PAINT_SR_TILE','128'))
GUIDANCE=float(os.getenv('PAINT_GUIDANCE','2')); REF_SCALE=float(os.getenv('PAINT_REF_SCALE','1')); SEED=int(os.getenv('PAINT_SEED','0'))

# Convolutions do not mix batch elements, so processing view batches in slices
# preserves the operation. Multiview attention remains global across all views.
_original_conv = nn.Conv2d.__call__
def _chunked_conv(self, x):
 if x.ndim != 4 or x.shape[0] <= VIEW_CHUNK:
  return _original_conv(self, x)
 parts=[]
 for start in range(0,x.shape[0],VIEW_CHUNK):
  part=_original_conv(self,x[start:start+VIEW_CHUNK]); mx.eval(part); parts.append(part)
 return mx.concatenate(parts,axis=0)
nn.Conv2d.__call__ = _chunked_conv

# Attention queries are independent given the full set of keys and values.
# Splitting queries preserves all-view attention rather than generating each
# view independently. The feed-forward layers are token-wise too.
def _chunked_attention(self, x, context=None):
 context=x if context is None else context
 b,n,_=x.shape; m=context.shape[1]; h,d=self.heads,self.dim_head
 q=self.to_q(x).reshape(b,n,h,d).transpose(0,2,1,3)
 k=self.to_k(context).reshape(b,m,h,d).transpose(0,2,1,3)
 v=self.to_v(context).reshape(b,m,h,d).transpose(0,2,1,3)
 mx.eval(q,k,v)
 chunks=[]
 for start in range(0,n,TOKEN_CHUNK):
  part=mx.fast.scaled_dot_product_attention(q[:,:,start:start+TOKEN_CHUNK,:],k,v,scale=self.scale)
  mx.eval(part);chunks.append(part)
 output=mx.concatenate(chunks,axis=2).transpose(0,2,1,3).reshape(b,n,h*d)
 return self.to_out[0](output)
def _chunked_feedforward(self,x):
 if x.shape[1] <= TOKEN_CHUNK:
  return self.net[2](self.net[0](x))
 chunks=[]
 for start in range(0,x.shape[1],TOKEN_CHUNK):
  part=self.net[2](self.net[0](x[:,start:start+TOKEN_CHUNK]));mx.eval(part);chunks.append(part)
 return mx.concatenate(chunks,axis=1)
Attention.__call__ = _chunked_attention
FeedForward.__call__ = _chunked_feedforward

def log(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)

def vae_load():
 v=AutoencoderKL.from_config(json.load(open(W/'vae/config.json')))
 load_torch_weights(v,mx.load(str(W/'vae/diffusion_pytorch_model.safetensors')),renames=[('.to_out.0.','.to_out.')])
 return v

def load_prefix(module, prefix):
 """Map only the requested network's fp16 tensors, not both networks at once."""
 items=[]
 with safe_open(str(W/'unet/diffusion_pytorch_model.safetensors'),framework='np') as f:
  for k in f.keys():
   if not k.startswith(prefix): continue
   key=k[len(prefix):].replace('transformer_blocks.0.transformer.','transformer_blocks.0.')
   arr=mx.array(f.get_tensor(k))
   if arr.ndim==4: arr=arr.transpose(0,2,3,1)
   items.append((key,arr))
 module.update(tree_unflatten(items))
 mx.eval(module.parameters())
 log(f'Loaded {prefix} ({len(items)} tensors)')

def stage_prepare():
 if (DIR/'mesh-data.npz').exists():
  log('Reusing unwrapped mesh')
  data=np.load(DIR/'mesh-data.npz'); V,F,uvs=data['V'],data['F'],data['uvs']
 else:
  log('Unwrapping mesh')
  mesh=trimesh.load(str(MESH),force='mesh')
  vm,indices,uvs=xatlas.parametrize(mesh.vertices,mesh.faces)
  V=np.asarray(mesh.vertices)[vm]; F=indices.astype(np.int64)
  np.savez_compressed(DIR/'mesh-data.npz',V=V,F=F,uvs=uvs)
 mesh_render._rasterize=mesh_render.cr_raster.rasterize_gpu
 renderer=MeshRender();renderer.load_mesh(V,F);renderer.set_uv(uvs,F)
 v=vae_load()
 def enc(image):
  x=mx.array(np.asarray(image,dtype=np.float16)[None])*2-1
  out=v.encode_mean(x)*SF
  mx.eval(out)
  return np.asarray(out[0])
 normals=[]; positions=[]
 for i,(e,a) in enumerate(zip(ELEVS,AZIMS)):
  log(f'Encoding view {i+1}/6')
  n,p=renderer.render_control(e,a,R)
  normals.append(enc(n)); positions.append(enc(p));mx.clear_cache()
 pil=Image.open(IMAGE).resize((R,R)).convert('RGBA')
 bg=Image.new('RGB',pil.size,(255,255,255));bg.paste(pil,mask=pil.getchannel('A'))
 ref=enc(np.asarray(bg,dtype=np.float32)/255.0)
 np.savez_compressed(DIR/'controls.npz',normal=np.stack(normals),position=np.stack(positions),ref=ref)
 log('Controls ready')

def stage_reference():
 log('Preparing reference features')
 c=np.load(DIR/'controls.npz'); ref=mx.array(c['ref'])[None,None]
 net=UNet2p5DConditionModel(json.load(open(W/'unet/config.json')))
 load_prefix(net.unet_dual,'unet_dual.')
 # The cached reference prompt belongs to the main network and is the only main weight needed here.
 with safe_open(str(W/'unet/diffusion_pytorch_model.safetensors'),framework='np') as f:
  net.unet.learned_text_clip_ref=mx.array(f.get_tensor('unet.learned_text_clip_ref'))
 ced=net.compute_condition_embed(ref)
 mx.eval(list(ced.values()))
 mx.save_safetensors(str(DIR/'reference.safetensors'),ced)
 log('Reference features saved')

def stage_denoise():
 log('Loading main paint network')
 c=np.load(DIR/'controls.npz'); n=mx.array(c['normal'])[None]; p=mx.array(c['position'])[None];ref=mx.array(c['ref'])[None,None]
 ced=mx.load(str(DIR/'reference.safetensors'))
 net=UNet2p5DConditionModel(json.load(open(W/'unet/config.json')))
 load_prefix(net.unet,'unet.')
 gen=net.unet.learned_text_clip_gen;neg=mx.zeros_like(gen);zero=mx.zeros_like(ref)
 N=6; h=R//8; cam=mx.array(np.arange(N)[None,:].astype(np.int32));cref=mx.array([[0]])
 sched=UniPCScheduler();sched.set_timesteps(STEPS);mx.random.seed(SEED)
 z=(mx.random.normal((1,N,h,h,4))*sched.init_noise_sigma()).astype(mx.float16)
 for i,t in enumerate(sched.timesteps):
  t=int(t)
  vc=net(z,t,gen,n,p,ref,cam,cref,mva_scale=1.0,ref_scale=REF_SCALE,condition_embed_dict=ced);mx.eval(vc)
  vu=net(z,t,neg,n,p,zero,cam,cref,mva_scale=1.0,ref_scale=0.0,condition_embed_dict=None);mx.eval(vu)
  z=sched.step(vu+GUIDANCE*(vc-vu),t,z).astype(mx.float16);mx.eval(z)
  log(f'Denoise {i+1}/{STEPS}')
 np.save(DIR/'latents.npy',np.asarray(z))
 log('Denoising complete')

def load_view_strip():
 strip=np.asarray(Image.open(DIR/'paint-views.png').convert('RGB'),dtype=np.float32)/255.0
 size=strip.shape[0]
 if strip.shape[1] != size*6: raise ValueError('Saved texture views are invalid')
 return [strip[:,i*size:(i+1)*size] for i in range(6)]

def stage_decode():
 log('Decoding views one at a time')
 z=np.load(DIR/'latents.npy');v=vae_load();views=[]
 for i in range(6):
  d=v.decode(mx.array(z[0,i:i+1])/SF);mx.eval(d)
  views.append(np.clip((np.asarray(d[0])+1)/2,0,1));mx.clear_cache()
 Image.fromarray((np.concatenate(views,axis=1)*255).astype(np.uint8)).save(DIR/'paint-views.png')
 log('Views saved')

def stage_superres():
 from hy3dpaint_mlx.realesrgan import RRDBNet
 from mlx.utils import tree_unflatten
 if SR_SCALE not in (2,4) or SR_TILE not in (64,128,256): raise ValueError('Invalid detail settings')
 log('Loading texture detail model')
 model=RRDBNet()
 weights=mx.load(str(W/'realesrgan/rrdbnet_mlx.safetensors'))
 model.update(tree_unflatten([(k,v.astype(mx.float16)) for k,v in weights.items()]))
 mx.eval(model.parameters());del weights
 source=Image.open(DIR/'paint-views.png').convert('RGB'); size=source.height
 if source.width != size*6: raise ValueError('Saved texture views are invalid')
 for i in range(6):
  log(f'Enhancing view {i+1}/6')
  view=np.asarray(source.crop((i*size,0,(i+1)*size,size)),dtype=np.float16)/np.float16(255)
  out=np.empty((size*4,size*4,3),dtype=np.uint8)
  for y in range(0,size,SR_TILE):
   for x in range(0,size,SR_TILE):
    y1=min(y+SR_TILE,size);x1=min(x+SR_TILE,size)
    py=max(0,y-10);px=max(0,x-10);py1=min(size,y1+10);px1=min(size,x1+10)
    tile=model(mx.array(view[py:py1,px:px1])[None])[0]
    crop=tile[(y-py)*4:(y1-py)*4,(x-px)*4:(x1-px)*4]
    mx.eval(crop)
    out[y*4:y1*4,x*4:x1*4]=(np.clip(np.asarray(crop),0,1)*255).astype(np.uint8)
    del tile,crop
    mx.clear_cache()
  img=Image.fromarray(out)
  if SR_SCALE==2: img=img.resize((size*2,size*2),Image.Resampling.LANCZOS)
  img.save(DIR/f'upscaled-view-{i}.png')
  del out,img
 log('Enhanced views saved')

def stage_finish(reuse_views=False):
 c=np.load(DIR/'mesh-data.npz');V,F,uvs=c['V'],c['F'],c['uvs']
 if SR_SCALE in (2,4):
  log('Using enhanced views')
  views=[np.asarray(Image.open(DIR/f'upscaled-view-{i}.png').convert('RGB'),dtype=np.float16)/np.float16(255) for i in range(6)]
 else:
  log('Using saved texture views')
  views=load_view_strip()
 log('Baking texture')
 mesh_render._rasterize=mesh_render.cr_raster.rasterize_gpu
 rend=MeshRender();rend.load_mesh(V,F);rend.set_uv(uvs,F)
 texture,covered=rend.bake(views,ELEVS,AZIMS,texture_size=TEX,weights=[1.,.1,.5,.1,.05,.05],shadow_bias=SHADOW_BIAS)
 texture=rend.inpaint(texture,covered)
 uv=rend.vtx_uv.copy();uv[:,1]=1-uv[:,1]
 out=trimesh.Trimesh(vertices=V,faces=F,process=False,visual=trimesh.visual.TextureVisuals(uv=uv,image=Image.fromarray((np.clip(texture,0,1)*255).astype(np.uint8))))
 out.export(DIR/'painted.glb')
 log('Painted GLB saved')

if __name__=='__main__':
 {'prepare':stage_prepare,'reference':stage_reference,'denoise':stage_denoise,'decode':stage_decode,'superres':stage_superres,'finish':stage_finish,'rebake':lambda:stage_finish(reuse_views=True)}[sys.argv[1]]()
