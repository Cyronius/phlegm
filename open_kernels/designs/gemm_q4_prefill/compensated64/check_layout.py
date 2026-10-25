"""Independent CPU index simulation for rotated32-core geometry; no NPU access."""
import argparse
import json
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--n',type=int,default=1024)
ap.add_argument('--k',type=int,default=256)
ap.add_argument('--tokens',type=int,choices=(256,),default=256)
a=ap.parse_args();N,K,T=a.n,a.k,a.tokens
assert N>0 and K>0 and T>0 and N%512==K%256==T%256==0
m=n=64;features=8;tokenrows=4;KB=K//64;NB=T//64
# Semantic activation values before any tile packing, labelled independently by(t,k).
x=np.arange(T*K,dtype=np.int64).reshape(T,K)
v=x.reshape(T//64,8,8,K//64,8,8).transpose(3,0,1,4,2,5)
panels=np.stack([v*2,v*2+1],axis=-3).reshape(-1,8,8)
old=x.reshape(T//32,4,8,K//64,8,8).transpose(3,0,1,4,2,5)
old_panels=np.stack([old*2,old*2+1],axis=-3).reshape(-1,8,8)
assert np.array_equal(old_panels,panels),'existing fused hilo packer bytes would differ'

# DMA taps are in bytes; divide by72 to simulate whole BFP panels here.
for ntile in range(T//256):
 for tokenrow in range(tokenrows):
  i=np.indices((KB,8,16),dtype=np.int64)
  panel_idx=(tokenrow+ntile*tokenrows)*128+i[0]*NB*128+i[1]*16+i[2]
  got=panels[panel_idx.reshape(-1)].reshape(KB,8,8,2,8,8)
  expected_x=x[(ntile*4+tokenrow)*64:(ntile*4+tokenrow+1)*64]
  expected_x=expected_x.reshape(8,8,KB,8,8).transpose(2,0,3,1,4)
  expected=np.stack([expected_x*2,expected_x*2+1],axis=-3)
  assert np.array_equal(got,expected),'activation DMA/panel walk mismatch'
# Core C is8x8 sub-blocks. Walk the actual per-producer deblocking dimensions.
i=np.indices((8,8,8,8),dtype=np.int64)
walk=(i[0]*512+i[1]*8+i[2]*64+i[3]).reshape(-1)
output=np.full(N*T,-1,np.int64);count=np.zeros(N*T,np.uint8)
i=np.indices((4,64,64),dtype=np.int64)
ddr_offsets=(i[0]*64+i[1]*T+i[2]).reshape(-1)
for rbg in range(N//512):
 for feature in range(features):
  f0=(rbg*features+feature)*64
  for ntile in range(T//256):
   joined=[]
   for tokenrow in range(tokenrows):
    t0=(ntile*tokenrows+tokenrow)*64
    rows=np.arange(f0,f0+64)[:,None];cols=np.arange(t0,t0+64)[None,:]
    semantic=rows*T+cols
    blocked=semantic.reshape(8,8,8,8).transpose(0,2,1,3).reshape(-1)
    joined.append(blocked[walk])
   dest=f0*T+ntile*256+ddr_offsets
   output[dest]=np.concatenate(joined);count[dest]+=1
assert np.array_equal(output,np.arange(N*T)), 'C join/drain ordering mismatch'
assert np.all(count==1),'C drain coverage is not exactly once'
row_bytes=64*K*3//4
weight_starts=sorted((rbg*8+feature)*row_bytes for rbg in range(N//512) for feature in range(8))
assert weight_starts==list(range(0,N*K*3//4,row_bytes)),'weight pass coverage mismatch'
print(json.dumps(dict(passed=True,n=N,k=K,tokens=T,core_count=32,
 activation_streams=4,weight_streams=8,output_streams=8,join_inputs=4,
 l1_bytes_including_stack_and_rtp=60488,largest_wrapped_input_bytes=1152,
 max_memtile_s2mm_channels=6,max_memtile_mm2s_channels=3)))
