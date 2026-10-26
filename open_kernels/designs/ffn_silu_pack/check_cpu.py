"""Independent host oracle and prototype-native CPU checks. No NPU calls."""
import argparse,ctypes,hashlib,json,os
from pathlib import Path
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def bf(v):
 a=np.asarray(v,dtype=np.float32).view(np.uint32)
 return ((a+np.uint32(0x7fff)+((a>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def encode(v):
 # Independent floating-point scalar oracle; no production bit-shift mantissa routine.
 v=np.asarray(v,dtype=np.float32);a=float(np.max(np.abs(v)))
 if a==0:return bytes(9),np.zeros(8,dtype=np.float32)
 e=max(7,int(np.float32(a).view(np.uint32))>>23)
 q=np.rint(v.astype(np.float64)*2.0**(133-e)).astype(np.int32)
 if np.any(q>127):
  e+=1;q=np.rint(v.astype(np.float64)*2.0**(133-e)).astype(np.int32)
 assert np.all((q>=-128)&(q<=127))
 return bytes([e])+q.astype(np.int8).tobytes(),(q.astype(np.float64)*2.0**(e-133)).astype(np.float32)
def pack8(v):
 v=bf(v);hi,d=encode(v);lo,_=encode(bf(v-d));return hi+lo
def oracle(g,u):
 # Full block [64,32] -> [token8][K8][row][hi,lo planes].
 out=bytearray(4608)
 ref=(g.astype(np.float64)/(1+np.exp(-g.astype(np.float64)))*u.astype(np.float64)).astype(np.float32)
 for nt in range(4):
  for ki in range(8):
   for row in range(8):
    pair=pack8(ref[ki*8:ki*8+8,nt*8+row]);off=(nt*8+ki)*144+row*9
    out[off:off+9]=pair[:9];out[off+72:off+81]=pair[9:]
 return ref,bytes(out)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT/'target'):p.error('artifacts must stay under target/')
 out.mkdir(parents=True,exist_ok=False)
 wrapper=out/'wrapper.cc';wrapper.write_text('#include "encoder.h"\nextern "C" { void ffn_silu_pack(const float*,const float*,uint8_t*); void pack(const float*x,uint8_t*y){fsp::pack8(x,y,y+9);} float act(float g,float u){return fsp::silu_up(g,u);} }\n')
 cxx=os.environ.get('CXX','g++');cmd=[cxx,'-std=c++17','-shared','-fPIC','-O2','-ffp-contract=off','-I'+str(HERE),str(wrapper),str(HERE/'kernel.cc'),'-o',str(out/'probe.so')]
 subprocess.run(cmd,check=True)
 lib=ctypes.CDLL(str(out/'probe.so'));fp=ctypes.POINTER(ctypes.c_float);bp=ctypes.POINTER(ctypes.c_uint8)
 lib.pack.argtypes=[fp,bp];lib.act.argtypes=[ctypes.c_float,ctypes.c_float];lib.act.restype=ctypes.c_float;lib.ffn_silu_pack.argtypes=[fp,fp,bp]
 rng=np.random.default_rng(212);fixtures=[np.zeros(8,np.float32),np.full(8,1.9921875,np.float32),np.full(8,-1.9921875,np.float32)]
 for scale in (1e-38,1e-20,1e-4,1,32,1e18):fixtures.extend((rng.normal(size=(64,8))*scale).astype(np.float32))
 # BF16 ties and shared-exponent positive rollover, including signs and zero lanes.
 for bits in (0x3f808000,0x3f818000,0x3fff0000,0x3fff8000,0xbfFF0000,1,0x80000001):
  fixtures.append(np.array([bits,0,0x3f000000,0xbf000000]*2,dtype=np.uint32).view(np.float32))
 for x in fixtures:
  y=np.zeros(18,np.uint8);lib.pack(x.ctypes.data_as(fp),y.ctypes.data_as(bp));assert y.tobytes()==pack8(x),x
 cases=[]
 for seed in (212,213,214):
  r=np.random.default_rng(seed);g=(r.normal(size=(64,32))*4).astype(np.float32);u=(r.normal(size=(64,32))*3).astype(np.float32)
  assert np.all(np.isfinite(g)) and np.max(np.abs(g))<=1e8 and np.max(np.abs(u))<=1e8
  ref,refpacked=oracle(g,u)
  candidate=np.array([lib.act(float(x),float(y)) for x,y in zip(g.ravel(),u.ravel())],np.float32).reshape(64,32)
  native=np.zeros(4608,np.uint8);lib.ffn_silu_pack(g.ctypes.data_as(fp),u.ctypes.data_as(fp),native.ctypes.data_as(bp))
  # Exact packing is tested against candidate values, independently from SiLU approximation.
  _,expected=oracle(np.zeros_like(g),np.zeros_like(u));packed=bytearray(expected)
  for nt in range(4):
   for ki in range(8):
    for row in range(8):
     pair=pack8(candidate[ki*8:ki*8+8,nt*8+row]);off=(nt*8+ki)*144+row*9
     packed[off:off+9]=pair[:9];packed[off+72:off+81]=pair[9:]
  assert native.tobytes()==bytes(packed)
  fp32ref=(g/(np.float32(1)+np.exp(-g))*u).astype(np.float32)
  err=float(np.linalg.norm(candidate.astype(np.float64)-ref)/np.linalg.norm(ref));err32=float(np.linalg.norm(candidate.astype(np.float64)-fp32ref)/np.linalg.norm(fp32ref))
  assert err<=.02 and err32<=.02
  cases.append(dict(seed=seed,relative_error_double_silu=err,relative_error_fp32_silu=err32,pack_exact_for_candidate=True,packed_matches_exact_silu=native.tobytes()==refpacked))
  g.tofile(out/f'gate-{seed}.bin');u.tofile(out/f'up-{seed}.bin');native.tofile(out/f'candidate-{seed}.bin')
 # Saturation/negative-underflow policy, and finite large values within preflight domain.
 for g,u in ((-80.,1.),(-1e8,1e8),(1e8,1e8),(0.,1e8)):
  y=lib.act(g,u);assert np.isfinite(y) and abs(y)<=1e20
 report=dict(passed=True,encoder_cases=len(fixtures),cases=cases,operator_relative_error_limit=.02,device_validation='not performed',compiler_command=cmd,
  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'encoder.h',HERE/'kernel.cc',Path(__file__))})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
