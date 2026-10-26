"""Prepare/check a standalone harness fixture. This script never opens the NPU."""
import argparse,ctypes,hashlib,json
from pathlib import Path
import numpy as np
from check_cpu import bf
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode_rows(v):
    v=np.asarray(v,np.float32)
    e=np.maximum(7,np.max(np.abs(v).view(np.uint32),axis=-1)>>23).astype(np.int32)
    q=np.rint(np.ldexp(v.astype(np.float64),133-e[...,None])).astype(np.int32)
    e=e+np.any(q>127,axis=-1)
    q=np.rint(np.ldexp(v.astype(np.float64),133-e[...,None])).astype(np.int32)
    assert np.all((q>=-128)&(q<=127))
    e=np.where(np.all(v==0,axis=-1),0,e)
    out=np.concatenate((e[...,None].astype(np.uint8),q.astype(np.int8).view(np.uint8)),axis=-1)
    decoded=np.ldexp(q.astype(np.float64),e[...,None]-133).astype(np.float32)
    return out,decoded

def independent_pack(values):
    # values are [F,T]; gather the host encoder's per-token, eight-K-value blocks.
    f,t=values.shape
    v=values.T.reshape(t//32,4,8,f//64,8,8).transpose(3,0,1,4,2,5).copy()
    v=bf(v);hi,decoded=encode_rows(v);lo,_=encode_rows(bf(v-decoded))
    return np.stack((hi,lo),axis=4).reshape(-1)

def decode(packed,f,t=256):
    blocks=packed.reshape(f//64,t//32,4,8,2,8,9)
    e=blocks[...,0].astype(np.int32)
    q=blocks[...,1:].copy().view(np.int8).astype(np.float64)
    values=np.ldexp(q,e[...,None]-133).sum(axis=4)
    return values.transpose(1,2,4,0,3,5).reshape(t,f).T

def relative(a,b):
    return float(np.linalg.norm(a.astype(np.float64)-b)/max(np.linalg.norm(b),1e-30))

def prepare(a):
    out=a.out.resolve();build=a.build.resolve();cpu=a.cpu_probe.resolve()
    if not out.is_relative_to(ROOT/'target'):raise ValueError('fixtures must stay under ignored target/')
    meta=json.loads((build/'build.json').read_text());assert meta['complete']
    f=meta['features'];assert meta['tokens']==256
    source={p.name:digest(p) for p in (HERE/'encoder.h',HERE/'kernel.cc',HERE/'check_cpu.py')}
    cpumeta=json.loads((cpu/'report.json').read_text());assert cpumeta['passed'] and cpumeta['source_sha256']==source
    for path,sha in meta['source_sha256'].items():assert digest(ROOT/path)==sha
    out.mkdir(parents=True,exist_ok=False)
    r=np.random.default_rng(a.seed)
    g=(r.normal(size=(f,256))*4).astype(np.float32);u=(r.normal(size=(f,256))*3).astype(np.float32)
    # Caller preflight contract; unvalidated device input is not supported.
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(u)) and np.max(np.abs(g))<=1e8 and np.max(np.abs(u))<=1e8
    exact=(g.astype(np.float64)/(1+np.exp(-g.astype(np.float64)))*u.astype(np.float64)).astype(np.float32)
    fp32=(g/(np.float32(1)+np.exp(-g))*u).astype(np.float32)
    assert np.all(np.isfinite(exact)) and np.max(np.abs(exact))<=1e20
    lib=ctypes.CDLL(str(cpu/'probe.so'));ptr=ctypes.POINTER(ctypes.c_float);bp=ctypes.POINTER(ctypes.c_uint8)
    lib.ffn_silu_pack.argtypes=[ptr,ptr,bp]
    # CPU kernel reference for all blocks; preserve exact native [F,T] gather offsets.
    candidate=np.empty(f*256*9//4,np.uint8)
    for kb in range(f//64):
        for nb in range(8):
            gg=np.ascontiguousarray(g[kb*64:kb*64+64,nb*32:nb*32+32]);uu=np.ascontiguousarray(u[kb*64:kb*64+64,nb*32:nb*32+32])
            dst=candidate[(kb*8+nb)*4608:(kb*8+nb+1)*4608]
            lib.ffn_silu_pack(gg.ctypes.data_as(ptr),uu.ctypes.data_as(ptr),dst.ctypes.data_as(bp))
    # An independent vectorized polynomial evaluation checks FP32 arithmetic separately.
    aa=np.abs(g);x=-aa*np.float32(1.4426950408889634);n=np.floor(x).astype(np.int32)
    rr=(x-n.astype(np.float32))*np.float32(.6931471805599453)
    poly=np.float32(1)+rr*(np.float32(1)+rr*(np.float32(.5)+rr*(np.float32(1/6)+rr*(np.float32(1/24)+rr*np.float32(1/120)))))
    z=poly*np.ldexp(np.ones_like(poly),n);ss=np.float32(1)/(np.float32(1)+z);ss=np.where(g<0,z*ss,ss)
    candidate_values=(g*ss)*u
    assert np.array_equal(candidate,independent_pack(candidate_values)), 'CPU kernel packing or indexing mismatch'
    metrics=dict(approximation_vs_double=relative(candidate_values,exact),approximation_vs_fp32=relative(candidate_values,fp32),packed_operator_vs_double=relative(decode(candidate,f),exact))
    assert max(metrics.values())<=.02
    arrays={'gate.bin':g,'up.bin':u,'expected-candidate.bin':candidate,'exact-double-values.bin':exact,'exact-fp32-values.bin':fp32,
            'expected-double-packed.bin':independent_pack(exact),'expected-fp32-packed.bin':independent_pack(fp32)}
    for name,value in arrays.items():value.tofile(out/name)
    cfg=['device',f'xclbin F {build}/build/final.xclbin',f'kernelx k F {build}/build/insts.bin',
         f'buf g {g.nbytes} {out}/gate.bin',f'buf u {u.nbytes} {out}/up.bin',f'buf p {candidate.nbytes}',
         'run k g u p',f'dump p {out}/device-packed.bin {candidate.nbytes}','']
    (out/'run.cfg').write_text('\n'.join(cfg))
    record=dict(features=f,tokens=256,seed=a.seed,operator_relative_error_limit=.02,metrics=metrics,pack_exact_for_cpu_candidate=True,device_validation='not performed',
                source_sha256=source,build_json_sha256=digest(build/'build.json'),files={name:digest(out/name) for name in arrays})
    (out/'fixture.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

def check(a):
    out=a.out.resolve();meta=json.loads((out/'fixture.json').read_text())
    for name,sha in meta['files'].items():assert digest(out/name)==sha
    f=meta['features'];raw=np.fromfile(out/'device-packed.bin',np.uint8);expected=np.fromfile(out/'expected-candidate.bin',np.uint8)
    assert raw.size==expected.size,'wrong output size'
    same=np.array_equal(raw,expected)
    double=np.fromfile(out/'exact-double-values.bin',np.float32).reshape(f,256);fp32=np.fromfile(out/'exact-fp32-values.bin',np.float32).reshape(f,256)
    values=decode(raw,f)
    metrics=dict(relative_error_double=relative(values,double),relative_error_fp32=relative(values,fp32),candidate_byte_mismatches=int(np.count_nonzero(raw!=expected)))
    result=dict(passed=bool(same and metrics['relative_error_double']<=.02 and metrics['relative_error_fp32']<=.02),
                exact_cpu_candidate_bytes=bool(same),metrics=metrics,device_output_sha256=digest(out/'device-packed.bin'),scope='standalone synthetic operator only; not a model or performance gate')
    (out/'device-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out',required=True,type=Path);q.add_argument('--build',required=True,type=Path);q.add_argument('--cpu-probe',required=True,type=Path);q.add_argument('--seed',default=212,type=int)
    q=sub.add_parser('check');q.add_argument('--out',required=True,type=Path)
    a=p.parse_args();return prepare(a) if a.command=='prepare' else check(a)
if __name__=='__main__':raise SystemExit(main())
