"""trit-embed-3l encoder: all layers of a 384-wide ternary BERT in one 32-core dispatch.

Four token rows per core; every core sees the whole broadcast. Defaults are the shipped graph:
3 layers, one 128-token slot, a double-buffered broadcast of 8192-byte elements.

Profiling switches (build.py sets them): TRIT_STUB=all compiles every kernel as an
immediate return (data movement and sync only); a comma list such as TRIT_STUB=attend
stubs only those kernels. TRIT_BDEPTH sets the broadcast FIFO depth.
TRIT_DEFS adds preprocessor macros to every kernel (profiling switches in fused2.h).
TRIT_OPT sets aiecc's whole-program optimization level for core code.

The mask element holds a segment ID per token (0 = padding). Tokens attend only within
their own segment, so one 128-token graph serves any packing of texts.

A layer is 170 weight packets of 4096 bytes: 32 output channels' packed ternary weights and, at
byte 3072, their bias and scale (packet format 2, trit_npu.weight_stream and fused2.h). The
broadcast pairs them into 86 elements of 8192 bytes; each LayerNorm packet gets an element of its
own. K and V cross the cores through DDR and come back 64 keys of one head an element.

The runtime awaits a layer's trace drain only at the next layer's K/V exchange (await order must
follow each shim channel's issue order) and fetches the K/V back with one 4-D shim task."""
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

CAP = int(os.environ.get('TRIT_CAP','128'))
LAYERS = int(os.environ.get('TRIT_LAYERS','3'))
STUB = {s for s in os.environ.get('TRIT_STUB','').split(',') if s}
BDEPTH = int(os.environ.get('TRIT_BDEPTH','2'))
OPT = os.environ.get('TRIT_OPT','')
ESZ = 8192   # broadcast element: two packets
DEFS = [s for s in os.environ.get('TRIT_DEFS','').split(',') if s]
assert CAP in (8,32,128)
assert LAYERS in (2,3)
NC,NR = 8,4
PACKETS = 170
ELEMS = (PACKETS-2)//2+2   # broadcast elements per layer: each LayerNorm packet has its own


@iron.jit(aiecc_flags=['--dynamic-objFifos','--alloc-scheme=basic-sequential']+([f'-O{OPT}'] if OPT else []))
def encoder(wts: In, mask: In, x: In, kv: InOut, out: Out, *, cfg: CompileTime[int], depth: CompileTime[int], srchash: CompileTime[int]):
    E = np.ndarray[(ESZ,),np.dtype[np.uint8]]
    O = np.ndarray[(8192,),np.dtype[np.uint8]]
    OC = np.ndarray[(NR*8192,),np.dtype[np.uint8]]
    X = np.ndarray[(1536,),np.dtype[np.float32]]
    XC = np.ndarray[(NR*1536,),np.dtype[np.float32]]
    P = np.ndarray[(1024,),np.dtype[np.float32]]
    Q = np.ndarray[(1536,),np.dtype[bfloat16]]
    AQ = np.ndarray[(1664,),np.dtype[np.int8]]
    AQ2 = AQ
    ML = np.ndarray[(32,),np.dtype[np.float32]]
    B = np.ndarray[(512,),np.dtype[np.float32]]
    i32=np.int32
    def kernel(name,args):
        name=name.replace('ep_','f2_')
        return ExternalFunction(name,source_file=str(ROOT/f'{name}.cc'),arg_types=args,
                                include_dirs=[str(ROOT)]+include_dirs(),
                                compile_flags=(['-DF2_STUB'] if STUB&{'all',name[3:]} else [])+[f'-D{m}' for m in DEFS])
    loadmask=kernel('ep_mask',[E,B,i32])
    qx=kernel('ep_quant_x',[X,AQ])
    qq=kernel('ep_quant_q',[Q,AQ2,i32])
    qq384=qq
    linear=kernel('ep_linear',[E,AQ,X,Q,O,i32,i32,i32])
    linear256=linear
    ln=kernel('ep_norm',[X,E])
    att=kernel('ep_attend',[Q,E,P,ML,B,i32,i32])
    xo=kernel('ep_out',[X,O])
    stream=ObjectFifo(E,name='broadcast',depth=BDEPTH)
    inputs,outputs,workers=[],[],[]

    def make_body(core):
        def body(ax,bx,oy,q,par,aq,aq2,ml,bias,lm,qxf,qqf,qq384f,lin,lin256,norm,attn,outf):
            xv=ax.acquire(1)
            e=bx.acquire(1);lm(e,bias,core);bx.release(1)
            # Constant loop counts are preserved by dynamic FIFO lowering.
            for _ in range_(depth):
                qxf(xv,aq)
                # Q remains local; K and V leave in one 8192-byte core packet. A projection
                # element is two packets: blocks 2j and 2j + 1 (f2_linear).
                dest=oy.acquire(1)
                for kind in range_(3):
                    for j in range_(6):
                        e=bx.acquire(1);lin(e,aq,xv,q,dest,j,kind,384);bx.release(1)
                oy.release(1)
                for h in range_(12):
                    for block in range_(2):
                        e=bx.acquire(1);attn(q,e,par,ml,bias,h,block);bx.release(1)
                # O projection and the first post-residual LayerNorm.
                qq384f(q,aq,384)
                # The dummy destination is the inactive output FIFO packet;
                # residual/GELU modes never access it.
                dest=oy.acquire(1)
                for j in range_(6):
                    e=bx.acquire(1);lin(e,aq,xv,q,dest,j,3,384);bx.release(1)
                e=bx.acquire(1);norm(xv,e);bx.release(1)
                qxf(xv,aq)
                for group in range_(6):
                    for j in range_(4):
                        e=bx.acquire(1);lin(e,aq,xv,q,dest,j,4,384);bx.release(1)
                    qqf(q,aq2,256)
                    for j in range_(6):
                        e=bx.acquire(1);lin256(e,aq2,xv,q,dest,j,3,256);bx.release(1)
                e=bx.acquire(1);norm(xv,e);bx.release(1)
                # This extra packet is consumed by the runtime as a layer trace.
                outf(xv,dest);oy.release(1)
            ax.release(1)
        return body

    for c in range(NC):
        fi=ObjectFifo(XC,name=f'xi{c}',depth=1)
        fo=ObjectFifo(OC,name=f'yo{c}',depth=1)
        ins=fi.cons().split([r*1536 for r in range(NR)],tile=Tile(c,1),obj_types=[X]*NR,
                            names=[f'x{c}_{r}' for r in range(NR)],depths=[1]*NR)
        outs=fo.prod().join([r*8192 for r in range(NR)],tile=Tile(c,1),obj_types=[O]*NR,
                            names=[f'y{c}_{r}' for r in range(NR)],depths=[1]*NR)
        inputs.append(fi.prod(tile=Tile(c,0)));outputs.append(fo.cons(tile=Tile(c,0)))
        for r in range(NR):
            tile=Tile(c,2+r)
            args=[ins[r].cons(),stream.cons(),outs[r].prod(),
                  Buffer(Q),Buffer(P),Buffer(AQ),Buffer(AQ2),Buffer(ML),Buffer(B),
                  loadmask,qx,qq,qq384,linear,linear256,ln,att,xo]
            workers.append(Worker(make_body(c*NR+r),fn_args=args,tile=tile,stack_size=0x1400))

    WT=depth*ELEMS*ESZ
    KV=32*8192
    XO=depth*KV
    def linear_tap(total,offset,size):
        return TensorAccessPattern((1,total),offset,[1,1,1,size],[0,0,0,1])
    def sequence(w,m,x,scratch,o,wp,ins,outs):
        if any(d.startswith('F2_PROF_CYCLES') for d in DEFS):
            # Timer reads (fused2.h f2_now) go over the processor bus; without this the core hangs.
            from aie.dialects.aiex import npu_maskwrite32
            for c in range(NC):
                for r in range(NR):
                    npu_maskwrite32(column=c,row=2+r,address=0x32038,value=0x1,mask=0x1)
        tx=TaskGroup()
        for c in range(NC):
            ins[c].fill(x,tap=linear_tap(32*1536,c*NR*1536,NR*1536),wait=True,group=tx)
        pipe=Pipeline(inflight=3)
        pipe.fill(wp,m,linear_tap(ESZ,0,ESZ))
        for layer in range(depth):
            base=layer*ELEMS*ESZ
            pipe.fill(wp,w,linear_tap(WT,base,36*4096))
            tk=TaskGroup()
            for c in range(NC):
                outs[c].drain(scratch,tap=linear_tap(KV,c*NR*8192,NR*8192),wait=True,group=tk)
            if layer:
                # The previous layer's trace drain, long since done. Awaits must follow each
                # channel's issue order, so it is awaited before this layer's K/V drain.
                to.finish()
            tk.finish()
            # Each element: one head's 512 bytes from 16 cores, 64 keys; heads, then key halves.
            pipe.fill(wp,scratch,TensorAccessPattern((1,KV),0,[12,2,16,512],[512,16*8192,8192,1]))
            pipe.fill(wp,w,linear_tap(WT,base+36*4096,ELEMS*ESZ-36*4096))
            to=TaskGroup()
            for c in range(NC):
                outs[c].drain(o,tap=linear_tap(XO,layer*KV+c*NR*8192,NR*8192),wait=True,group=to)
        to.finish()
        pipe.finish();tx.finish()
    rt=Runtime(sequence,[np.ndarray[(WT,),np.dtype[np.uint8]],E,
                         np.ndarray[(32*1536,),np.dtype[np.float32]],
                         np.ndarray[(KV,),np.dtype[np.uint8]],np.ndarray[(XO,),np.dtype[np.uint8]],
                         stream.prod(tile=Tile(0,0)),inputs,outputs])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()

DESIGN=encoder
SOURCES=['fused2.h','gelu_lut.h','trit_design.py','../../include/vecmath.h']+sorted(p.name for p in ROOT.glob('f2_*.cc'))
SPECIALIZE={'cfg':CAP,'depth':LAYERS,'srchash':int(hashlib.sha256(b''.join(
    (ROOT/name).read_bytes() for name in SOURCES)+(f'{sorted(STUB)}{BDEPTH}{OPT}{DEFS}').encode()).hexdigest()[:8],16)}
