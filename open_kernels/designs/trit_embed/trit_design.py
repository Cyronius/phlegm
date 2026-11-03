"""trit-embed-3l encoder: all layers of a 384-wide ternary BERT in one 32-core dispatch.

A dispatch carries two 128-token slots, eight token rows per core: slot 0 on cores 0-15 (columns 0-3),
slot 1 on cores 16-31. Each slot has its own two broadcast streams of 4096-byte elements, one per core
input channel (slot 0 from shim columns 0 and 1, slot 1 from 4 and 5): the weights go out on all four,
the input states and the K/V only to their slot's cores. Defaults are the shipped graph: 3 layers, each
stream double-buffered.

Profiling switches (build.py sets them): TRIT_STUB=all compiles every kernel as an immediate return
(data movement and sync only); a comma list such as TRIT_STUB=step stubs only those kernels.
TRIT_BDEPTH sets the depth of each broadcast stream. TRIT_DEFS adds preprocessor macros to every kernel
(profiling switches in fused2.h). TRIT_OPT sets aiecc's whole-program optimization level for core code.
TRIT_NOSYNC (timing only) fetches the K/V back without awaiting their drain.

The cores consume element pairs, one from each of their slot's streams. A dispatch starts with 25
preamble pairs (f2_pre): the mask packet, a segment ID per token (0 = padding; tokens attend only
within their own segment, so a slot serves any packing of texts), then the slot's input states, of
which each core keeps its own. A layer is then 110 pairs (f2_step, fused2.h kSteps): stream A carries
packet 2j and stream B packet 2j + 1 of the layer's 8192-byte elements (170 packets of 4096 bytes: 32
output channels' packed ternary weights and, at byte 3072, their bias and scale, packet format 3,
trit_npu.weight_stream; each LayerNorm packet has an element of its own), except for attention: K and
V cross the cores through DDR and come back on the slot's streams, a head's 128 keys in four elements
of four cores' 8 keys each, two elements a pair.

Only the last layer's states leave the array, in two packets a core after the layer loop; the runtime
fetches each layer's K/V back with one shim task a stream."""
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
NOSYNC = os.environ.get('TRIT_NOSYNC','')=='1'
ESZ = 8192   # a layer element: two packets, one on each stream
HSZ = ESZ//2  # stream element
HEAD = 576    # fused2.h kHeadBytes
OSZ = 12*HEAD # core output packet: 12 heads' K/V, or half the final states
STACK = 0x900
DEFS = [s for s in os.environ.get('TRIT_DEFS','').split(',') if s]
assert CAP in (8,32,128)
assert LAYERS in (2,3)
NC,NR = 8,4
PACKETS = 170
ELEMS = (PACKETS-2)//2+2   # 8192-byte elements per layer: each LayerNorm packet has its own
PREAMBLE = 25               # mask, then the slot's input states in 24 pairs (f2_pre)
STEPS = ELEMS+24            # weight elements plus 24 attention pairs (f2_step)


@iron.jit(aiecc_flags=['--dynamic-objFifos','--alloc-scheme=basic-sequential']+([f'-O{OPT}'] if OPT else []))
def encoder(wts: In, mask: In, x: In, kv: InOut, out: Out, *, cfg: CompileTime[int], depth: CompileTime[int], srchash: CompileTime[int]):
    H = np.ndarray[(HSZ,),np.dtype[np.uint8]]
    O = np.ndarray[(OSZ,),np.dtype[np.uint8]]
    OC = np.ndarray[(NR//2*OSZ,),np.dtype[np.uint8]]
    X = np.ndarray[(3072,),np.dtype[np.float32]]
    P = np.ndarray[(768,),np.dtype[np.float32]]
    Q = np.ndarray[(3072,),np.dtype[bfloat16]]
    AQ = np.ndarray[(4096,),np.dtype[np.int8]]
    AQ2 = np.ndarray[(3072,),np.dtype[np.int8]]
    ML = np.ndarray[(32,),np.dtype[np.float32]]
    B = np.ndarray[(1024,),np.dtype[bfloat16]]
    ST = np.ndarray[(16,),np.dtype[np.int32]]
    i32=np.int32
    def kernel(name,args):
        return ExternalFunction(name,source_file=str(ROOT/f'{name}.cc'),arg_types=args,
                                include_dirs=[str(ROOT)]+include_dirs(),
                                compile_flags=(['-DF2_STUB'] if STUB&{'all',name[3:]} else [])+[f'-D{m}' for m in DEFS])
    pre=kernel('f2_pre',[H,H,X,B,ST,i32])
    step=kernel('f2_step',[H,H,X,Q,P,ML,B,AQ,AQ2,O,ST,i32])
    xo=kernel('f2_out',[X,O,i32])
    streams=[(ObjectFifo(H,name=f'stream_a{s}',depth=BDEPTH),ObjectFifo(H,name=f'stream_b{s}',depth=BDEPTH)) for s in range(2)]
    outputs,workers=[],[]

    def make_body(core):
        slot=core//16
        def body(ca,cb,oy,xv,q,par,aq,aq2,ml,bias,st,pref,stepf,outf):
            for _ in range_(PREAMBLE):
                a=ca.acquire(1);b=cb.acquire(1);pref(a,b,xv,bias,st,core);ca.release(1);cb.release(1)
            # Constant loop counts are preserved by dynamic FIFO lowering.
            for _ in range_(depth):
                # K and V, which leave in one core packet, and two Q steps; then the other Q steps while
                # K and V cross the cores. (Loop counts that are 2 mod 4 here, then 0 mod 4, keep the
                # lowering from peeling iterations into program memory.)
                dest=oy.acquire(1)
                for _ in range_(14):
                    a=ca.acquire(1);b=cb.acquire(1);stepf(a,b,xv,q,par,ml,bias,aq,aq2,dest,st,slot);ca.release(1);cb.release(1)
                oy.release(1)
                # The rest of the layer, which never writes the packet: dest only fills the argument.
                for _ in range_(STEPS-14):
                    a=ca.acquire(1);b=cb.acquire(1);stepf(a,b,xv,q,par,ml,bias,aq,aq2,dest,st,slot);ca.release(1);cb.release(1)
            # The final states, in two packets.
            dest=oy.acquire(1);outf(xv,dest,0);oy.release(1)
            dest=oy.acquire(1);outf(xv,dest,1);oy.release(1)
        return body

    for c in range(NC):
        # Each column's packets leave on both of its shim's input channels: rows 0-1 joined on one,
        # rows 2-3 on the other (outputs[2c + part]), which halves the time the K/V take to drain.
        outs=[]
        for part in range(2):
            fo=ObjectFifo(OC,name=f'yo{c}_{part}',depth=1)
            outs+=fo.prod().join([r*OSZ for r in range(NR//2)],tile=Tile(c,1),obj_types=[O]*(NR//2),
                                 names=[f'y{c}_{2*part+r}' for r in range(NR//2)],depths=[1]*(NR//2))
            outputs.append(fo.cons(tile=Tile(c,0)))
        for r in range(NR):
            tile=Tile(c,2+r)
            sa,sb=streams[c//4]
            args=[sa.cons(),sb.cons(),outs[r].prod(),
                  Buffer(X),Buffer(Q),Buffer(P),Buffer(AQ),Buffer(AQ2),Buffer(ML),Buffer(B),Buffer(ST,initial_value=np.zeros(16,np.int32)),
                  pre,step,xo]
            workers.append(Worker(make_body(c*NR+r),fn_args=args,tile=tile,stack_size=STACK))

    WT=depth*ELEMS*ESZ
    KVS=12*4*4096      # one slot's K/V: [head][element][4 cores][1024]
    KV=2*KVS
    XO=2*NC*NR*OSZ     # final states: [half][column][4 cores][OSZ]
    XB=256*384*4
    def halves(total,offset,n,half):
        """n 4096-byte halves of consecutive 8192-byte elements from offset: half 0 (stream A) or 1 (B)."""
        return TensorAccessPattern((1,total),offset+half*HSZ,[1,1,n,HSZ],[0,0,ESZ,1])
    def column(total,offset,size):
        return TensorAccessPattern((1,total),offset,[1,1,1,size],[0,0,0,1])
    def sequence(w,m,xs,scratch,o,pa0,pb0,pa1,pb1,outs):
        eps=((pa0,pb0),(pa1,pb1))      # each slot's two streams
        if any(d.startswith('F2_PROF_CYCLES') for d in DEFS):
            # Timer reads (fused2.h f2_now) go over the processor bus; without this the core hangs.
            from aie.dialects.aiex import npu_maskwrite32
            for c in range(NC):
                for r in range(NR):
                    npu_maskwrite32(column=c,row=2+r,address=0x32038,value=0x1,mask=0x1)
        pipe=Pipeline(inflight=3)
        for s in range(2):
            for h,ep in enumerate(eps[s]):
                pipe.fill(ep,m,halves(ESZ,0,1,h))
                pipe.fill(ep,xs,halves(XB,s*(XB//2),XB//2//ESZ,h))
        for layer in range(depth):
            base=layer*ELEMS*ESZ
            for s in range(2):
                for h,ep in enumerate(eps[s]):
                    pipe.fill(ep,w,halves(WT,base,18,h))
            tks=[]
            for s in range(2):
                tk=TaskGroup()
                for c in range(4*s,4*s+4):
                    for part in range(2):
                        # Column c holds slot c // 4's cores 4(c % 4) .. + 3: each head's 576 bytes into
                        # its 1024-byte place of element (head, c % 4).
                        outs[2*c+part].drain(scratch,tap=TensorAccessPattern((1,KV),s*KVS+(c%4)*4096+part*2048,
                                             [1,NR//2,12,HEAD],[0,1024,4*4096,1]),wait=True,group=tk)
                tks.append(tk)
            for s in range(2):
                if not NOSYNC:
                    tks[s].finish()
                # The slot's K/V, elements (head, 2k) on stream A and (head, 2k + 1) on stream B.
                for h,ep in enumerate(eps[s]):
                    pipe.fill(ep,scratch,halves(KV,s*KVS,KVS//ESZ,h))
                if NOSYNC:
                    tks[s].finish()     # timing only: attention reads the K/V before they are written
            for s in range(2):
                for h,ep in enumerate(eps[s]):
                    pipe.fill(ep,w,halves(WT,base+18*ESZ,ELEMS-18,h))
        to=TaskGroup()
        for c in range(NC):
            for half in range(2):
                for part in range(2):
                    outs[2*c+part].drain(o,tap=column(XO,(half*NC+c)*NR*OSZ+part*(NR//2)*OSZ,(NR//2)*OSZ),wait=True,group=to)
        to.finish()
        pipe.finish()
    rt=Runtime(sequence,[np.ndarray[(WT,),np.dtype[np.uint8]],np.ndarray[(ESZ,),np.dtype[np.uint8]],
                         np.ndarray[(XB,),np.dtype[np.uint8]],
                         np.ndarray[(KV,),np.dtype[np.uint8]],np.ndarray[(XO,),np.dtype[np.uint8]],
                         streams[0][0].prod(tile=Tile(0,0)),streams[0][1].prod(tile=Tile(1,0)),
                         streams[1][0].prod(tile=Tile(4,0)),streams[1][1].prod(tile=Tile(5,0)),outputs])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()

DESIGN=encoder
SOURCES=['fused2.h','gelu_lut.h','trit_design.py','../../include/vecmath.h']+sorted(p.name for p in ROOT.glob('f2_*.cc'))
SPECIALIZE={'cfg':CAP,'depth':LAYERS,'srchash':int(hashlib.sha256(b''.join(
    (ROOT/name).read_bytes() for name in SOURCES)+(f'{sorted(STUB)}{BDEPTH}{OPT}{DEFS}{NOSYNC}').encode()).hexdigest()[:8],16)}
