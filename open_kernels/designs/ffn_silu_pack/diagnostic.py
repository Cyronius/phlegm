"""Diagnostic packed output plus actual prepack FP32 values; separate image."""
from pathlib import Path
import os
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
HERE=Path(__file__).resolve().parent

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def design(G: In,U: In,P: Out,*,features: CompileTime[int],cores: CompileTime[int]):
    assert cores in (4,8) and features>0 and features%(64*cores)==0
    count=features//(64*cores); nb=8
    input_ty=np.ndarray[(64*32,),np.dtype[np.float32]]
    output_ty=np.ndarray[(4608,),np.dtype[np.uint8]]
    g_ty=np.ndarray[(features*256,),np.dtype[np.float32]]
    p_ty=np.ndarray[(features*256*9//4,),np.dtype[np.uint8]]
    kern=ExternalFunction('ffn_silu_pack_diagnostic',source_file=str(HERE/'diagnostic.cc'),arg_types=[input_ty,input_ty,output_ty,input_ty],
                          include_dirs=[str(HERE)],compile_flags=['-ffp-contract=off'])
    gf=[ObjectFifo(input_ty,name=f'g{c}',depth=2) for c in range(cores)]
    uf=[ObjectFifo(input_ty,name=f'u{c}',depth=2) for c in range(cores)]
    pf=[ObjectFifo(output_ty,name=f'p{c}',depth=2) for c in range(cores)]
    vf=[ObjectFifo(input_ty,name=f'v{c}',depth=1) for c in range(cores)]
    def worker(g,u,p,v,fn):
        for _ in range_(count*nb):
            a=g.acquire(1);b=u.acquire(1);o=p.acquire(1);vv=v.acquire(1)
            fn(a,b,o,vv)
            g.release(1);u.release(1);p.release(1);v.release(1)
    workers=[Worker(worker,[gf[c].cons(),uf[c].cons(),pf[c].prod(),vf[c].prod(),kern],tile=Tile(c,2),stack_size=4096) for c in range(cores)]
    def sequence(g,u,p,v,gs,us,ps,vs):
        tg=TaskGroup()
        for c in range(cores):
            it=TensorAccessPattern((features,256),c*count*64*256,[count,nb,64,32],[64*256,32,256,1])
            ot=TensorAccessPattern((features*256*9//4,),c*count*nb*4608,[count,nb,144,32],[nb*4608,4608,32,1])
            gs[c].fill(g,tap=it,group=tg)
            us[c].fill(u,tap=it,group=tg)
            ps[c].drain(p,tap=ot,wait=True,group=tg)
            vs[c].drain(v,tap=it,wait=True,group=tg)
        tg.finish()
    rt=Runtime(sequence,[g_ty,g_ty,p_ty,g_ty,[f.prod() for f in gf],[f.prod() for f in uf],[f.cons() for f in pf],[f.cons() for f in vf]])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()
DESIGN=design
SPECIALIZE={'features':int(os.environ.get('FSP_FEATURES','8192')),'cores':int(os.environ.get('FSP_CORES','4'))}
