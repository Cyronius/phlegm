"""Build prepack-values diagnostic; never replaces the failed packed fixture."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT/'target'):p.error('generated artifacts must stay under target/')
 out.mkdir(parents=True,exist_ok=False)
 sources=[HERE/n for n in ('encoder.h','diagnostic.cc','diagnostic.py','build_diagnostic.py')]
 record=dict(complete=False,source_sha256={str(f.relative_to(ROOT)):digest(f) for f in sources},features=8192,tokens=256,cores=4,output='packed bfp16_hilo plus float32[F,T]',hardware_validation='not performed')
 cmd=[sys.executable,str(ROOT/'open_kernels/build_design.py'),str(HERE/'diagnostic.py'),str(out/'build')]
 try:
  with (out/'build.log').open('w') as log:subprocess.run(cmd,env=os.environ|{'FSP_FEATURES':'8192','FSP_CORES':'4','OFLM_KEEP_FAILED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
  assert record['source_sha256']=={str(f.relative_to(ROOT)):digest(f) for f in sources}
  record.update(complete=True,artifacts={n:digest(out/'build'/n) for n in ('final.xclbin','insts.bin','insts.elf')})
 finally:(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
 print(out)
if __name__=='__main__':main()
