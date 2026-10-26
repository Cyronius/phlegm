"""Compile standalone prototype. Invoke under dibs run build; never launches NPU."""
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--features',type=int,default=8192);p.add_argument('--cores',type=int,choices=(4,8),default=4)
 a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT/'target'):p.error('generated artifacts must stay under target/')
 if a.features<=0 or a.features%(64*a.cores):p.error('features must divide 64*cores')
 out.mkdir(parents=True,exist_ok=False)
 files=[HERE/n for n in ('encoder.h','kernel.cc','design.py','build.py')]
 hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
 env=os.environ|{'FSP_FEATURES':str(a.features),'FSP_CORES':str(a.cores),'OFLM_KEEP_FAILED':'1'}
 cmd=[sys.executable,str(ROOT/'open_kernels/build_design.py'),str(HERE/'design.py'),str(out/'build')]
 record=dict(complete=False,features=a.features,tokens=256,cores=a.cores,source_sha256=hashes,command=cmd,hardware_validation='not performed')
 try:
  with (out/'build.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert hashes=={str(f.relative_to(ROOT)):digest(f) for f in files},'source changed during build'
  record.update(complete=True,artifacts={n:digest(out/'build'/n) for n in ('final.xclbin','insts.bin','insts.elf')})
 finally:(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
 print(out)
if __name__=='__main__':main()
