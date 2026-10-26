"""Prepare/check packed + actual prepack-value diagnostic. No device execution."""
import argparse,json
from pathlib import Path
import numpy as np
from probe import ROOT,digest,independent_pack,relative

def prepare(a):
 out=a.out.resolve();base=a.fixture.resolve();build=a.build.resolve()
 if not out.is_relative_to(ROOT/'target'):raise ValueError('ignored target/ destination required')
 meta=json.loads((base/'fixture.json').read_text());bm=json.loads((build/'build.json').read_text());assert bm['complete']
 for name,sha in meta['files'].items():assert digest(base/name)==sha
 for name,sha in bm['source_sha256'].items():assert digest(ROOT/name)==sha
 out.mkdir(parents=True,exist_ok=False)
 for name in meta['files']:(out/name).symlink_to(base/name)
 f=meta['features'];assert f==8192 and meta['tokens']==256
 cfg=['device',f'xclbin F {build}/build/final.xclbin',f'kernelx k F {build}/build/insts.bin',
  f'buf g {f*256*4} {out}/gate.bin',f'buf u {f*256*4} {out}/up.bin',f'buf p {f*256*9//4}',f'buf v {f*256*4}',
  'run k g u p v',f'dump p {out}/device-packed.bin {f*256*9//4}',f'dump v {out}/device-values.bin {f*256*4}','']
 (out/'run.cfg').write_text('\n'.join(cfg))
 record=dict(features=f,tokens=256,base_fixture=str(base),base_fixture_sha256=digest(base/'fixture.json'),
  original_strict_failure_device_sha256=digest(base/'device-packed.bin'),build_json=str(build/'build.json'),build_json_sha256=digest(build/'build.json'),
  generator_sha256=digest(Path(__file__)),artifacts={str(build/'build'/n):sha for n,sha in bm['artifacts'].items()},
  contract='Independent packing must match actual device F32 exactly; actual F32 relative error <=.02 separately vs double and FP32 SiLU; original strict byte failure retained')
 (out/'diagnostic-fixture.json').write_text(json.dumps(record,indent=2)+'\n');print(out)

def check(a):
 out=a.out.resolve();meta=json.loads((out/'diagnostic-fixture.json').read_text());base=Path(meta['base_fixture']);f=meta['features']
 assert digest(base/'fixture.json')==meta['base_fixture_sha256']
 assert digest(base/'device-packed.bin')==meta['original_strict_failure_device_sha256']
 assert digest(Path(meta['build_json']))==meta['build_json_sha256']
 for name,sha in meta['artifacts'].items():assert digest(Path(name))==sha
 for name,sha in json.loads((base/'fixture.json').read_text())['files'].items():assert digest(out/name)==sha
 values=np.fromfile(out/'device-values.bin',np.float32);packed=np.fromfile(out/'device-packed.bin',np.uint8)
 assert values.size==f*256 and packed.size==f*256*9//4,'output size mismatch'
 values=values.reshape(f,256)
 assert np.all(np.isfinite(values)) and np.max(np.abs(values))<=1e20,'invalid actual prepack activation'
 expected=independent_pack(values);mismatches=int(np.count_nonzero(expected!=packed))
 reference={name:np.fromfile(out/name,np.float32).reshape(f,256) for name in ('exact-double-values.bin','exact-fp32-values.bin')}
 errors={name:relative(values,v) for name,v in reference.items()}
 original=np.fromfile(base/'device-packed.bin',np.uint8);cpu=np.fromfile(out/'expected-candidate.bin',np.uint8)
 result=dict(passed=bool(mismatches==0 and max(errors.values())<=.02),actual_device_values_packing_byte_mismatches=mismatches,
  actual_device_values_relative_errors=errors,original_strict_gate_passed=bool(np.array_equal(original,cpu)),
  original_strict_byte_mismatches=int(np.count_nonzero(original!=cpu)),diagnostic_vs_original_device_byte_mismatches=int(np.count_nonzero(packed!=original)),
  original_failure_retained=True,device_values_sha256=digest(out/'device-values.bin'),device_packed_sha256=digest(out/'device-packed.bin'),
  scope='Standalone arithmetic/encoder decomposition; no model, timing or runtime acceptance')
 (out/'diagnostic-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 1

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('prepare');q.add_argument('--fixture',required=True,type=Path);q.add_argument('--build',required=True,type=Path);q.add_argument('--out',required=True,type=Path)
 q=s.add_parser('check');q.add_argument('--out',required=True,type=Path)
 a=p.parse_args();return prepare(a) if a.command=='prepare' else check(a)
if __name__=='__main__':raise SystemExit(main())
