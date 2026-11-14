import json, pathlib, subprocess, time, urllib.request, sys
out=pathlib.Path(sys.argv[1]); out.mkdir(exist_ok=True)
models=['KAT-Coder-V2.5-Dev-Q4pure-NPU2','gemma-4-26B-A4B-it-NPU2','gpt-oss-20b-NPU2','flint-NPU2','Hy-Memory-30B-A3B-NPU2','LFM2.5-24B-A2B-NPU2']
def post(route,data):
 return json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8415/'+route,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=300).read())
for model in models:
 d=pathlib.Path('/home/cyrus/models')/model
 cmd=[str(pathlib.Path.home()/'.local/bin/llama'),'serve','-m',str(d/'model.gguf'),'--device','none','-ngl','0','-t','8','-c','2048','-np','1','--port','8415','--host','127.0.0.1','--jinja','--chat-template-file',str(d/'chat_template.jinja')]
 log=open(out/(model+'.log'),'w'); p=subprocess.Popen(cmd,stdout=log,stderr=log)
 data={'model':model,'messages':[{'role':'user','content':'Call get_weather for Paris now. Do not answer from memory. Do not explain.'}], 'tools':[{'type':'function','function':{'name':'get_weather','description':'Get the weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}], 'chat_template_kwargs':{'enable_thinking':False,'reasoning_effort':'no_think' if model.startswith('Hy-') else 'low'}}
 meta={'model':model,'command':cmd,'build':'0.8.0-dev build 12015 commit a34de8618 Clang 22.1.8 Linux x86_64','request':data}
 try:
  for i in range(180):
   if p.poll() is not None: raise RuntimeError('load failed: '+(out/(model+'.log')).read_text()[-2500:])
   try:
    urllib.request.urlopen('http://127.0.0.1:8415/health',timeout=1); break
   except Exception: time.sleep(1)
  applied=post('apply-template',data); prompt=applied['prompt']; meta['prompt']=prompt
  result=post('completion',{'prompt':prompt,'n_predict':192,'temperature':0,'top_k':1,'seed':1,'cache_prompt':False,'special':True})
  meta['output']=result['content']; meta['tokens_predicted']=result.get('tokens_predicted'); meta['source']='CPU model output'; print(model,repr(meta['output']),flush=True)
 except Exception as e: meta['error']=str(e); print(model,'ERROR',e,flush=True)
 finally:
  p.terminate()
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill(); p.wait()
  log.close(); (out/(model+'.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
