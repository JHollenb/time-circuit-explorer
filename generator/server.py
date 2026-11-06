"""Loopback-only live website + prediction queue; scheduler credentials stay server-side."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import threading
import uuid
from urllib.parse import unquote,urlparse

import run as runner
from worker import validate_input

PROJECT=Path(__file__).resolve().parents[1]
TERMINAL={'succeeded','failed','cancelled','killed','timeout','timed_out','rejected'}


class Queue:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.records={};self.nonces={}
        ledger=self.root/'queue.json'
        if ledger.exists():
            saved=runner.read(ledger);self.nonces=saved['nonces'];self.records=saved['records']
            for record in self.records.values():
                if record['state'] not in TERMINAL:record.update(state='unknown',error='Server restarted; inspect the retained request if scheduler status is unavailable.')
        for path in self.root.iterdir():
            if path.is_dir() and re.fullmatch('[0-9a-f]{32}',path.name) and (path/'job.json').is_file():
                self.records.setdefault(path.name,{'id':path.name,'state':'unknown','runDirectory':str(path)})

    def persist(self):
        temporary=self.root/'queue.json.tmp'
        temporary.write_text(json.dumps({'nonces':self.nonces,'records':{key:{k:v for k,v in value.items() if k!='report'} for key,value in self.records.items()}},sort_keys=True))
        temporary.replace(self.root/'queue.json')

    def snapshot(self,key):
        with self.lock:
            if key not in self.records:raise KeyError(key)
            record=self.records[key];root=Path(record['runDirectory'])
            if (root/'job.json').exists() and not (root/'collection.json').exists():
                state=runner.status(root);record.update(state)
                if state['state']=='succeeded':
                    runner.collect(root)
            if (root/'collection.json').exists():
                receipt=runner.read(root/'collection.json')
                if runner.sha(root/'result/report.json')!=receipt['reportSHA256']:raise ValueError('Collected report changed')
                report=runner.read(root/'result/report.json');record.update(state='succeeded',report=report)
            return{k:v for k,v in record.items() if k!='runDirectory'}

    def create(self,prompt,seed,nonce):
        prompt,seed=validate_input(prompt,seed)
        if not isinstance(nonce,str) or not re.fullmatch('[A-Za-z0-9_-]{12,80}',nonce):raise ValueError('A valid request ID is required.')
        with self.lock:
            if nonce in self.nonces:
                record=self.records[self.nonces[nonce]]
                if record.get('prompt')!=prompt or record.get('seed')!=seed:raise ValueError('Request ID already belongs to different inputs.')
                return self.snapshot(self.nonces[nonce])
            for key in list(self.records):
                state=self.snapshot(key)['state']
                if state not in TERMINAL:raise RuntimeError('A generation is already active. Wait for it to finish.')
            key=uuid.uuid4().hex;root=self.root/key;self.nonces[nonce]=key
            self.records[key]={'id':key,'state':'preparing','prompt':prompt,'seed':seed,'runDirectory':str(root)}
            self.persist()
            def launch():
                submitting=False
                try:
                    runner.stage(root,[{'prompt':prompt,'seed':seed}]);submitting=True;job=runner.submit(root)
                    with self.lock:self.records[key].update(job,state='queued');self.persist()
                except Exception as error:
                    with self.lock:
                        self.records[key].update(state='unknown' if submitting else 'failed',error=('Submission outcome needs inspection; request retained. ' if submitting else '')+str(error));self.persist()
            threading.Thread(target=launch,daemon=True).start()
            return{k:v for k,v in self.records[key].items() if k!='runDirectory'}


def handler(queue,static,port):
    static=Path(static).resolve()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass
        def json(self,status,value):
            raw=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def valid_host(self):return self.headers.get('Host') in {f'127.0.0.1:{port}',f'localhost:{port}'}
        def do_POST(self):
            if not self.valid_host():return self.json(403,{'error':'Loopback host required.'})
            origin=self.headers.get('Origin')
            if origin not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}:return self.json(403,{'error':'Use the local explorer to submit a generation.'})
            if self.path!='/api/generator/jobs':return self.json(404,{'error':'Unknown endpoint.'})
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.json(415,{'error':'JSON request required.'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 1<=length<=8192:raise ValueError('Request body exceeds limit.')
                value=json.loads(self.rfile.read(length))
                if not isinstance(value,dict):raise ValueError('JSON object required.')
                result=queue.create(value.get('prompt'),value.get('seed'),value.get('requestID'))
                self.json(202,result)
            except ValueError as error:self.json(400,{'error':str(error)})
            except RuntimeError as error:self.json(409,{'error':str(error)})
            except Exception:self.json(503,{'error':'The scheduler is unavailable. Check the existing request before submitting another.'})
        def do_GET(self):
            if not self.valid_host():return self.json(403,{'error':'Loopback host required.'})
            path=unquote(urlparse(self.path).path)
            if path=='/api/generator/capabilities':return self.json(200,{'schema':1,'mode':'live','profile':'FLUX.2 Klein 4B · 512×512 · 4 passes','maxActive':1,'backend':'mrun','nativeTransformerLoaded':False})
            match=re.fullmatch(r'/api/generator/jobs/([0-9a-f]{32})(?:/(sample-\d{2}\.png|report\.json))?',path)
            if match:
                key,filename=match.groups()
                try:
                    value=queue.snapshot(key)
                    if filename:
                        if value['state']!='succeeded':return self.json(409,{'error':'Result is not ready.'})
                        target=queue.root/key/'result'/filename
                        if not target.is_file():return self.json(404,{'error':'Unknown result.'})
                        return self.file(target)
                    return self.json(200,value)
                except KeyError:return self.json(404,{'error':'Unknown local generation.'})
                except Exception:return self.json(503,{'error':'Unable to read this scheduler job. It remains in the local ledger.'})
            if path.startswith('/api/'):return self.json(404,{'error':'Unknown endpoint.'})
            target=(static/('.'+(path+'index.html' if path.endswith('/') else path))).resolve()
            if not target.is_relative_to(static) or not target.is_file():return self.json(404,{'error':'Not found.'})
            self.file(target)
        def file(self,target):
            raw=target.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=4177);parser.add_argument('--runs',type=Path,default=PROJECT/'artifacts/generator-live')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Choose a port from 1024 to 65535.')
    runner.api_client().json('GET','/healthz')
    subprocess.run(['node',str(PROJECT/'viz/scripts/build.mjs')],check=True)
    queue=Queue(args.runs);server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(queue,PROJECT/'dist',args.port))
    print(f'Live generator: http://127.0.0.1:{args.port}/?model=flux2-klein-4b#generator',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
