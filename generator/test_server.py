import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server import handler,Queue


class FakeQueue:
    def __init__(self):self.created=[]
    def create(self,prompt,seed,nonce):self.created.append((prompt,seed,nonce));return{'id':'a'*32,'state':'preparing'}
    def snapshot(self,key):raise KeyError(key)


class LocalServerMechanics(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'index.html').write_text('explorer')
        self.queue=FakeQueue();self.server=ThreadingHTTPServer(('127.0.0.1',0),lambda *args:None)
        self.port=self.server.server_port;self.server.RequestHandlerClass=handler(self.queue,self.root,self.port)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def request(self,method,path,body=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.request(method,path,body=body,headers=headers or {});response=connection.getresponse();result=(response.status,response.read());connection.close();return result
    def test_capabilities_and_static_site(self):
        status,raw=self.request('GET','/api/generator/capabilities');self.assertEqual(status,200);self.assertEqual(json.loads(raw)['mode'],'live')
        self.assertEqual(self.request('GET','/'),(200,b'explorer'))
    def test_cross_origin_and_rebound_hosts_cannot_submit(self):
        data=json.dumps({'prompt':'a cone','seed':113,'requestID':'stable-request-123'})
        for headers in [{'Content-Type':'application/json'},{'Content-Type':'application/json','Origin':'https://external.example'},
                        {'Host':'external.example','Content-Type':'application/json','Origin':f'http://127.0.0.1:{self.port}'}]:
            self.assertEqual(self.request('POST','/api/generator/jobs',data,headers)[0],403)
        self.assertEqual(self.queue.created,[])
    def test_same_origin_submission_and_unknown_job(self):
        data=json.dumps({'prompt':'a cone','seed':113,'requestID':'stable-request-123'})
        status,raw=self.request('POST','/api/generator/jobs',data,{'Content-Type':'application/json','Origin':f'http://127.0.0.1:{self.port}'})
        self.assertEqual(status,202);self.assertEqual(self.queue.created,[('a cone',113,'stable-request-123')])
        self.assertEqual(self.request('GET','/api/generator/jobs/'+'f'*32)[0],404)
    def test_static_path_traversal_and_non_json_rejected(self):
        self.assertEqual(self.request('GET','/../outside')[0],404)
        self.assertEqual(self.request('GET','/%2e%2e/outside')[0],404)
        self.assertEqual(self.request('POST','/api/generator/jobs','prompt=x',{'Content-Type':'text/plain','Origin':f'http://127.0.0.1:{self.port}'})[0],415)


class QueueRecovery(unittest.TestCase):
    def test_ambiguous_scheduler_submission_keeps_one_request_for_inspection(self):
        with tempfile.TemporaryDirectory() as directory,patch('server.runner.stage'),patch('server.runner.submit',side_effect=TimeoutError('Response lost')) as submit,patch('server.threading.Thread',side_effect=lambda **kw:SimpleNamespace(start=kw['target'])):
            queue=Queue(directory);first=queue.create('a cone',113,'stable-request-123')
            self.assertEqual(first['state'],'unknown')
            second=queue.create('a cone',113,'stable-request-123')
            self.assertEqual(first['id'],second['id']);self.assertEqual(submit.call_count,1)
            with self.assertRaisesRegex(RuntimeError,'already active'):queue.create('another prompt',42,'different-request-456')

    def test_partial_collection_cannot_be_served_as_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/('a'*32);(run/'result').mkdir(parents=True)
            (run/'job.json').write_text('{}');(run/'result/report.json').write_text('{"samples":[]}')
            queue=Queue(root)
            with patch('server.runner.status',return_value={'state':'succeeded'}),patch('server.runner.collect',side_effect=ValueError('Inventory differs')):
                with self.assertRaisesRegex(ValueError,'Inventory differs'):queue.snapshot('a'*32)
            self.assertNotIn('report',queue.records['a'*32])

    def test_reload_preserves_nonce_and_refuses_duplicate_or_uncertain_launch(self):
        with tempfile.TemporaryDirectory() as directory,patch('server.threading.Thread') as thread:
            queue=Queue(directory);first=queue.create('a cone',113,'stable-request-123')
            self.assertEqual(thread.call_count,1)
            restored=Queue(directory)
            again=restored.create('a cone',113,'stable-request-123')
            self.assertEqual(again['id'],first['id']);self.assertEqual(again['state'],'unknown')
            with self.assertRaisesRegex(ValueError,'different inputs'):restored.create('a cube',42,'stable-request-123')
            with self.assertRaisesRegex(RuntimeError,'already active'):restored.create('a cube',42,'different-request-456')
            self.assertEqual(thread.call_count,1)


if __name__=='__main__':unittest.main()
