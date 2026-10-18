"""Serve the project playground with the unmodified MosaicGym Toolbox.

Only transport, session snapshots, PNG encoding and service resource limits live
here. Tool validation, execution, RGBA handling and image registration come from
mosaic_gym.toolbox.Toolbox.run. No tool algorithm is reimplemented.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import inspect
import json
import secrets
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SITE=ROOT/'docs'
DEFAULT_SOURCE=ROOT/'src'
MAX_PIXELS=16_777_216
MAX_WORKSPACE_BYTES=96*1024*1024
MAX_IMAGES=24


class ServiceError(Exception):
    def __init__(self,message,status=400):
        super().__init__(message);self.status=status


def load_toolbox(source):
    source=Path(source).resolve()
    sys.path.insert(0,str(source))
    from mosaic_gym.toolbox import Toolbox
    from mosaic_gym.contracts import ToolCall
    from PIL import Image
    for symbol in (Toolbox,ToolCall):
        if not Path(inspect.getfile(symbol)).resolve().is_relative_to(source):
            raise RuntimeError('A different MosaicGym package is already imported. Start a fresh Python process.')
    return Toolbox,ToolCall,Image


class Session:
    def __init__(self,box,example):
        self.box=box;self.example=example;self.initial_count=len(box.image_ids())
        self.origins={iid:'Input' for iid in box.image_ids()}
        self.preview=None;self.revision=0;self.touched=time.monotonic();self.lock=threading.Lock()


class GymService:
    def __init__(self,site=DEFAULT_SITE,source=DEFAULT_SOURCE):
        self.site=Path(site).resolve();self.source=Path(source).resolve()
        self.Toolbox,self.ToolCall,self.Image=load_toolbox(source)
        self.specs=self.Toolbox().registry.list_tools()
        self.examples=json.loads((self.site/'assets/gym-examples.json').read_text())
        self.sessions={};self.lock=threading.Lock();self.compute_lock=threading.Lock()
        files=sorted((self.source/'mosaic_gym/toolbox').rglob('*.py'))+[self.source/'mosaic_gym/contracts.py']
        self.digest=hashlib.sha256(b''.join(p.relative_to(self.source).as_posix().encode()+p.read_bytes() for p in files)).hexdigest()

    def config(self):
        return {'engine':'MosaicGym Toolbox','toolbox_digest':self.digest,'tools':self.specs,
                'examples':[{k:e[k] for k in ['id','label','credits']} for e in self.examples]}

    def session(self,sid):
        with self.lock:
            session=self.sessions.get(sid)
            if session is None or time.monotonic()-session.touched>1800:
                self.sessions.pop(sid,None);raise ServiceError('Workspace expired. Reset to reload the example.',410)
            session.touched=time.monotonic();return session

    def create(self,example_id):
        example=next((e for e in self.examples if e['id']==example_id),None)
        if example is None:raise ServiceError('Unknown example.')
        images=[]
        for item in example['images']:
            path=(self.site/item['file']).resolve()
            if not path.is_relative_to(self.site):raise ServiceError('Invalid example asset.')
            with self.Image.open(path) as image:
                if image.width*image.height>MAX_PIXELS:raise ServiceError('Example image exceeds the service pixel budget.')
                images.append(image.convert('RGBA'))
        box=self.Toolbox();box.reset(images)
        sid=secrets.token_hex(20);session=Session(box,example)
        with self.lock:
            self.sessions={k:v for k,v in self.sessions.items() if time.monotonic()-v.touched<1800}
            if len(self.sessions)>=8:raise ServiceError('Tool service is busy. Try again shortly.',503)
            self.sessions[sid]=session
        return self.describe(sid,session)

    def describe_image(self,sid,session,iid,image,preview=None):
        base=f'/api/gym/sessions/{sid}/'+(f'previews/{preview}' if preview else f'images/{iid}')
        return {'id':iid,'width':image.width,'height':image.height,
                'rgba':base+'/rgba.png?revision='+str(session.revision),
                'alpha':base+'/alpha.png?revision='+str(session.revision),
                'thumbnail':base+'/thumbnail.png?revision='+str(session.revision),
                'origin':session.origins.get(iid,'Input')}

    def describe(self,sid,session):
        return {'session_id':sid,'revision':session.revision,'initial_count':session.initial_count,
                'images':[self.describe_image(sid,session,iid,im) for iid,im in session.box.images_by_id().items()]}

    def preview(self,sid,call,revision):
        session=self.session(sid)
        with session.lock,self.compute_lock:
            if revision!=session.revision:raise ServiceError('Workspace changed. Select the inputs again.',409)
            name=call.get('name');arguments=call.get('arguments');targets=call.get('target_images')
            if not isinstance(name,str) or not isinstance(arguments,dict) or not isinstance(targets,list) or not all(isinstance(i,str) for i in targets):raise ServiceError('Invalid tool request.')
            # Admission control only; the original Toolbox validates all tool semantics.
            if name=='make_collage' and all(i in session.box.images_by_id() for i in targets) and targets:
                rows,cols=arguments.get('rows'),arguments.get('cols')
                if isinstance(rows,int) and isinstance(cols,int) and rows>0 and cols>0:
                    inputs=[session.box.images_by_id()[i] for i in targets]
                    if rows*cols*max(im.width for im in inputs)*max(im.height for im in inputs)>MAX_PIXELS:
                        raise ServiceError('Requested collage exceeds the demo memory budget. Use fewer cells or resize the inputs.')
            # A preview branches the real state. Shared source images are read-only
            # under the Toolbox contract; its image dictionary is independently copied.
            branch=self.Toolbox();branch.state=copy.copy(session.box.state)
            branch.state.images=dict(session.box.state.images)
            output=branch.run([self.ToolCall(name,arguments,targets)])[0]
            session.preview=None
            if output.error:return {'error':output.error,'tool_name':output.tool_name,'tool_feedback':output.tool_feedback,'new_image_ids':[]}
            if len(output.produced_images)!=1:raise ServiceError('This interface requires one output image.')
            image=output.produced_images[0]
            if image.width*image.height>MAX_PIXELS:raise ServiceError('Result exceeds the demo memory budget. Reduce output size.')
            token=secrets.token_hex(16)
            session.preview={'token':token,'box':branch,'output':output,'call':copy.deepcopy(call)}
            return {'preview_id':token,'revision':session.revision,'tool_name':output.tool_name,
                    'tool_feedback':output.tool_feedback,'new_image_ids':output.new_image_ids,'error':None,
                    'image':self.describe_image(sid,session,output.new_image_ids[0],image,token)}

    def commit(self,sid,token,revision):
        session=self.session(sid)
        with session.lock:
            draft=session.preview
            if revision!=session.revision or not draft or draft['token']!=token:raise ServiceError('Preview changed. Run the tool again.',409)
            box=draft['box'];images=box.images_by_id()
            if len(images)>MAX_IMAGES or sum(im.width*im.height*4 for im in images.values())>MAX_WORKSPACE_BYTES:
                raise ServiceError('Workspace is full. Undo a result or reset the example.')
            session.box=box
            for iid in draft['output'].new_image_ids:session.origins[iid]=draft['call']['name']
            session.preview=None;session.revision+=1
            return self.describe(sid,session)

    def undo(self,sid,revision):
        session=self.session(sid)
        with session.lock:
            if revision!=session.revision:raise ServiceError('Workspace changed. Try again.',409)
            ids=session.box.image_ids()
            if len(ids)>session.initial_count:
                session.box.state.images.pop(ids[-1]);session.origins.pop(ids[-1],None)
                session.revision+=1;session.preview=None
            return self.describe(sid,session)

    def png(self,sid,kind,identifier,channel):
        session=self.session(sid)
        with session.lock:
            if kind=='images':
                image=session.box.images_by_id().get(identifier)
            else:
                draft=session.preview
                image=draft['output'].produced_images[0] if draft and draft['token']==identifier else None
            if image is None:raise ServiceError('Image no longer available.',404)
            if channel=='alpha.png':image=image.getchannel('A')
            elif channel=='thumbnail.png':
                image=image.copy();image.thumbnail((160,160),self.Image.Resampling.LANCZOS)
            elif channel!='rgba.png':raise ServiceError('Unknown image channel.',404)
            buffer=io.BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()

    def close(self,sid):
        with self.lock:self.sessions.pop(sid,None)


def handler_for(service, allowed_origins=()):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(service.site),**kwargs)
        def origin_allowed(self):
            origin=self.headers.get('Origin')
            host=self.headers.get('Host')
            return not origin or origin in allowed_origins or origin in (f'http://{host}',f'https://{host}')
        def end_headers(self):
            origin=self.headers.get('Origin')
            if origin and self.origin_allowed():
                self.send_header('Access-Control-Allow-Origin',origin)
                self.send_header('Vary','Origin')
            super().end_headers()
        def do_OPTIONS(self):
            if not self.origin_allowed():return self.send_data({'error':{'message':'Origin not allowed.'}},403)
            self.send_response(204)
            self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers','Content-Type, X-Mosaic-Playground')
            self.send_header('Access-Control-Max-Age','600')
            self.send_header('Content-Length','0');self.end_headers()
        def send_data(self,data,status=200,content_type='application/json'):
            if content_type=='application/json':data=json.dumps(data,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path=='/api/gym/config':return self.send_data(service.config())
                if path.startswith('/api/gym/'):
                    parts=path.strip('/').split('/')
                    if len(parts)==7 and parts[:3]==['api','gym','sessions'] and parts[4] in ['images','previews']:
                        return self.send_data(service.png(parts[3],parts[4],parts[5],parts[6]),content_type='image/png')
                    raise ServiceError('Unknown endpoint.',404)
                translated=Path(self.translate_path(self.path)).resolve()
                if not translated.is_relative_to(service.site):raise ServiceError('Not found.',404)
                if translated.is_dir() and not (translated/'index.html').exists():raise ServiceError('Not found.',404)
                return super().do_GET()
            except ServiceError as exc:self.send_data({'error':{'message':str(exc)}},exc.status)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            try:
                origin=self.headers.get('Origin')
                if not self.origin_allowed():raise ServiceError('Origin not allowed.',403)
                if self.headers.get('X-Mosaic-Playground')!='1' or self.headers.get_content_type()!='application/json':raise ServiceError('JSON request required.',415)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ServiceError('Invalid request size.',413)
                data=json.loads(self.rfile.read(length),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Non-finite number')))
                if not isinstance(data,dict):raise ServiceError('Expected a JSON object.')
                parts=urlsplit(self.path).path.strip('/').split('/')
                if parts==['api','gym','sessions']:result=service.create(data.get('example'))
                elif len(parts)==5 and parts[:3]==['api','gym','sessions']:
                    sid,action=parts[3:]
                    if action=='preview':result=service.preview(sid,data.get('call',{}),data.get('revision'))
                    elif action=='commit':result=service.commit(sid,data.get('preview_id'),data.get('revision'))
                    elif action=='undo':result=service.undo(sid,data.get('revision'))
                    elif action=='close':service.close(sid);result={'closed':True}
                    else:raise ServiceError('Unknown endpoint.',404)
                else:raise ServiceError('Unknown endpoint.',404)
                self.send_data(result)
            except ServiceError as exc:self.send_data({'error':{'message':str(exc)}},exc.status)
            except (ValueError,TypeError,AttributeError):self.send_data({'error':{'message':'Invalid request.'}},400)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,*args,**kwargs):
        self.slots=threading.BoundedSemaphore(24)
        super().__init__(*args,**kwargs)
    def get_request(self):
        request,address=super().get_request();request.settimeout(30);return request,address
    def process_request(self,request,client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request);return
        try:super().process_request(request,client_address)
        except BaseException:self.slots.release();raise
    def process_request_thread(self,request,client_address):
        try:super().process_request_thread(request,client_address)
        finally:self.slots.release()


def main(default_site=DEFAULT_SITE,default_source=DEFAULT_SOURCE):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--allowed-origin',action='append',default=[])
    parser.add_argument('--site',type=Path,default=default_site)
    parser.add_argument('--toolbox-src',type=Path,default=default_source)
    args=parser.parse_args();service=GymService(args.site,args.toolbox_src)
    server=BoundedHTTPServer((args.host,args.port),handler_for(service,args.allowed_origin))
    print(f'MosaicGym Toolbox: http://{args.host}:{args.port}/#alpha-canvas',flush=True)
    print(f'Toolbox source: {service.source}; digest: {service.digest}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
