import {createServer} from 'node:http';
import {createReadStream, statSync, watch} from 'node:fs';
import {extname,resolve,sep} from 'node:path';
import {build,outputDirectory,projectRoot} from './build.mjs';

build();
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml','.txt':'text/plain; charset=utf-8'};
const port=Number(process.env.PORT||4177);
const server=createServer((request,response)=>{
  try {
    const pathname=decodeURIComponent(new URL(request.url,'http://127.0.0.1').pathname);
    const target=resolve(outputDirectory,`.${pathname.endsWith('/')?pathname+'index.html':pathname}`);
    if(!target.startsWith(outputDirectory+sep)){response.writeHead(403);response.end('Forbidden');return;}
    if(!statSync(target).isFile())throw new Error('Not a file');
    response.writeHead(200,{'Content-Type':mime[extname(target)]||'application/octet-stream','Cache-Control':'no-store'});
    createReadStream(target).pipe(response);
  }catch {response.writeHead(404);response.end('Not found');}
});
const watchers=[];let pending;
for(const path of ['viz/src','viz/public']) {
  try {watchers.push(watch(resolve(projectRoot,path),{recursive:true},()=>{
    clearTimeout(pending);pending=setTimeout(()=>{try{build();console.log('Rebuilt. Refresh the browser.');}catch(error){console.error(error.message);}},80);
  }));}catch{console.log(`File watching unavailable for ${path}; run npm run build after edits.`);}
}
server.on('error',error=>{console.error(error.message);for(const watcher of watchers)watcher.close();process.exitCode=1;});
server.listen(port,'127.0.0.1',()=>console.log(`Local: http://127.0.0.1:${port}`));
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{clearTimeout(pending);for(const watcher of watchers)watcher.close();server.close(()=>process.exit(0));});
