import {validatePack} from './models.js';

// Family adapters own their state semantics. The host owns switching and cleanup.
export class ModelHost {
  constructor(packs, loadAdapter, environment={}) {
    this.packs=new Map(packs.map(p=>[p.id,validatePack(p)]));
    this.loadAdapter=loadAdapter;this.environment=environment;
    this.active=null;this.revision=0;this.controller=null;this.sessions=new Map();this.queue=Promise.resolve();
  }
  async select(id) {
    const pack=this.packs.get(id);if(!pack)throw new Error('Unknown model pack.');
    const revision=++this.revision;this.controller?.abort();
    const operation=async()=>{
      if(revision!==this.revision)return;
      if(this.current&&this.active?.getState)this.sessions.set(this.current.id,this.active.getState());
      this.active?.dispose?.();this.active=null;
      const controller=new AbortController();this.controller=controller;
      const adapter=await this.loadAdapter(pack);
      if(controller.signal.aborted||revision!==this.revision)return;
      if(typeof adapter.mount!=='function')throw new Error('A model adapter must export mount(pack, environment).');
      this.environment.beforeMount?.(pack);
      let mounted;
      try {mounted=await adapter.mount(pack,{...this.environment,signal:controller.signal,state:this.sessions.get(pack.id)});}
      catch(error){if(controller.signal.aborted||revision!==this.revision)return;throw error;}
      if(revision!==this.revision||controller.signal.aborted){mounted?.dispose?.();return;}
      this.active=mounted||{};this.current=pack;this.environment.afterMount?.(pack);return pack;
    };
    const result=this.queue.catch(()=>{}).then(operation);this.queue=result;return result;
  }
  dispose(){this.revision++;this.controller?.abort();this.active?.dispose?.();this.active=null;}
}
