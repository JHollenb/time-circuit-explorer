import {registry} from './models.js';
import {ModelHost} from './model-host.js';
import {installDialog} from './components.js';

const main=document.getElementById('main'),select=document.getElementById('model');
installDialog();
const host=new ModelHost(registry,pack=>import(pack.adapter),{
  beforeMount(pack){document.getElementById('detail-dialog').close();main.innerHTML='<div class="pack-loading">Opening the computation…</div>';main.setAttribute('aria-busy','true');main.dataset.model=pack.id;document.title=`Time Circuit — ${pack.name}`;},
  afterMount(pack){select.value=pack.id;const url=new URL(location.href);url.searchParams.set('model',pack.id);history.replaceState(null,'',url);}
});
select.replaceChildren(...registry.map(pack=>new Option(pack.name,pack.id)));
select.onchange=()=>host.select(select.value).catch(error=>{
  main.innerHTML='<div class="error-notice" role="alert"></div>';main.firstChild.textContent=`Model pack could not load: ${error.message}`;main.setAttribute('aria-busy','false');
});
const requested=new URL(location.href).searchParams.get('model');
host.select(registry.some(p=>p.id===requested)?requested:registry[0].id).catch(error=>{
  main.innerHTML='<div class="error-notice" role="alert"></div>';main.firstChild.textContent=`Explorer could not start: ${error.message}`;main.setAttribute('aria-busy','false');
});
