import {registry} from './models.js';
import {ModelHost} from './model-host.js';

const main=document.getElementById('main'),template=main.innerHTML,select=document.getElementById('model');
const host=new ModelHost(registry,pack=>import(pack.adapter),{
  beforeMount(pack){main.innerHTML=template;document.title=`Time Circuit — ${pack.name}`;},
  afterMount(pack){select.value=pack.id;}
});
select.replaceChildren(...registry.map(pack=>new Option(pack.name,pack.id)));
select.onchange=()=>host.select(select.value).catch(error=>{
  document.getElementById('comparison-status').textContent=`Model pack could not load: ${error.message}`;
});
host.select(registry[0].id).catch(error=>{
  document.getElementById('comparison-status').textContent=`Explorer could not start: ${error.message}`;
});
