import {controlsFromSpecs} from './gym-controls.mjs?v=de4bc0ece7';
const demo=document.querySelector('#alpha-canvas'),$=s=>demo.querySelector(s);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let TOOLS=[],examples=[],assets=[],initialCount=0,targets=[],args={},tool='rotate',channel='rgba',result=null,revision=0,loadVersion=0,session=null,busy=false,pending=null,debounce;
const imageLabel=id=>id?.replace(/^img_0*/, 'img')||'';
const apiBase=['localhost','127.0.0.1','[::1]'].includes(location.hostname)?'/api/gym':demo.dataset.toolboxApi||'/api/gym';
const apiOrigin=new URL(apiBase,location.href).origin;
function resolveImage(image){return {...image,...Object.fromEntries(['rgba','alpha','thumbnail'].map(key=>[key,new URL(image[key],apiOrigin).href]))};}
async function request(path,body,keepalive=false){
 const response=await fetch(apiBase+path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json','X-Mosaic-Playground':'1'},body:JSON.stringify(body),keepalive});
 let data;try{data=await response.json();}catch{throw Error('MosaicGym Toolbox service unavailable.');}
 if(!response.ok||data.error)throw Error(data.error?.message||'MosaicGym Toolbox request failed.');
 if(data.image)data.image=resolveImage(data.image);
 if(data.images)data.images=data.images.map(resolveImage);
 return data;
}
const sessionPath=(sid,action)=>'/sessions/'+sid+'/'+action;
function closeSession(sid){if(sid)request(sessionPath(sid,'close'),{},true).catch(()=>{});}
window.addEventListener('pagehide',event=>{if(!event.persisted)closeSession(session?.id);});
function applyWorkspace(data){session={id:data.session_id,revision:data.revision};assets=data.images;initialCount=data.initial_count;}
function lockEditing(locked){for(const id of ['gym-workspace','gym-inputs','gym-parameters'])$('#'+id).inert=locked;$('#gym-tool').disabled=locked;$('#gym-undo').disabled=locked||assets.length<=initialCount;$('#gym-reset').disabled=locked;$('#gym-example').disabled=locked;}
const definition=()=>TOOLS.find(item=>item.id===tool);
function status(message='',error=false){$('#gym-status').textContent=message;$('#gym-status').hidden=!message;$('#gym-status').classList.toggle('error',error);}
function showResult(){if(!result)return;const image=result.image;$('#gym-result').src=image[channel];$('#gym-result').hidden=false;$('#gym-size').textContent=`${image.width} × ${image.height}`;$('#gym-result').alt=channel==='alpha'?'Result alpha channel: black is transparent, white is opaque':'MosaicGym Toolbox result on transparent canvas';}
function showWorkspace(){
 $('#gym-workspace').innerHTML=assets.map(a=>`<button class="gym-asset" type="button" data-asset="${a.id}" aria-label="Use ${imageLabel(a.id)}" aria-pressed="${targets.includes(a.id)}"><span class="gym-thumb"><img src="${a.thumbnail}" alt="${imageLabel(a.id)}"></span><span><strong>${imageLabel(a.id)}</strong><small>${a.width} × ${a.height}</small></span><span class="gym-origin">${esc(TOOLS.find(t=>t.id===a.origin)?.label||a.origin)}</span></button>`).join('');
 $('#gym-undo').disabled=assets.length<=initialCount;
}
function showInputs(){
 const d=definition(),labels=tool==='overlay_images'?['Background','Foreground']:d.inputs===2?['Image 1','Image 2']:['Image'];
 const options=selected=>assets.map(a=>`<option value="${a.id}" ${selected===a.id?'selected':''}>${imageLabel(a.id)} · ${a.width} × ${a.height}</option>`).join('');
 $('#gym-inputs').innerHTML=d.inputs?labels.map((label,i)=>`<label class="gym-field">${label}<select data-input-index="${i}" aria-label="${label}">${options(targets[i])}</select></label>`).join('')+(d.inputs===2?'<button class="gym-swap" type="button" id="gym-swap">Swap images</button>':''):
 `<fieldset class="gym-image-choices"><legend>Images</legend>${assets.map(a=>`<label><input type="checkbox" data-collage-image="${a.id}" ${targets.includes(a.id)?'checked':''}>${imageLabel(a.id)}${targets.includes(a.id)?`<span>${targets.indexOf(a.id)+1}</span>`:''}</label>`).join('')}</fieldset>`;
 for(const el of demo.querySelectorAll('[data-input-index]'))el.onchange=()=>{targets[Number(el.dataset.inputIndex)]=el.value;markTargets();schedule();};
 for(const el of demo.querySelectorAll('[data-collage-image]'))el.onchange=()=>{targets=targets.filter(id=>id!==el.dataset.collageImage);if(el.checked)targets.push(el.dataset.collageImage);showInputs();markTargets();schedule();};
 if($('#gym-swap'))$('#gym-swap').onclick=()=>{targets.reverse();showInputs();markTargets();schedule();};
}
function markTargets(){for(const b of demo.querySelectorAll('[data-asset]'))b.setAttribute('aria-pressed',String(targets.includes(b.dataset.asset)));}
function showParameters(){
 $('#gym-parameters').innerHTML=definition().params.map(({key,label,min,max,step,sliderMin,sliderMax})=>`<div class="gym-parameter"><label for="gym-param-${key}">${label}</label><input id="gym-param-${key}" type="number" data-param="${key}" min="${min}" max="${max}" step="${step}" value="${args[key]}"><input type="range" data-slider="${key}" aria-label="${label} slider" min="${sliderMin}" max="${sliderMax}" step="${step}" value="${args[key]}"></div>`).join('');
 if(tool==='flip')$('#gym-parameters').innerHTML='<label class="gym-field">Direction<select id="gym-flip"><option value="horizontal">Horizontal</option><option value="vertical">Vertical</option></select></label>';
 if(tool==='apply_homography_transformation')$('#gym-parameters').innerHTML='<fieldset class="gym-matrix"><legend>Homography matrix</legend>'+args.matrix.map((value,i)=>`<input type="number" data-matrix="${i}" aria-label="Matrix row ${Math.floor(i/3)+1} column ${i%3+1}" step="any" value="${value}">`).join('')+'</fieldset>';
 for(const input of demo.querySelectorAll('[data-param],[data-slider]'))input.oninput=()=>{
  const key=input.dataset.param||input.dataset.slider;args[key]=input.value===''?'':Number(input.value);
  const pair=$(`[${input.dataset.param?'data-slider':'data-param'}="${key}"]`);if(pair)pair.value=input.value;schedule();
 };
 if($('#gym-flip')){$('#gym-flip').value=args.mode;$('#gym-flip').onchange=e=>{args.mode=e.target.value;schedule();};}
 for(const input of demo.querySelectorAll('[data-matrix]'))input.oninput=()=>{args.matrix[Number(input.dataset.matrix)]=input.value===''?NaN:Number(input.value);schedule();};
}
function setTool(name){
 const previous=targets[0];tool=name;$('#gym-tool').value=name;const d=definition();
 args=Object.fromEntries(d.params.map(p=>[p.key,p.value]));
 if(name==='flip')args.mode='horizontal';
 if(name==='apply_homography_transformation')args.matrix=[1,.15,0,.05,1,0,.0004,.00015,1];
 if(name==='resize'&&assets.length){const image=assets.find(a=>a.id===previous)||assets[0];args.width=image.width;args.height=image.height;}
 if(d.inputs===1)targets=[previous||assets[0]?.id];
 else if(d.inputs===2)targets=name==='overlay_images'&&previous!==assets[0]?.id?[assets[0]?.id,previous]:[previous||assets[0]?.id,assets.find(a=>a.id!==previous)?.id||assets[0]?.id];
 else targets=assets.slice(0,Math.min(4,assets.length)).map(a=>a.id);
 if(name==='make_collage')args.cols=Math.max(2,targets.length);
 showInputs();showParameters();markTargets();schedule();
}
async function sendPending(){
 if(busy||!pending)return;busy=true;const job=pending;pending=null;
 try{
  const data=await request(sessionPath(job.session.id,'preview'),{revision:job.session.revision,call:job.call});
  if(job.revision!==revision||session?.id!==job.session.id)return;
  result=data;showResult();status();$('#gym-add').disabled=false;
 }catch(error){
  if(job.revision!==revision||session?.id!==job.session.id)return;
  status(error.message,true);$('#gym-result').hidden=true;$('#gym-size').textContent='';
 }finally{busy=false;if(job.revision===revision)$('#gym-stage').classList.remove('computing');sendPending();}
}
function schedule(){
 clearTimeout(debounce);revision++;pending=null;result=null;$('#gym-add').disabled=true;$('#gym-stage').classList.add('computing');
 const call={name:tool,arguments:structuredClone(args),target_images:[...targets]};
 $('#gym-call-json').textContent=JSON.stringify(call,null,2);const current=revision;
 debounce=setTimeout(()=>{
  if(!session)return;
  pending={revision:current,session:{...session},call};status('Updating…');sendPending();
 },180);
}
async function loadExample(id){
 const version=++loadVersion;revision++;pending=null;clearTimeout(debounce);result=null;$('#gym-add').disabled=true;
 closeSession(session?.id);session=null;$('#gym-result').hidden=true;$('#gym-size').textContent='';status('Loading images…');lockEditing(true);
 // Keep example navigation available; obsolete session responses are discarded.
 $('#gym-example').disabled=false;
 try{
  const data=await request('/sessions',{example:id});
  if(version!==loadVersion){closeSession(data.session_id);return;}
  applyWorkspace(data);targets=[assets[0].id];showWorkspace();setTool('rotate');
  const example=examples.find(e=>e.id===id);
  $('#gym-credits').innerHTML=example.credits.map(c=>`<p><a href="${esc(c.source_url)}">${esc(c.source_dataset)}</a> · ${esc(c.attribution||'')} · <a href="${esc(c.license_url)}">${esc(c.source_license)}</a></p>`).join('');
  lockEditing(false);
 }catch(error){if(version!==loadVersion)return;status(error.message,true);$('#gym-reset').disabled=false;}
}
$('#gym-tool').onchange=e=>setTool(e.target.value);
$('#gym-example').onchange=e=>loadExample(e.target.value);
$('#gym-reset').onclick=()=>loadExample($('#gym-example').value);
$('#gym-workspace').onclick=e=>{
 const b=e.target.closest('[data-asset]');if(!b)return;
 if(definition().inputs===0){targets=targets.includes(b.dataset.asset)?targets.filter(id=>id!==b.dataset.asset):[...targets,b.dataset.asset];}
 else targets[0]=b.dataset.asset;
 showInputs();markTargets();schedule();
};
$('#gym-add').onclick=async()=>{
 if(!result||!session)return;const preview=result,previous=[...targets],sid=session.id,current=++revision;
 clearTimeout(debounce);pending=null;result=null;$('#gym-add').disabled=true;lockEditing(true);
 try{
  const data=await request(sessionPath(sid,'commit'),{preview_id:preview.preview_id,revision:session.revision});
  if(current!==revision)return;
  applyWorkspace(data);const id=preview.image.id;
  targets=definition().inputs===1?[id]:definition().inputs===2?[previous[0],id]:[...previous,id];
  if(tool==='make_collage')args.cols=Math.max(args.cols,Math.ceil(targets.length/args.rows));
  showWorkspace();showInputs();showParameters();schedule();
 }catch(error){status(error.message,true);}finally{if(session?.id===sid)lockEditing(false);}
};
$('#gym-undo').onclick=async()=>{
 if(!session||assets.length<=initialCount)return;const sid=session.id,current=++revision,removed=assets.at(-1).id;
 clearTimeout(debounce);pending=null;result=null;$('#gym-add').disabled=true;lockEditing(true);
 try{
  const data=await request(sessionPath(sid,'undo'),{revision:session.revision});if(current!==revision)return;
  applyWorkspace(data);targets=[...new Set(targets.map(id=>id===removed?assets.at(-1).id:id))];
  if(definition().inputs===2&&targets.length<2)targets.push(assets[0].id);
  showWorkspace();showInputs();schedule();
 }catch(error){status(error.message,true);}finally{if(session?.id===sid)lockEditing(false);}
};
for(const button of demo.querySelectorAll('[data-gym-channel]'))button.onclick=()=>{
 channel=button.dataset.gymChannel;for(const b of demo.querySelectorAll('[data-gym-channel]'))b.setAttribute('aria-pressed',String(b===button));
 $('#gym-alpha-legend').innerHTML=channel==='alpha'?'<i class="gym-alpha-zero"></i>α = 0 <i class="gym-alpha-one"></i>α = 1':'<i class="checker-surface"></i>Transparent';showResult();
};
async function init(){
 $('#gym-reset').disabled=true;
 try{
  const config=await request('/config');TOOLS=controlsFromSpecs(config.tools);examples=config.examples;
  $('#gym-example').innerHTML=examples.map(e=>`<option value="${e.id}">${esc(e.label)}</option>`).join('');$('#gym-example').disabled=false;
  $('#gym-tool').innerHTML=TOOLS.map(t=>`<option value="${t.id}">${esc(t.label)}</option>`).join('');
  $('#gym-reset').textContent='Reset';$('#gym-reset').onclick=()=>loadExample($('#gym-example').value);
  await loadExample(examples[0].id);
 }catch(error){status(error.message,true);$('#gym-add').disabled=true;$('#gym-reset').disabled=false;$('#gym-reset').textContent='Retry';$('#gym-reset').onclick=init;}
}
const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){observer.disconnect();init();}},{rootMargin:'200px'});observer.observe(demo);
