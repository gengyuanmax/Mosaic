import {buildRounds,frameAt,moveFrame} from './replay-state.mjs?v=92f66c88b1';
import {createEpisodeLoader,filterEpisodes,createSelectionGate,validateCollections,collectionOptions,chooseCollection,chooseEpisode} from './episode-library.mjs?v=0004d09523';
const $ = selector => document.querySelector(selector);
const paths = {
  user:'<circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/>',
  workspace:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 17h7m-3.5-3.5v7"/>',
  tool:'<path d="m14 6 4-4a6 6 0 0 0-8 8L3 17a3 3 0 0 0 4 4l7-7a6 6 0 0 0 8-8l-4 4-4-4Z"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
  up:'<path d="M12 20V4m-6 6 6-6 6 6"/>',
  restart:'<path d="M3 10a9 9 0 1 1 1 7M3 4v6h6"/>',
  paper:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
  code:'<path d="m8 7-5 5 5 5m8-10 5 5-5 5M14 4l-4 16"/>',
  model:'<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  data:'<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  left:'<path d="m14 6-6 6 6 6"/>',right:'<path d="m10 6 6 6-6 6"/>',arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
  down:'<path d="m6 9 6 6 6-6"/>',play:'<path d="m8 5 11 7-11 7V5Z"/>',pause:'<path d="M8 5v14M16 5v14"/>',
  copy:'<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>',close:'<path d="m6 6 12 12M6 18 18 6"/>',expand:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.model}</svg>`;
function hydrateIcons(root=document) {root.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));}
hydrateIcons();
const escapeHTML = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label = id => /^img_\d+$/.test(id)?'img'+Number(id.split('_')[1]):String(id);
const toolLabel = name => ({overlay_images:'Overlay',crop:'Crop',make_collage:'Collage'}[name] || name);
let record,rounds=[],current=0,phaseIndex=0,selected=null,timer=null,replayMode=false,scrollingTo=null,scrollFrame=null;
const imageMeta = id => record.images.find(image=>image.id===id);
const frame = () => frameAt(rounds,current,phaseIndex);
const refs = ids => ids.map(id=>imageMeta(id)?`<button class="image-reference" data-select="${escapeHTML(id)}">${escapeHTML(label(id))}</button>`:`<code>${escapeHTML(label(id))}</code>`).join('');
const feedbackText = result => result.error?(typeof result.error==='string'?result.error:result.error.message||result.error.code):result.tool_feedback||'';
function quote(text,visible) {
  return text.split(/\n\s*\n/).map(paragraph=>'<p>'+escapeHTML(paragraph).replace(/\[img(\d+)\]/g,(match,n)=>{
    const id=record.images.find(image=>label(image.id)==='img'+Number(n))?.id;
    return id&&visible.includes(id)?`<button class="inline-reference" data-select="${id}">${match}</button>`:match;
  })+'</p>').join('');
}
function imageFigure(id,{crop=null,result=false}={}) {
  const meta=imageMeta(id);
  if(!meta)return `<span class="unavailable-image">${escapeHTML(label(id))} unavailable</span>`;
  if(crop&&!['x1','y1','x2','y2'].every(key=>Number.isFinite(crop[key])))crop=null;
  const box=crop?`<span class="crop-selection" style="left:${crop.x1*100}%;top:${crop.y1*100}%;width:${(crop.x2-crop.x1)*100}%;height:${(crop.y2-crop.y1)*100}%"></span>`:'';
  return `<figure class="env-figure ${crop?'crop-figure':''} ${result?'result-figure':''}"><button class="env-image" data-zoom="${id}" aria-label="Enlarge ${label(id)}"><img src="assets/${meta.file}" alt="${label(id)}">${box}</button><figcaption><span>${label(id)}</span>${icon('expand')}</figcaption></figure>`;
}
function renderAgent() {
  $('#agent-content').innerHTML=rounds.map(r=>{
    const messages=r.reasoning||r.decision?[`<li class="conversation-reasoning" data-phase="reasoning"><h5>Reasoning</h5><div class="agent-prose">${quote(r.reasoning,r.input)}</div>${r.decision?`<div class="conversation-decision"><h6>${r.answer?'Conclusion':'Decision'}</h6><div class="agent-prose">${quote(r.decision,r.input)}</div></div>`:''}</li>`]:[];
    for(const [index,call] of r.calls.entries()){
      messages.push(`<li class="conversation-call" data-phase="action"><h5>Tool call</h5><div class="conversation-function"><code>${escapeHTML(call.name)}</code><div>${refs(call.target_images)}</div></div><dl class="conversation-arguments">${Object.entries(call.arguments).map(([key,value])=>`<div><dt>${escapeHTML(key)}</dt><dd>${escapeHTML(JSON.stringify(value))}</dd></div>`).join('')}</dl></li>`);
      const result=r.results[index];
      messages.push(`<li class="conversation-response" data-phase="feedback"><h5>${escapeHTML(activeCollection.environment.label)}</h5><div class="conversation-receipt"><span>${icon(result.error?'close':'check')}${result.error?'Error':'Success'}</span>${refs(result.new_image_ids||[])}</div>${feedbackText(result)?`<p>${escapeHTML(feedbackText(result))}</p>`:''}</li>`);
    }
    if(r.answer)messages.push(`<li class="conversation-answer" data-phase="answer"><h5>Final answer</h5><strong>${escapeHTML(r.answer)}</strong></li>`);
    else if(r.index===rounds.length-1)messages.push('<li class="conversation-status" data-phase="status">No final answer</li>');
    return `<section class="trajectory-step" data-step="${r.index}" aria-labelledby="trajectory-step-${r.index}"><h4 class="trajectory-step-heading" id="trajectory-step-${r.index}">Step ${r.index+1}</h4><ol class="conversation">${messages.join('')}</ol></section>`;
  }).join('');
}
function renderEnvironment(f) {
  const r=f.round,canvas=$('#environment-canvas');
  if(selected&&!f.visible.includes(selected))selected=null;
  $('#reset-view').hidden=!selected;
  canvas.className='environment-canvas';
  if(selected) {
    const m=imageMeta(selected);
    $('#canvas-label').textContent=label(selected)+' · '+(m.tool?toolLabel(m.tool):'User input');
    canvas.classList.add('single-image');canvas.innerHTML=imageFigure(selected);
  } else if(r.calls.length&&(f.phase==='action'||f.phase==='feedback')) {
    const feedback=f.phase==='feedback';
    $('#canvas-label').textContent=feedback?'Tool result':'Action on workspace';
    canvas.classList.add('operation-stack');
    canvas.innerHTML=r.calls.map((call,index)=>{
      const result=r.results[index];
      if(feedback&&result.error)return `<div class="tool-failure"><code>${escapeHTML(call.name)}</code><p>${escapeHTML(feedbackText(result))}</p></div>`;
      return `<div class="operation-view">${r.calls.length>1?`<code class="operation-name">${escapeHTML(call.name)}</code>`:''}<div class="operation-sources ${call.target_images.length>1?'multiple':''}">${call.target_images.map(id=>imageFigure(id,{crop:call.name==='crop'?call.arguments:null})).join('')}</div><div class="operation-arrow">${icon('arrow')}</div>${feedback?`<div class="operation-result">${(result.new_image_ids||[]).map(id=>imageFigure(id,{result:true})).join('')}${!result.new_image_ids?.length?`<p class="tool-feedback">${escapeHTML(feedbackText(result)||'No image returned')}</p>`:''}</div>`:`<div class="pending-result">${icon('workspace')}<span>Tool result</span></div>`}</div>`;
    }).join('');
  } else {
    const latest=rounds.slice(0,current).findLast(round=>round.added.length);
    const focusIds=latest?latest.added:r.input;
    $('#canvas-label').textContent=current?'Tool result':'Input images';
    canvas.classList.add(focusIds.length===1?'single-image':'initial-images');
    canvas.innerHTML=focusIds.map(id=>imageFigure(id)).join('');
  }
  $('#workspace-images').innerHTML=f.visible.map(id=>{
    const m=imageMeta(id),fresh=f.added.includes(id),target=f.phase==='action'&&r.calls.some(call=>call.target_images.includes(id));
    return `<button class="workspace-image ${fresh?'new-image':''} ${target?'target-image':''}" data-select="${id}" aria-pressed="${selected===id}" aria-label="View ${label(id)}"><div class="workspace-preview"><img src="assets/${m.thumbnail||m.file}" alt="" loading="lazy"></div><span>${label(id)}</span>${fresh?'<b>New</b>':target?'<b>Target</b>':''}</button>`;
  }).join('');

}
function render() {
  if(!record)return;
  document.querySelectorAll('[data-round]').forEach(button=>{
    if(Number(button.dataset.round)===current)button.setAttribute('aria-current','step');
    else button.removeAttribute('aria-current');
  });
  document.querySelectorAll('.trajectory-step').forEach(section=>{
    const active=Number(section.dataset.step)===current;
    section.classList.toggle('current-step',active);
    section.querySelectorAll('[data-phase]').forEach(message=>message.classList.toggle('playing-message',active&&replayMode&&message.dataset.phase===frame().phase));
  });
  renderEnvironment(frame());
  const nav=$('#round-nav'),active=nav.querySelector('[aria-current=step]');
  if(nav.scrollWidth>nav.clientWidth)nav.scrollTo({left:Math.max(0,active.offsetLeft-12),behavior:'auto'});
}
function stop(){clearInterval(timer);timer=null;$('#play').innerHTML=icon('play')+'<span>Play</span>';}
function locateStep(message=false){
  const viewport=$('#agent-content'),section=$(`.trajectory-step[data-step="${current}"]`);
  const target=message?section.querySelector(`[data-phase="${frame().phase}"]`)||section:section;
  const top=Math.max(0,Math.min(viewport.scrollTop+target.getBoundingClientRect().top-viewport.getBoundingClientRect().top,viewport.scrollHeight-viewport.clientHeight));
  scrollingTo=null;
  if(Math.abs(viewport.scrollTop-top)>1){scrollingTo=top;viewport.scrollTo({top,behavior:'instant'});}
}
function showStep(index,locate=true){stop();replayMode=false;current=index;phaseIndex=rounds[index].phases.length-1;selected=null;render();if(locate)locateStep();}
$('#agent-content').addEventListener('scroll',()=>{
  if(scrollFrame!==null)return;
  scrollFrame=requestAnimationFrame(()=>{
    scrollFrame=null;
    const viewport=$('#agent-content');
    if(scrollingTo!==null){const expected=scrollingTo;scrollingTo=null;if(Math.abs(viewport.scrollTop-expected)<2)return;}
    const marker=viewport.getBoundingClientRect().top+48;
    const sections=[...viewport.querySelectorAll('.trajectory-step')];
    const index=Math.max(0,sections.findLastIndex(section=>section.getBoundingClientRect().top<=marker));
    if(index!==current||replayMode)showStep(index,false);
  });
},{passive:true});
$('#round-nav').onclick=e=>{const b=e.target.closest('[data-round]');if(b){showStep(Number(b.dataset.round));$(`#round-nav [data-round="${current}"]`).focus({preventScroll:true});}};
$('#play').onclick=()=>{
  if(!record)return;
  if(timer){stop();return;}
  if(!replayMode){phaseIndex=rounds[current].phases.indexOf('reasoning');selected=null;}
  replayMode=true;render();locateStep(true);
  $('#play').innerHTML=icon('pause')+'<span>Pause</span>';
  timer=setInterval(()=>{
    [current,phaseIndex]=moveFrame(rounds,current,phaseIndex);
    if(rounds[current].phases[phaseIndex]==='input')phaseIndex++;
    selected=null;
    const ended=current===rounds.length-1&&phaseIndex===rounds[current].phases.length-1;
    if(ended){stop();replayMode=false;}
    render();locateStep(true);
  },4200);
};
$('#demo').onclick=e=>{
  const select=e.target.closest('[data-select]');
  if(select?.closest('.trajectory-step'))showStep(Number(select.closest('.trajectory-step').dataset.step),false);
  if(select&&frame().visible.includes(select.dataset.select)){stop();selected=select.dataset.select;renderEnvironment(frame());return;}
  const zoom=e.target.closest('[data-zoom]');
  if(zoom){stop();const m=imageMeta(zoom.dataset.zoom);const file=m.width*m.height>16_000_000?m.file:(m.full_file||m.file);$('#zoom-image').src='assets/'+file;$('#zoom-image').alt=label(m.id);$('#zoom-label').textContent=label(m.id)+(m.tool?' ← '+m.tool+'('+m.source.map(label).join(', ')+')':' · User input');$('#zoom').showModal();return;}
};
$('#reset-view').onclick=()=>{selected=null;renderEnvironment(frame());};
$('#close-zoom').onclick=()=>$('#zoom').close();
$('#zoom').onclick=e=>{if(e.target===$('#zoom')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('#zoom').close();}};
$('#copy').onclick=async()=>{try{await navigator.clipboard.writeText($('#bibtex').textContent);$('#copy-status').textContent='Copied.';}catch{$('#copy-status').textContent='Select the citation to copy.';}};
function mountEpisode(data,collection){
  stop();replayMode=false;current=0;selected=null;scrollingTo=null;
  if(scrollFrame!==null){cancelAnimationFrame(scrollFrame);scrollFrame=null;}
  if($('#zoom').open)$('#zoom').close();
  record=data;rounds=buildRounds(data);phaseIndex=rounds[0].phases.length-1;
  const agentName=collection.model.role_label||collection.model.label;
  $('.agent-lane h3').textContent=agentName;
  $('.agent-lane').setAttribute('aria-label',agentName);
  $('#agent-content').setAttribute('aria-label',agentName+' trajectory');
  $('.environment-lane h3').textContent=collection.environment.label;
  $('.environment-lane').setAttribute('aria-label',collection.environment.label);
  $('#collection-attribution').href=collection.attribution;
  $('#collection-license').href=collection.license;
  const optionStart=data.question.search(/^\([A-Z]\)/m);
  $('#user-question').textContent=(optionStart<0?data.question:data.question.slice(0,optionStart)).trim();
  $('#question-options').textContent=optionStart<0?'':data.question.slice(optionStart).trim();
  $('.question-options').hidden=optionStart<0;$('.question-options').open=false;
  $('#ground-truth').textContent=data.ground_truth;
  const inputs=data.images.filter(m=>!m.tool);
  $('#user-images').classList.toggle('single-source',inputs.length===1);
  $('#user-images').innerHTML=inputs.map(m=>`<figure class="source-image"><button data-zoom="${m.id}" aria-label="Enlarge input ${label(m.id)}"><img src="assets/${m.file}" alt="${label(m.id)}"></button><figcaption><span>${label(m.id)}</span>${icon('expand')}</figcaption></figure>`).join('');
  $('#round-nav').innerHTML=rounds.map((_,i)=>`<li><button data-round="${i}">Step ${i+1}</button></li>`).join('');
  $('#episode-credits').innerHTML=(data.credits||[]).map(credit=>`<p><a href="${escapeHTML(credit.source_url)}">${escapeHTML(credit.source_dataset)}</a>${credit.attribution?' · '+escapeHTML(credit.attribution):''} · <a href="${escapeHTML(credit.license_url)}">${escapeHTML(credit.source_license)}</a>${credit.modifications?'<br>'+escapeHTML(credit.modifications):''}</p>`).join('');
  renderAgent();
  $('#agent-content').scrollTop=0;
  $('.environment-lane').scrollTop=0;
  render();
}

const loadEpisode=createEpisodeLoader();
const beginSelection=createSelectionGate();
const beginCollection=createSelectionGate();
const loadCollectionIndex=createEpisodeLoader({limit:4});
let collections=[],activeCollection=null,mountedCollectionId='',episodes=[],filtered=[],episodeId='',prefetchRevision=0;

function setCollectionOptions(field,options,value){
  const select=$('#demo-'+field);
  select.innerHTML=options.map(item=>`<option value="${escapeHTML(item.id)}">${escapeHTML(item.label)}</option>`).join('');
  select.value=value;select.disabled=options.length<=1;
}
function updateCollectionOptions(collection){
  setCollectionOptions('dataset',collectionOptions(collections,'dataset'),collection.dataset.id);
  setCollectionOptions('model',collectionOptions(collections,'model',{dataset:collection.dataset.id}),collection.model.id);
  const settings=collectionOptions(collections,'setting',{dataset:collection.dataset.id,model:collection.model.id});
  setCollectionOptions('setting',settings,collection.setting.id);
  $('#demo-setting-field').hidden=settings.length<=1;
}
async function selectCollection(collection){
  if(!collection)return;
  const previousSample=mountedCollectionId===activeCollection?.id&&activeCollection?.dataset.id===collection.dataset.id?record?.public_id:'';
  const isCurrent=beginCollection();beginSelection();prefetchRevision++;stop();
  if($('#zoom').open)$('#zoom').close();
  activeCollection=collection;episodes=[];filtered=[];episodeId='';
  updateCollectionOptions(collection);
  for(const id of ['episode-task','episode-tool','episode-select','previous-episode','next-episode'])$('#'+id).disabled=true;
  $('#episode-select').innerHTML='<option>Loading…</option>';
  $('#episode-position').textContent='';
  $('.interaction-shell').hidden=true;$('#demo>.credits').hidden=true;
  $('#episode-status').textContent='Loading trajectories…';$('#episode-status').hidden=false;
  try{
    const index=await loadCollectionIndex(collection.index);
    if(!isCurrent())return;
    if(!Array.isArray(index.episodes))throw Error('Invalid trajectory index');
    episodes=index.episodes;episodeId=chooseEpisode(index,previousSample);
    $('#episode-task').innerHTML='<option value="">All tasks</option>'+[...new Set(episodes.map(item=>item.task))].sort().map(task=>`<option>${escapeHTML(task)}</option>`).join('');
    $('#episode-tool').innerHTML='<option value="">All tools</option>'+[...new Set(episodes.flatMap(item=>item.tools))].sort().map(tool=>`<option>${escapeHTML(tool)}</option>`).join('');
    $('#episode-task').disabled=false;$('#episode-tool').disabled=false;
    updateLibrary();
  }catch{
    if(!isCurrent())return;
    $('#episode-status').innerHTML='Trajectories unavailable. <button type="button" id="retry-collection">Retry</button>';
    $('#retry-collection').onclick=()=>selectCollection(collection);
  }
}
function updateLibrary(){
  filtered=filterEpisodes(episodes,$('#episode-task').value,$('#episode-tool').value);
  const taskNumbers=new Map();
  $('#episode-select').innerHTML=filtered.map(episode=>{
    const number=(taskNumbers.get(episode.task)||0)+1;taskNumbers.set(episode.task,number);
    return `<option value="${episode.id}">${escapeHTML(episode.task)} · ${number} · ${episode.steps} steps</option>`;
  }).join('')||'<option value="">No matching trajectories</option>';
  $('#episode-select').disabled=!filtered.length;
  if(filtered.some(episode=>episode.id===episodeId)){
    $('#episode-select').value=episodeId;updatePosition();
    if(mountedCollectionId!==activeCollection.id||(record?.episode_id||record?.public_id)!==episodeId||$('.interaction-shell').hidden)selectEpisode(episodeId);
    else prefetchNeighbours();
  }else if(filtered.length){selectEpisode(filtered[0].id);}
  else{
    beginSelection();prefetchRevision++;stop();
    $('#episode-position').textContent='0 / 0';
    $('#previous-episode').disabled=true;$('#next-episode').disabled=true;
    $('.interaction-shell').hidden=true;
    $('#demo>.credits').hidden=true;
    $('#episode-status').hidden=true;
  }
}
function updatePosition(){
  const index=filtered.findIndex(episode=>episode.id===episodeId);
  $('#episode-position').textContent=`${index+1} / ${filtered.length}`;
  $('#previous-episode').disabled=index<=0;
  $('#next-episode').disabled=index<0||index>=filtered.length-1;
}
async function selectEpisode(id){
  const episode=episodes.find(item=>item.id===id);if(!episode)return;
  const isCurrent=beginSelection(),collection=activeCollection;prefetchRevision++;
  stop();episodeId=id;$('#episode-select').value=id;updatePosition();
  $('.interaction-shell').hidden=false;
  $('.interaction-shell').inert=true;
  $('.interaction-shell').setAttribute('aria-busy','true');
  $('#demo>.credits').hidden=true;
  $('#episode-status').textContent='Loading trajectory…';$('#episode-status').hidden=false;
  try{
    const data=await loadEpisode(episode.file);
    if(!isCurrent())return;
    if(!data.rounds?.length)throw Error('Empty trajectory');
    mountEpisode(data,collection);mountedCollectionId=collection.id;
    $('.interaction-shell').inert=false;
    $('.interaction-shell').removeAttribute('aria-busy');
    $('#episode-status').hidden=true;
    $('#demo>.credits').hidden=false;
    prefetchNeighbours();
  }catch{
    if(!isCurrent())return;
    $('.interaction-shell').hidden=true;
    $('#episode-status').innerHTML='Trajectory unavailable. <button type="button" id="retry-episode">Retry</button>';
    $('#retry-episode').onclick=()=>selectEpisode(id);
  }
}
function prefetchNeighbours(){
  if(navigator.connection?.saveData||/2g/.test(navigator.connection?.effectiveType||''))return;
  const revision=++prefetchRevision;
  const index=filtered.findIndex(episode=>episode.id===episodeId);
  const neighbours=[filtered[index+1],filtered[index-1]].filter(Boolean);
  const work=async()=>{
    for(const episode of neighbours){
      if(revision!==prefetchRevision||document.hidden)return;
      try{
        const data=await loadEpisode(episode.file);
        if(revision!==prefetchRevision)return;
        // JSON and small source thumbnails only; full-size images load on zoom.
        for(const item of data.images.filter(image=>!image.tool)){
          const image=new Image();image.src='assets/'+(item.thumbnail||item.file);
        }
      }catch{/* Explicit selection exposes a retry without background error noise. */}
    }
  };
  if('requestIdleCallback' in window)requestIdleCallback(work,{timeout:2000});
  else setTimeout(work,500);
}
$('#episode-task').onchange=updateLibrary;
$('#episode-tool').onchange=updateLibrary;
$('#episode-select').onchange=e=>selectEpisode(e.target.value);
$('#previous-episode').onclick=()=>{const i=filtered.findIndex(item=>item.id===episodeId);if(i>0)selectEpisode(filtered[i-1].id);};
$('#next-episode').onclick=()=>{const i=filtered.findIndex(item=>item.id===episodeId);if(i<filtered.length-1)selectEpisode(filtered[i+1].id);};
for(const field of ['dataset','model','setting'])$('#demo-'+field).onchange=()=>{
  selectCollection(chooseCollection(collections,{dataset:$('#demo-dataset').value,model:$('#demo-model').value,setting:$('#demo-setting').value}));
};
async function loadCatalogue(){
  try{
    const response=await fetch('assets/replays/collections.json',{cache:'no-cache'});
    if(!response.ok)throw Error('Library unavailable');
    const catalogue=await response.json();collections=validateCollections(catalogue);
    await selectCollection(collections.find(item=>item.id===catalogue.default_collection));
  }catch{
    $('#episode-status').innerHTML='Trajectory library unavailable. <button type="button" id="retry-library">Retry</button>';
    $('#episode-status').hidden=false;$('.interaction-shell').hidden=true;
    $('#retry-library').onclick=loadCatalogue;
  }
}
loadCatalogue();
