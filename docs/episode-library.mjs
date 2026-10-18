// Small JSON cache shared by navigation and low-priority prefetches.
export function createEpisodeLoader({fetcher=fetch,limit=12}={}) {
  const cache=new Map();
  return function load(url) {
    if(cache.has(url)){
      const promise=cache.get(url);cache.delete(url);cache.set(url,promise);return promise;
    }
    const promise=fetcher(url).then(response=>{
      if(!response.ok)throw new Error('Trajectory unavailable');
      return response.json();
    }).catch(error=>{if(cache.get(url)===promise)cache.delete(url);throw error;});
    cache.set(url,promise);
    while(cache.size>limit)cache.delete(cache.keys().next().value);
    return promise;
  };
}

export function filterEpisodes(episodes,task='',tool='') {
  return episodes.filter(episode=>(!task||episode.task===task)&&(!tool||episode.tools.includes(tool)));
}

// Each collection belongs to one dataset, model and recorded evaluation setting.
// Keep indices separate so opening a dataset never downloads every trajectory.
export function validateCollections(catalogue) {
  if(catalogue.schema_version!==1||!Array.isArray(catalogue.collections)||!catalogue.collections.length)throw Error('Invalid collection catalogue');
  const ids=new Set(),combinations=new Set();
  for(const item of catalogue.collections){
    if(!item.id||ids.has(item.id)||!item.index||!item.attribution||!item.license||!item.environment?.label)throw Error('Invalid collection');
    for(const field of ['dataset','model','setting'])if(!item[field]?.id||!item[field]?.label)throw Error('Missing collection identity');
    const combination=JSON.stringify([item.dataset.id,item.model.id,item.setting.id]);
    if(combinations.has(combination))throw Error('Duplicate collection identity');
    ids.add(item.id);combinations.add(combination);
  }
  if(!ids.has(catalogue.default_collection))throw Error('Missing default collection');
  return catalogue.collections;
}

export function collectionOptions(collections,field,filters={}) {
  return [...new Map(collections.filter(item=>Object.entries(filters).every(([key,value])=>!value||item[key].id===value))
    .map(item=>[item[field].id,item[field]])).values()];
}

export function chooseCollection(collections,{dataset,model,setting}={}) {
  let candidates=collections.filter(item=>!dataset||item.dataset.id===dataset);
  if(candidates.some(item=>item.model.id===model))candidates=candidates.filter(item=>item.model.id===model);
  return candidates.find(item=>item.setting.id===setting)||candidates[0];
}

export function chooseEpisode(index,previousSample='') {
  return index.episodes.find(item=>previousSample&&(item.sample_id||item.id)===previousSample)?.id
    ||index.episodes.find(item=>item.id===index.default_id)?.id||index.episodes[0]?.id||'';
}

// Only the latest requested episode may replace the visible conversation.
export function createSelectionGate() {
  let revision=0;
  return () => {const own=++revision;return ()=>own===revision;};
}
