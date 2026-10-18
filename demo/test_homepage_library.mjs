import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {createEpisodeLoader,filterEpisodes,createSelectionGate,validateCollections,collectionOptions,chooseCollection,chooseEpisode} from '../docs/episode-library.mjs';
import {buildRounds,frameAt} from '../docs/replay-state.mjs';
const root=new URL('../docs/',import.meta.url);

test('navigation shares in-flight prefetch, retries failures, and evicts old entries',async()=>{
  const calls=[];let fail=true;
  const load=createEpisodeLoader({limit:2,fetcher:async url=>{
    calls.push(url);return {ok:url!=='bad'||!fail,json:async()=>({url})};
  }});
  await Promise.all([load('one'),load('one')]);assert.deepEqual(calls,['one']);
  await assert.rejects(load('bad'));fail=false;await load('bad');
  assert.equal(calls.filter(url=>url==='bad').length,2);
  await load('three');await load('one');assert.equal(calls.filter(url=>url==='one').length,2);
});

test('rapid switches cannot mount an earlier response; filters combine task and tool',()=>{
  const begin=createSelectionGate(),old=begin(),latest=begin();
  assert.equal(old(),false);assert.equal(latest(),true);
  const episodes=[{task:'A',tools:['rotate']},{task:'A',tools:['crop']},{task:'B',tools:['rotate']}];
  assert.deepEqual(filterEpisodes(episodes,'A','rotate'),[episodes[0]]);
  assert.deepEqual(filterEpisodes(episodes,'B','crop'),[]);
});

test('dataset/model/setting selectors only expose available combinations',()=>{
  const collection=(dataset,model,setting)=>({id:[dataset,model,setting].join('-'),dataset:{id:dataset,label:dataset},model:{id:model,label:model},setting:{id:setting,label:setting},environment:{label:'MosaicGym'},index:'index.json',attribution:'credits.md',license:'license.md'});
  const collections=[collection('A','M1','visual'),collection('A','M2','visual'),collection('A','M2','text'),collection('B','M3','text')];
  assert.deepEqual(validateCollections({schema_version:1,default_collection:collections[0].id,collections}),collections);
  assert.deepEqual(collectionOptions(collections,'model',{dataset:'A'}).map(x=>x.id),['M1','M2']);
  assert.deepEqual(collectionOptions(collections,'setting',{dataset:'A',model:'M1'}).map(x=>x.id),['visual']);
  assert.equal(chooseCollection(collections,{dataset:'A',model:'M2',setting:'text'}),collections[2]);
  assert.equal(chooseCollection(collections,{dataset:'B',model:'M2',setting:'visual'}),collections[3]);
  assert.equal(chooseCollection(collections,{dataset:'missing'}),undefined);
  assert.throws(()=>validateCollections({schema_version:1,default_collection:'missing',collections}));
  assert.throws(()=>validateCollections({schema_version:1,default_collection:collections[0].id,collections:[...collections,{...collections[0],id:'duplicate-config'}]}));
});

test('model switch keeps the same question when available and falls back honestly',()=>{
  const index={default_id:'run-b',episodes:[{id:'run-a',sample_id:'question-a'},{id:'run-b',sample_id:'question-b'}]};
  assert.equal(chooseEpisode(index,'question-a'),'run-a');
  assert.equal(chooseEpisode(index,'absent'),'run-b');
  assert.equal(chooseEpisode({episodes:[]}), '');
});

test('collections are loaded separately and old dataset requests cannot win',async()=>{
  const calls=[],pending=new Map(),load=createEpisodeLoader({fetcher:url=>{
    calls.push(url);return new Promise(resolve=>pending.set(url,resolve));
  }});
  const begin=createSelectionGate();let mounted='';
  const select=async url=>{const current=begin(),data=await load(url);if(current())mounted=data.id;};
  const first=select('dataset-a.json'),second=select('dataset-b.json');
  pending.get('dataset-b.json')({ok:true,json:async()=>({id:'B'})});await second;
  pending.get('dataset-a.json')({ok:true,json:async()=>({id:'A'})});await first;
  assert.equal(mounted,'B');assert.deepEqual(calls,['dataset-a.json','dataset-b.json']);
  await select('dataset-a.json');assert.equal(mounted,'A');assert.equal(calls.length,2);
});

test('public collection catalogue points to real assets with explicit identities',()=>{
  const catalogue=JSON.parse(readFileSync(new URL('assets/replays/collections.json',root),'utf8'));
  for(const collection of validateCollections(catalogue)){
    for(const key of ['index','attribution','license'])assert.ok(existsSync(new URL(collection[key],root)));
    const index=JSON.parse(readFileSync(new URL(collection.index,root),'utf8'));
    assert.ok(index.episodes.length);assert.ok(index.episodes.some(item=>item.id===index.default_id));
    assert.ok(!/\/home\/|\/Users\//.test(JSON.stringify(collection)));
  }
});

test('every bundled trajectory has complete assets and preserves workspace timing',()=>{
  const index=JSON.parse(readFileSync(new URL('assets/replays/index.json',root),'utf8'));
  assert.ok(index.episodes.length>250);
  assert.equal(new Set(index.episodes.map(e=>e.id)).size,index.episodes.length);
  let errors=0,multi=0,transparent=0,incomplete=0;
  for(const entry of index.episodes){
    const record=JSON.parse(readFileSync(new URL(entry.file,root),'utf8'));
    assert.equal(record.episode_id||record.public_id,entry.id);
    assert.ok(!record.credits.some(c=>c.source_dataset==='COCO'));
    const ids=new Set(record.images.map(i=>i.id));
    assert.equal(ids.size,record.images.length);
    for(const image of record.images){
      for(const key of ['file','thumbnail','full_file'])assert.ok(existsSync(new URL('assets/'+image[key],root)),image[key]);
      transparent+=Number(image.has_transparency);
    }
    const rounds=buildRounds(record);
    for(const round of rounds){
      assert.equal(round.calls.length,round.results.length);
      multi+=Number(round.calls.length>1);
      errors+=round.results.filter(r=>r.error).length;
      assert.ok(round.after.every(id=>ids.has(id)));
      for(const [phaseIndex,phase] of round.phases.entries()){
        assert.deepEqual(frameAt(rounds,round.index,phaseIndex).visible,phase==='feedback'?round.after:round.input);
      }
      assert.equal([round.reasoning,round.decision].filter(Boolean).join('\n\n'),
        (round.raw||'').replace(/<tool_call>[\s\S]*?<\/tool_call>/g,'').replace(/<answer>[\s\S]*?<\/answer>/g,'').trim());
    }
    if(!record.final_answer)incomplete++;
  }
  assert.ok(errors>0);assert.ok(multi>0);assert.ok(transparent>0);assert.ok(incomplete>0);
});
