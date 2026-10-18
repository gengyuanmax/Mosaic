import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {buildRounds,frameAt,moveFrame} from '../docs/replay-state.mjs';
const record=JSON.parse(readFileSync(new URL('../docs/assets/replay.json',import.meta.url),'utf8'));
const rounds=buildRounds(record);
test('new images enter workspace only on environment response, then become next model input',()=>{
  for(const [index,round] of rounds.entries()){
    assert.equal(new Set(round.after).size,round.after.length);
    for(const [phaseIndex,phase] of round.phases.entries()){
      const frame=frameAt(rounds,index,phaseIndex);
      assert.deepEqual(frame.visible,phase==='feedback'?round.after:round.input);
      if(phase!=='feedback')assert.ok(round.added.every(id=>!frame.visible.includes(id)));
    }
    for(const call of round.calls)assert.ok(call.target_images.every(id=>round.input.includes(id)));
    if(index+1<rounds.length)assert.deepEqual(round.after,rounds[index+1].input);
  }
});
test('forward and backward traversal visits all 19 states without skipping feedback or creating images',()=>{
  let r=0,p=0;const seen=[];
  while(true){seen.push([r,p]);const next=moveFrame(rounds,r,p);if(next[0]===r&&next[1]===p)break;[r,p]=next;}
  assert.equal(seen.length,19);assert.equal(rounds[r].phases[p],'answer');
  assert.deepEqual(frameAt(rounds,r,p).visible,record.images.map(i=>i.id));
  for(let i=seen.length-2;i>=0;i--){[r,p]=moveFrame(rounds,r,p,-1);assert.deepEqual([r,p],seen[i]);}
  assert.deepEqual(moveFrame(rounds,0,0,-1),[0,0]);
});
test('reasoning and decision preserve recorded prose verbatim, final answer is its own stage',()=>{
  for(const round of rounds){const prose=round.raw.replace(/<tool_call>[\s\S]*?<\/tool_call>/g,'').replace(/<answer>[\s\S]*?<\/answer>/g,'').trim();assert.equal([round.reasoning,round.decision].filter(Boolean).join('\n\n'),prose);}
  assert.deepEqual(rounds.at(-1).phases,['input','reasoning','answer']);
  assert.equal(rounds.at(-1).answer,'A');
});
