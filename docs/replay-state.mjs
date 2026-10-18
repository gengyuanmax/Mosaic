// Recorded responses are kept verbatim. Stages expose the interaction without generating new model text.
export const phaseLabels = {input:'Model input', reasoning:'Reasoning & decision', action:'Tool call', feedback:'Environment response', answer:'Final answer'};
export function buildRounds(record) {
  let images=record.images.filter(image=>!image.tool).map(image=>image.id);
  return record.rounds.map((round,index)=>{
    const input=[...images], calls=round.tool_calls||[], results=round.tool_results||[];
    const added=results.flatMap(result=>result.new_image_ids||[]);
    images=[...images,...added];
    const prose=(round.model_raw_response||'').replace(/<tool_call>[\s\S]*?<\/tool_call>/g,'').replace(/<answer>[\s\S]*?<\/answer>/g,'').trim();
    const paragraphs=prose.split(/\n\s*\n/);
    return {index,input,after:[...images],added,calls,results,raw:round.model_raw_response,answer:round.final_answer,
      reasoning:paragraphs.length>1?paragraphs.slice(0,-1).join('\n\n'):prose,
      decision:paragraphs.length>1?paragraphs.at(-1):'',
      phases:calls.length?['input','reasoning','action','feedback']:['input','reasoning',round.final_answer?'answer':'status']};
  });
}
export function frameAt(rounds,roundIndex,phaseIndex) {
  const round=rounds[roundIndex],phase=round.phases[phaseIndex];
  if(!phase)throw new Error('Unknown interaction stage');
  return {round,phase,visible:phase==='feedback'?round.after:round.input,added:phase==='feedback'?round.added:[]};
}
export function moveFrame(rounds,roundIndex,phaseIndex,direction=1) {
  const last=rounds.length-1;
  if(direction>0){if(phaseIndex<rounds[roundIndex].phases.length-1)return [roundIndex,phaseIndex+1];if(roundIndex<last)return [roundIndex+1,0];}
  else {if(phaseIndex>0)return [roundIndex,phaseIndex-1];if(roundIndex>0)return [roundIndex-1,rounds[roundIndex-1].phases.length-1];}
  return [roundIndex,phaseIndex];
}
