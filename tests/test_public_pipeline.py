"""CPU checks for multi-image tool execution and runner protocol integration."""
import json
from pathlib import Path
import httpx
import yaml
from PIL import Image
from openai import OpenAI
from mosaic_gym.agent import agent as agent_module
from mosaic_gym.runner.inference import main
from mosaic_gym.toolbox import Toolbox
from mosaic_gym.contracts import ToolCall


def test_crop_preserves_original_and_registers_output():
    box = Toolbox()
    box.reset([Image.new('RGB', (40, 20), 'red'), Image.new('RGB', (30, 30), 'blue')])
    before = box.images_by_id()['img_0001'].tobytes()
    result = box.run([ToolCall('crop', {'x1':0,'y1':0,'x2':0.5,'y2':1}, ['img_0001'])])[0]
    assert result.error is None
    assert result.produced_images[0].size == (20, 20)
    assert len(box.image_ids()) == 3
    assert box.images_by_id()['img_0001'].tobytes() == before


def test_unknown_image_rejected():
    box = Toolbox()
    box.reset([Image.new('RGB', (20, 20))])
    result = box.run([ToolCall('crop', {'x1':0,'y1':0,'x2':1,'y2':1}, ['img_9999'])])[0]
    assert result.error and not result.produced_images
    assert len(box.image_ids()) == 1


def test_runner_multimodal_tool_roundtrip(tmp_path, monkeypatch):
    data = tmp_path/'datasets'
    data.mkdir()
    for name in ['a','b']:
        Image.new('RGB', (40,20), 'red').save(tmp_path/f'{name}.png')
    sample={'sample_id':'test-example','prompt':'Compare the two images. (A) same (B) different',
            'images':[{'image_id':name,'uri_or_path':f'{name}.png'} for name in ['a','b']], 'target':'A'}
    (data/'example.json').write_text(json.dumps([sample]))
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/default.yaml').read_text())
    cfg['data'].update(dataset_dir=str(data),base_dir=str(tmp_path))
    cfg['runner'].update(out_dir=str(tmp_path/'results'),resume=False,max_steps=3)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    requests=[]
    def respond(request):
        body=json.loads(request.content);requests.append(body)
        reply=('<tool_call>{"name":"crop","arguments":{"target_image":"img1","x1":0,"y1":0,"x2":0.5,"y2":1}}</tool_call>'
               if len(requests)==1 else '<answer>A</answer>')
        return httpx.Response(200,json={'id':'test','object':'chat.completion','created':0,'model':'test',
          'choices':[{'index':0,'message':{'role':'assistant','content':reply},'finish_reason':'stop'}],
          'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
    monkeypatch.setattr(agent_module,'OpenAI',lambda **kwargs: OpenAI(**kwargs,http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    assert main(['--config',str(config)]) == 0
    record=json.loads((tmp_path/'results/record/example.jsonl').read_text())
    assert 'run_error' not in record
    assert record['reward']==1 and record['final_answer']=='A'
    assert len(requests)==2
    first=requests[0]['messages'][-1]['content']
    second=requests[1]['messages'][-1]['content']
    assert sum(x['type']=='image_url' for x in first)==2
    assert sum(x['type']=='image_url' for x in second)==1
    assert not any(r.get('parse_error') for r in record['rounds'])
    assert not any(t.get('error') for r in record['rounds'] for t in r.get('tool_results',[]))
