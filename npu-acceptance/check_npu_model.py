#!/usr/bin/env python3
"""Reload a completed model, verify chunked KV logits, and time fixed-token decoding."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import statistics
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.update(NANOCHAT_DTYPE='bfloat16',NANOCHAT_DISABLE_VE='1',
                  NANOCHAT_ATTN_IMPL='sdpa',NANOCHAT_RMSNORM_IMPL='native',NANOCHAT_DISABLE_COMPILE='1')


def choose_checkpoint(base,tag=None):
    candidates=[]
    folders=[base/'base_checkpoints'/tag] if tag else sorted((base/'base_checkpoints').glob('d8-npu-course-*'))
    for folder in folders:
        for path in folder.glob('meta_*.json'):
            meta=json.loads(path.read_text(encoding='utf-8'))
            step=meta['step']
            weights=folder/f'model_{step:06d}.pt'
            if step==meta['user_config']['num_iterations'] and weights.is_file():
                candidates.append((folder.name,step,weights,meta))
    if not candidates:
        raise FileNotFoundError('No completed formal d8 checkpoint found')
    return sorted(candidates,key=lambda item:(item[0],item[1]))[-1]


def main():
    import torch
    import torch_npu
    from nanochat.gpt import GPT,GPTConfig
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.engine import KVCache,Engine
    from nanochat.tokenizer import RustBPETokenizer
    from scripts.bench_attention import error_metrics,measure
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-dir',type=Path,default=Path('outputs/npu'))
    ap.add_argument('--model-tag')
    ap.add_argument('--trust-local-checkpoints',action='store_true')
    ap.add_argument('--decode-tokens',type=int,default=32)
    ap.add_argument('--iters',type=int,default=3)
    ap.add_argument('--output',type=Path,default=Path('outputs/npu/acceptance/model-results.json'))
    args=ap.parse_args()
    if args.model_tag and (not re.fullmatch(r'[A-Za-z0-9_.-]+',args.model_tag) or args.model_tag in ('.','..')):
        ap.error('Model tag must be a simple directory name')
    if min(args.decode_tokens,args.iters)<=0:
        ap.error('Token and iteration counts must be positive')
    assert torch.npu.is_available() and torch.npu.device_count()==1,'Need one real visible NPU'
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'created_at':datetime.now().isoformat(timespec='seconds'),'torch':torch.__version__,
            'torch_npu':torch_npu.__version__,'device':torch.npu.get_device_name(0),
            'dtype':'bfloat16','attention':'sdpa','rmsnorm':'native','compile':False}
    try:
        tag,step,weights,meta=choose_checkpoint(args.base_dir,args.model_tag)
        print('Loading completed checkpoint:',weights,flush=True)
        state=torch.load(weights,map_location='cpu',weights_only=not args.trust_local_checkpoints)
        state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
        cfg=GPTConfig(**meta['model_config'])
        with torch.device('meta'):
            model=GPT(cfg)
        model.to_empty(device='npu:0')
        model.init_weights()
        model.load_state_dict(state,strict=True)
        del state
        model.eval()
        tokenizer=RustBPETokenizer.from_directory(str(args.base_dir/'tokenizer'))
        assert tokenizer.get_vocab_size()==cfg.vocab_size
        report.update(model_tag=tag,step=step,parameters=sum(p.numel() for p in model.parameters()),
                      model_config=meta['model_config'],training_validation_bpb=meta.get('val_bpb'))

        def new_cache(length):
            return KVCache(1,cfg.n_kv_head,length,cfg.n_embd//cfg.n_head,cfg.n_layer,
                           torch.device('npu:0'),COMPUTE_DTYPE)

        with torch.inference_mode():
            ids=torch.randint(1,cfg.vocab_size-20,(1,160),device='npu:0')
            full=model(ids)
            cache=new_cache(160)
            chunked=torch.cat([model(ids[:,:70],kv_cache=cache),model(ids[:,70:],kv_cache=cache)],dim=1)
            metrics=error_metrics(chunked,full)
            logits_passed=metrics['relative_rms']<.02 and metrics['max_abs']<.2
            report['chunked_cache']={'chunks':[70,90],'tokens':160,'errors':metrics,
                                      'argmax_agreement':(chunked.argmax(-1)==full.argmax(-1)).float().mean().item(),
                                      'thresholds':{'relative_rms':.02,'max_abs':.2},'passed':logits_passed}
            print('CHUNK CACHE',json.dumps(report['chunked_cache']),flush=True)
            if not logits_passed:
                raise RuntimeError('Cached/full logit consistency check failed')
            del full,chunked,cache
            prefix=256
            ids=torch.randint(1,cfg.vocab_size-20,(1,prefix+args.decode_tokens),device='npu:0')

            def naive():
                return torch.stack([model(ids[:,:prefix+i])[:,-1,:] for i in range(args.decode_tokens)],dim=1)

            def cached():
                c=new_cache(prefix+args.decode_tokens)
                values=[model(ids[:,:prefix],kv_cache=c)[:,-1,:]]
                for i in range(1,args.decode_tokens):
                    values.append(model(ids[:,prefix+i-1:prefix+i],kv_cache=c)[:,-1,:])
                return torch.stack(values,dim=1)

            logits=error_metrics(cached(),naive())
            if logits['relative_rms']>=.02 or logits['max_abs']>=.2:
                raise RuntimeError('Fixed-token cached/naive logit check failed: '+str(logits))
            timings={name:measure(fn,1,args.iters) for name,fn in (('naive',naive),('cached',cached))}
            for row in timings.values():
                row['generated_positions_per_second']=args.decode_tokens/(row['median_ms']/1000)
            c=new_cache(prefix+args.decode_tokens)
            kv_bytes=c.k_cache.numel()*c.k_cache.element_size()+c.v_cache.numel()*c.v_cache.element_size()
            def prefill_only():
                c=new_cache(prefix+args.decode_tokens)
                model(ids[:,:prefix],kv_cache=c)
            prefill_timing=measure(prefill_only,1,args.iters)
            decode_samples=[]
            if args.decode_tokens>1:
                for repeat in range(args.iters+1):
                    c=new_cache(prefix+args.decode_tokens)
                    model(ids[:,:prefix],kv_cache=c)
                    torch.npu.synchronize()
                    start=time.perf_counter()
                    for i in range(1,args.decode_tokens):
                        model(ids[:,prefix+i-1:prefix+i],kv_cache=c)
                    torch.npu.synchronize()
                    if repeat:
                        decode_samples.append((time.perf_counter()-start)*1000)
            report['decode']={'method':'teacher-forced fixed continuation; includes prefill and cache allocation',
                              'prompt_tokens':prefix,'decode_tokens':args.decode_tokens,'logit_errors':logits,
                              'timings':timings,'speedup':timings['naive']['median_ms']/timings['cached']['median_ms'],
                              'kv_cache_allocated_bytes':kv_bytes,'kv_cache_capacity_tokens':c.max_seq_len,
                              'kv_bytes_per_token':kv_bytes/c.max_seq_len,
                              'prefill_with_cache_allocation':prefill_timing,
                              'decode_only_samples_ms':decode_samples,
                              'decode_only_forward_steps':max(0,args.decode_tokens-1),
                              'decode_only_tokens_per_second':((args.decode_tokens-1)/(statistics.median(decode_samples)/1000)
                                                               if decode_samples else None)}
            del c
            report['samples']=[]
            engine=Engine(model,tokenizer)
            for prompt in ('那少年提剑立在崖边，','清晨，客栈门外传来一阵马蹄声。'):
                tokens=tokenizer.encode(prompt,prepend=tokenizer.get_bos_token_id())
                sequences,_=engine.generate_batch(tokens,num_samples=1,max_tokens=64,temperature=0.0)
                sample={'prompt':prompt,'continuation':tokenizer.decode(sequences[0][len(tokens):]),
                        'generated_tokens':len(sequences[0])-len(tokens)}
                report['samples'].append(sample)
                print('SAMPLE',json.dumps(sample,ensure_ascii=False),flush=True)
            report['sample_note']='Greedy base-model continuations for human inspection; this is not SFT/chat-quality acceptance.'
        report['passed']=True
        print('NPU MODEL ACCEPTANCE PASSED',flush=True)
        return 0
    except Exception:
        report['passed']=False
        report['error']=traceback.format_exc()
        print(report['error'],file=sys.stderr,flush=True)
        return 1
    finally:
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print('Report:',args.output.resolve(),flush=True)


if __name__=='__main__':
    sys.exit(main())
