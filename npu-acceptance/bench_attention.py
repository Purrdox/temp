#!/usr/bin/env python3
"""NPU correctness and performance: RMSNorm, full/window attention, four training modes."""
import argparse
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('NANOCHAT_DTYPE','bfloat16')
os.environ['NANOCHAT_DISABLE_COMPILE']='1'
import torch
import torch.nn.functional as F

def error_metrics(actual,expected):
    a=actual.detach().cpu().double()
    b=expected.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite output/gradient')
    d=a-b
    return {'max_abs':d.abs().max().item(),
            'relative_rms':(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)).item()}

def values_and_grads(fn,inputs,upstream):
    leaves=[x.detach().clone().requires_grad_(True) for x in inputs]
    out=fn(*leaves)
    grads=torch.autograd.grad(out,leaves,upstream)
    return out.detach(),[g.detach() for g in grads]

def measure(fn,warmup,iters):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    samples=[]
    for _ in range(iters):
        torch.npu.synchronize()
        start=time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return {'median_ms':statistics.median(samples),'mean_ms':statistics.mean(samples),
            'samples_ms':samples,'peak_allocated_mib':torch.npu.max_memory_allocated()/2**20}

def operator_timing(fn,inputs,upstream,args):
    leaves=[x.detach().clone().requires_grad_(True) for x in inputs]
    def forward():
        with torch.no_grad(): fn(*leaves)
    def backward():
        for x in leaves: x.grad=None
        fn(*leaves).backward(upstream)
    return {'forward':measure(forward,args.warmup,args.iters),
            'forward_backward':measure(backward,args.warmup,args.iters)}

def bench_rmsnorm_shape(args,shape):
    import torch_npu
    from nanochat import gpt
    os.environ['NANOCHAT_RMSNORM_IMPL']='native'
    gpt._NPU_RMS_STATE.update(resolved=False,fn=None)
    x=torch.randn(shape,device='npu:0',dtype=torch.bfloat16)
    dy=torch.randn_like(x)*.1
    gamma=torch.ones(shape[-1],device=x.device,dtype=x.dtype)
    truth_x=x.detach().cpu().double().requires_grad_(True)
    truth=truth_x*torch.rsqrt(truth_x.square().mean(-1,keepdim=True)+1e-6)
    truth_grad,=torch.autograd.grad(truth,truth_x,dy.cpu().double())
    impls={'native':gpt.norm,
           'torch_rms_norm':lambda v:F.rms_norm(v,(shape[-1],),eps=1e-6),
           'npu_rms_norm':lambda v:torch_npu.npu_rms_norm(v,gamma,epsilon=1e-6)[0]}
    rows={}
    for name,fn in impls.items():
        out,grads=values_and_grads(fn,[x],dy)
        errors={'output_vs_cpu_fp64':error_metrics(out,truth),
                'dx_vs_cpu_fp64':error_metrics(grads[0],truth_grad)}
        passed=(errors['output_vs_cpu_fp64']['relative_rms']<.02 and
                errors['dx_vs_cpu_fp64']['relative_rms']<.03)
        rows[name]={'errors':errors,'passed':passed}
        if passed: rows[name].update(operator_timing(fn,[x],dy,args))
        print('RMSNorm',name,json.dumps(rows[name]),flush=True)
    for row in rows.values():
        if 'forward' in row and 'forward' in rows['native']:
            for phase in ('forward','forward_backward'):
                row[phase]['speedup_vs_native']=rows['native'][phase]['median_ms']/row[phase]['median_ms']
    return {'shape':shape,'eps':1e-6,'fp64_reference_device':'cpu','rows':rows,
            'passed':all(r['passed'] for r in rows.values())}

def bench_rmsnorm(args):
    cases={}
    for label,shape in (('hidden',(args.batch,args.seq,args.heads*args.head_dim)),
                        ('qk_per_head',(args.batch,args.seq,args.heads,args.head_dim))):
        print('RMS SHAPE',label,shape,flush=True)
        cases[label]=bench_rmsnorm_shape(args,shape)
    return {'cases':cases,'passed':all(row['passed'] for row in cases.values())}

def scan_attention(args):
    import torch_npu
    from nanochat.flash_attention import _sdpa_attention,_npu_causal_mask
    length=max(256,args.window+8)
    q=torch.zeros(1,length,args.heads,args.head_dim,device='npu:0',dtype=torch.bfloat16)
    k=torch.zeros_like(q)
    v=torch.zeros_like(q)
    v[:,0,:,0]=32
    ref=_sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                        (args.window,0),False,causal=True).transpose(1,2)
    rows=[]
    first_pre_outputs={}
    zero_next_outputs={}
    for mode in (2,3,4):
        for pre in (args.window-1,args.window,args.window+1):
            for following in (0,1):
                out=torch_npu.npu_fusion_attention(q,k,v,head_num=args.heads,input_layout='BSND',
                    atten_mask=_npu_causal_mask(length,q.device),scale=1/math.sqrt(args.head_dim),
                    keep_prob=1.,pre_tockens=pre,next_tockens=following,sparse_mode=mode)[0]
                errors=error_metrics(out,ref)
                same_mode_next=(mode,following)
                same_mode_pre=(mode,pre)
                if same_mode_next not in first_pre_outputs:
                    first_pre_outputs[same_mode_next]=out.detach().cpu()
                if following==0:
                    zero_next_outputs[same_mode_pre]=out.detach().cpu()
                row={'sparse_mode':mode,'pre_tockens':pre,'next_tockens':following,
                     'errors':errors,'boundary_passed':errors['max_abs']<.02,
                     'change_vs_first_pre_same_mode_next':error_metrics(out,first_pre_outputs[same_mode_next]),
                     'change_vs_next_zero_same_mode_pre':error_metrics(out,zero_next_outputs[same_mode_pre])}
                rows.append(row)
                print('WINDOW SCAN',json.dumps(row),flush=True)
    expected=next(r for r in rows if r['sparse_mode']==4 and r['pre_tockens']==args.window and r['next_tockens']==0)
    return {'probe':'q=k=0; v[key=0,channel=0]=32; inclusive left-boundary check',
            'rows':rows,'production_default_passed':expected['boundary_passed']}

def bench_attention(args):
    from nanochat.flash_attention import _sdpa_attention,_npu_attention
    scan=scan_attention(args)
    if not scan['production_default_passed']:
        return {'scan':scan,'passed':False,'reason':'Production band/window parameters failed'}
    shape=(args.batch,args.seq,args.heads,args.head_dim)
    inputs=[torch.randn(shape,device='npu:0',dtype=torch.bfloat16) for _ in range(3)]
    dy=torch.randn_like(inputs[0])*.1
    rows={}
    for label,window in (('full',(-1,-1)),('window',(args.window,0))):
        def sdpa(q,k,v):
            return _sdpa_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                   window,False,causal=True).transpose(1,2)
        def fused(q,k,v): return _npu_attention(q,k,v,window)
        ref,ref_grad=values_and_grads(sdpa,inputs,dy)
        out,grad=values_and_grads(fused,inputs,dy)
        errors={'output':error_metrics(out,ref),
                **{n:error_metrics(a,b) for n,a,b in zip(('dq','dk','dv'),grad,ref_grad)}}
        passed=errors['output']['relative_rms']<.02 and all(errors[n]['relative_rms']<.03 for n in ('dq','dk','dv'))
        row={'errors':errors,'passed':passed}
        if passed:
            row['sdpa']=operator_timing(sdpa,inputs,dy,args)
            row['fused']=operator_timing(fused,inputs,dy,args)
            for phase in ('forward','forward_backward'):
                row['fused'][phase]['speedup_vs_sdpa']=row['sdpa'][phase]['median_ms']/row['fused'][phase]['median_ms']
        rows[label]=row
        print('ATTENTION',label,json.dumps(row),flush=True)
    return {'shape':shape,'window':args.window,'scan':scan,'rows':rows,
            'passed':all(r['passed'] for r in rows.values())}

def bench_train_step(args):
    from nanochat import gpt,flash_attention
    width=math.ceil(64*args.depth/args.head_dim)*args.head_dim
    cfg=gpt.GPTConfig(sequence_len=args.seq,vocab_size=16384,n_layer=args.depth,
                      n_head=width//args.head_dim,n_kv_head=width//args.head_dim,
                      n_embd=width,window_pattern='SSSL')
    accum=args.total_batch//(args.batch*args.seq)
    rows={}
    for label,attn,rms in (('A_native','sdpa','native'),('B_rms','sdpa','npu'),
                           ('C_attention','npu_fa','native'),('D_both','npu_fa','npu')):
        os.environ['NANOCHAT_ATTN_IMPL']=attn
        os.environ['NANOCHAT_RMSNORM_IMPL']=rms
        gpt._NPU_RMS_STATE.update(resolved=False,fn=None)
        gpt._NPU_RMS_WEIGHT_CACHE.clear()
        flash_attention._NPU_MASK_CACHE.clear()
        gc.collect()
        torch.npu.empty_cache()
        torch.manual_seed(42)
        with torch.device('meta'): model=gpt.GPT(cfg)
        model.to_empty(device='npu:0')
        model.init_weights()
        model.train()
        optimizer=model.setup_optimizer(unembedding_lr=.002,embedding_lr=.075,
                                        matrix_lr=.005,weight_decay=.2,scalar_lr=.125)
        x=torch.randint(1,16000,(accum,args.batch,args.seq),device='npu:0')
        y=torch.randint(1,16000,(accum,args.batch,args.seq),device='npu:0')
        losses=[]
        def step():
            model.zero_grad(set_to_none=True)
            total_loss=torch.zeros((),device='npu:0',dtype=torch.float32)
            for i in range(accum):
                loss=model(x[i],y[i])
                (loss/accum).backward()
                total_loss.add_(loss.detach()/accum)
            optimizer.step()
            value=total_loss.item()
            if not math.isfinite(value): raise ValueError('Nonfinite training loss')
            losses.append(value)
        timing=measure(step,args.train_warmup,args.train_iters)
        rows[label]={'attention':attn,'rmsnorm':rms,'timing':timing,
                     'tokens_per_second':args.total_batch/(timing['median_ms']/1000),
                     'parameters':sum(p.numel() for p in model.parameters()),
                     'model_config':vars(cfg),'window_sizes':model.window_sizes,
                     'losses':losses,'passed':True}
        print('TRAIN STEP',label,json.dumps(rows[label]),flush=True)
        del step,optimizer,model,x,y
        gpt._NPU_RMS_WEIGHT_CACHE.clear()
        gc.collect()
        torch.npu.empty_cache()
    a,b,c,d=[rows[k]['timing']['median_ms'] for k in ('A_native','B_rms','C_attention','D_both')]
    for row in rows.values(): row['speedup_vs_native']=a/row['timing']['median_ms']
    return {'data':'fixed synthetic batches; speed comparison, not a model-quality experiment',
            'batch':args.batch,'seq':args.seq,'total_batch':args.total_batch,'accumulation':accum,
            'timing_scope':'forward+backward+Muon/AdamW update+zero_grad, incl. loss sync',
            'attention_policy':'fuse window layers only; full layers remain SDPA','rows':rows,
            'observed_combined_saving_ms':a-d,'sum_individual_savings_ms':(a-b)+(a-c),'passed':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only',choices=('rmsnorm','attn','train'))
    for name,default in (('batch',8),('seq',512),('heads',4),('head-dim',128),('window',128),
                         ('depth',8),('total-batch',32768),('warmup',3),('iters',10),
                         ('train-warmup',2),('train-iters',5)):
        ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--output',type=Path,default=Path('outputs/npu/acceptance/operator-results.json'))
    args=ap.parse_args()
    if min(args.batch,args.seq,args.heads,args.head_dim,args.depth,args.total_batch,args.iters,args.train_iters)<=0:
        ap.error('Sizes and measured iterations must be positive')
    if min(args.warmup,args.train_warmup)<0 or not 1<=args.window<args.seq-1:
        ap.error('Need nonnegative warmups and 1 <= window < seq-1')
    if args.total_batch%(args.batch*args.seq): ap.error('total-batch must be divisible by batch*seq')
    import torch_npu
    assert torch.npu.is_available() and torch.npu.device_count()==1,'Need one real visible NPU'
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    os.environ.update(NANOCHAT_ATTN_IMPL='sdpa',NANOCHAT_RMSNORM_IMPL='native',NANOCHAT_DISABLE_VE='1')
    for key in list(os.environ):
        if key.startswith('NANOCHAT_NPU_FA_'): os.environ.pop(key)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'created_at':datetime.now().isoformat(timespec='seconds'),'torch':torch.__version__,
            'torch_npu':torch_npu.__version__,'device':torch.npu.get_device_name(0),'dtype':'bfloat16',
            'compile':False,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'thresholds':{'output_relative_rms':.02,'gradient_relative_rms':.03,'window_probe_max_abs':.02},
            'methodology':'NPU synchronized wall-clock samples, warmup excluded; CPU fp64 RMSNorm truth'}
    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for name,fn in (('rmsnorm',bench_rmsnorm),('attn',bench_attention),('train',bench_train_step)):
            if args.only and args.only!=name: continue
            print('\nSTART',name,flush=True)
            report[name]=fn(args)
            save()
            if not report[name]['passed']: raise RuntimeError(name+' correctness gate failed; inspect JSON')
        report['passed']=True
        print('NPU OPERATOR ACCEPTANCE PASSED',flush=True)
        return 0
    except Exception:
        report['passed']=False
        report['error']=traceback.format_exc()
        print(report['error'],file=sys.stderr,flush=True)
        return 1
    finally:
        save()
        print('Report:',args.output.resolve(),flush=True)

if __name__=='__main__': sys.exit(main())
