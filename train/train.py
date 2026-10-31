"""Fine-tune Tashkeel-v4 on the Egyptian splits (build/build_splits.py); select on validation
only, test once at the end (test and test_tts). Vendored from tashkeel-layer (the
Tashkeel-v4-Saudi run) with three changes: --mem-frac caps the GPU memory, to share the
card with another run; every evaluated checkpoint is kept in kept/step_<n>, not only the
best; test_tts.jsonl, if present, is scored beside test."""
import argparse,functools,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common_v4 import ROOT,DATA,REPO,REVISION,V4,load,snapshot,forward,partial_crf_nll,predict,evaluate_texts,read,strip

def batches(rows,batch,seed):
    """Shuffle, then sort within pools of 50 batches by length to limit padding."""
    rng=random.Random(seed);idx=list(range(len(rows)));rng.shuffle(idx);out=[]
    for s in range(0,len(idx),batch*50):
        pool=sorted(idx[s:s+batch*50],key=lambda i:len(rows[i]['target']))
        out+=[pool[k:k+batch] for k in range(0,len(pool),batch)]
    rng.shuffle(out);return out

def score(model,tok,v4,rows,device):
    preds=predict(model,tok,v4,[strip(r['target']) for r in rows],device)
    return evaluate_texts([r['target'] for r in rows],preds,[r['repo'] for r in rows]),preds

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-model',default=REPO,help='hub id or local directory');p.add_argument('--base-revision',default=REVISION,help='ignored for a local directory')
    p.add_argument('--data-dir',default=str(DATA),help='holds train/validation/test.jsonl with target and repo fields')
    p.add_argument('--epochs',type=float,default=3);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr-encoder',type=float,default=2e-5);p.add_argument('--lr-head',type=float,default=1e-4)
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--evals-per-epoch',type=int,default=2)
    p.add_argument('--ce-weight',type=float,default=0.5);p.add_argument('--max-steps',type=int,help='smoke test: stop after this many steps, no evaluation')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--num-workers',type=int,default=6)
    p.add_argument('--out',default=str(ROOT/'run'));p.add_argument('--mem-frac',type=float,help='cap this process at this share of GPU memory');p.add_argument('--low-memory',action='store_true',help='freeze word embeddings + checkpoint encoder (shared GPU)');a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    device='cuda'
    if a.mem_frac:torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    base=snapshot(a.base_model,a.base_revision);model,tok,mod=load(base,device);v4=V4(mod);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.low_memory:
        # Shared GPU: freeze the 77M-row MARBERT word-embedding table and checkpoint encoder activations.
        model.encoder.embeddings.word_embeddings.weight.requires_grad_(False)
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    train,valid,test=(read(s,a.data_dir) for s in ('train','validation','test'))
    steps_per_epoch=math.ceil(len(train)/a.batch_size);total=a.max_steps or int(steps_per_epoch*a.epochs)
    eval_every=a.max_steps or max(1,steps_per_epoch//a.evals_per_epoch)
    cfg={**vars(a),'base':base,'revision':None if Path(a.base_model).is_dir() else a.base_revision,'parameters':sum(x.numel() for x in model.parameters()),'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),'frozen':'encoder.embeddings.word_embeddings' if a.low_memory else None,
         'train_chunks':len(train),'validation_chunks':len(valid),'test_chunks':len(test),'total_steps':total,'eval_every':eval_every,
         'loss':'partial-label CRF NLL (unknown letters marginalised) + ce_weight * token CE on fused emissions, per known label',
         'precision':'encoder bf16 autocast in training; fp32 for all evaluation','selection_metric':'validation marked_internal_der',
         'labels':'Same marked-word rule as Shakkelha fine-tuning; non-letters labelled no-mark; unsupported combos unknown'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(cfg),flush=True)
    enc=[p_ for n,p_ in model.named_parameters() if n.startswith('encoder.') and p_.requires_grad];rest=[p_ for n,p_ in model.named_parameters() if not n.startswith('encoder.')]
    opt=torch.optim.AdamW([{'params':enc,'lr':a.lr_encoder},{'params':rest,'lr':a.lr_head}],weight_decay=0.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/a.warmup if s<a.warmup else max(0.0,(total-s)/max(1,total-a.warmup)))
    collate=functools.partial(v4.collate,tok)
    history=[];best=float('inf');step=0;start=time.time();epoch=0;done=False
    if not a.max_steps:
        base_val,_=score(model,tok,v4,valid,device);history.append({'step':0,'validation':base_val});print('base validation '+json.dumps(base_val['all']),flush=True)
    while not done:
        epoch+=1
        loader=DataLoader(train,batch_sampler=batches(train,a.batch_size,a.seed+epoch),collate_fn=collate,num_workers=a.num_workers,pin_memory=True,persistent_workers=False)
        model.train();agg={'crf':0.0,'ce':0.0,'n':0}
        for f in loader:
            y=f['labels'].to(device);mask=f['char_mask'].to(device);n=int((y>=0).sum())
            em=forward(model,f,device).float()
            crf=partial_crf_nll(model.crf,em,y,mask)/max(n,1)
            ce=F.cross_entropy(em.transpose(1,2),y,ignore_index=-100)
            loss=crf+a.ce_weight*ce;opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();step+=1
            agg['crf']+=crf.item()*n;agg['ce']+=ce.item()*n;agg['n']+=n
            if step%100==0:
                el=time.time()-start
                print(json.dumps({'epoch':epoch,'step':step,'total':total,'crf':agg['crf']/agg['n'],'ce':agg['ce']/agg['n'],'lr_enc':sched.get_last_lr()[0],
                                  'elapsed_min':round(el/60,1),'eta_min':round(el/step*(total-step)/60,1),'peak_gb':round(torch.cuda.max_memory_allocated()/1e9,2)}),flush=True)
                agg={'crf':0.0,'ce':0.0,'n':0}
            if step%eval_every==0 or step==total:
                if a.max_steps:print('smoke ok',json.dumps({'steps':step,'loss':round(loss.item(),4),'seconds':round(time.time()-start),'peak_gb':round(torch.cuda.max_memory_allocated()/1e9,2)}),flush=True);return
                val,_=score(model,tok,v4,valid,device);model.train()
                item={'step':step,'epoch':round(step/steps_per_epoch,2),'validation':val,'elapsed_s':time.time()-start};history.append(item)
                (out/'history.json').write_text(json.dumps(history,indent=2))
                print('eval '+json.dumps({'step':step,**{k:val['all'][k] for k in ['marked_internal_der','internal_der','labelled_der']}}),flush=True)
                model.save_pretrained(out/'kept'/f'step_{step:06d}');tok.save_pretrained(out/'kept'/f'step_{step:06d}')
                for py in Path(base).glob('*.py'):shutil.copyfile(py,out/'kept'/f'step_{step:06d}'/py.name)
                if val['all']['marked_internal_der']<best:
                    best=val['all']['marked_internal_der'];model.save_pretrained(out/'best');tok.save_pretrained(out/'best')
                    for py in Path(base).glob('*.py'):shutil.copyfile(py,out/'best'/py.name)
                    (out/'best_step.json').write_text(json.dumps({'step':step,'epoch':item['epoch'],'validation_marked_internal_der':best}))
            if step>=total:done=True;break
    from transformers import AutoModel
    model=AutoModel.from_pretrained(out/'best',trust_remote_code=True).to(device).eval()
    base_model,_,_=load(base,device)
    before,_=score(base_model,tok,v4,test,device)
    after,preds=score(model,tok,v4,test,device)
    results={'base':before,'finetuned':after}
    if (Path(a.data_dir)/'test_tts.jsonl').exists():
        tts=read('test_tts',a.data_dir);results['base_tts'],_=score(base_model,tok,v4,tts,device);results['finetuned_tts'],_=score(model,tok,v4,tts,device)
    del base_model
    (out/'test_results.json').write_text(json.dumps(results,indent=2))
    print('FINAL '+json.dumps({'before':before['all'],'after':after['all'],'elapsed_min':round((time.time()-start)/60,1)}),flush=True)
if __name__=='__main__':main()
