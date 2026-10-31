"""Tashkeel-v4 fine-tuning helpers: features identical to the publisher's diacritize(),
partial-label CRF loss, fast batched prediction, and a text-level evaluator shared with Shakkelha."""
import collections,json,re,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from text_utils import units,strip,encode_target,LABELS
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
# Base model of the v4 Saudi run. Pass Rabe3/Tashkeel-v4-Saudi (or a local run/best) to continue from the fine-tuned model.
REPO='ahmedsamirtarjama/Tashkeel-v4';REVISION='c89245920ddc9f9fd94dc099a3d47ecd671dff6e'

def snapshot(repo=REPO,revision=REVISION):
    """Local directory for a hub model at a pinned revision; a local directory is returned unchanged."""
    if Path(repo).is_dir():return str(Path(repo).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.py','*.txt','model.safetensors'])

def load(path,device):
    from transformers import AutoModel,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    model=AutoModel.from_pretrained(path,trust_remote_code=True).to(device)
    mod=sys.modules[type(model).__module__]
    return model,tok,mod

class V4:
    """Binds the publisher module (segmentation, class list) to feature/label building."""
    def __init__(self,mod):
        self.mod=mod;self.letters=mod.STANDARD_ARABIC_LETTERS
        self.classes=mod.DIACRITICS_LIST
        self.key_to_id={''.join(sorted(set(d))):i for i,d in enumerate(self.classes)}

    def features(self,tok,texts):
        # Mirrors TashkeelV3ForDiacritization.diacritize feature construction exactly.
        enc=tok(texts,return_offsets_mapping=True,padding=True,truncation=True,max_length=512,return_tensors='pt')
        offsets_batch=enc['offset_mapping'].tolist();num_tokens=enc['input_ids'].shape[1]
        B=len(texts);L=max(len(t) for t in texts)
        c2t=np.zeros((B,L),np.int64);cmask=np.zeros((B,L),bool);cids=np.zeros((B,L),np.int64)
        wpos=np.zeros((B,L),np.int64);morph=np.zeros((B,L),np.int64);irab=np.zeros((B,L),bool)
        for b,text in enumerate(texts):
            offsets=offsets_batch[b];tok_idx=0;ciw=[];mm=[];se=[]
            for w in text.split(' '):
                wl=len(w);types=[0]*wl;ends=[False]*wl
                for start,end,_,t in self.mod.segment_word_morphemes(w):
                    for i in range(start,end):types[i]=t
                    if t==self.mod.MORPHEME_STEM and end>start:ends[end-1]=True
                for i in range(wl):ciw.append(min(i,31));mm.append(types[i]);se.append(ends[i] or i==wl-1)
                ciw.append(0);mm.append(0);se.append(False)
            for c in range(len(text)):
                while tok_idx<len(offsets) and offsets[tok_idx][1]<=c:tok_idx+=1
                c2t[b,c]=min(tok_idx,num_tokens-1);cmask[b,c]=True;cids[b,c]=min(ord(text[c]),255)
                if c<len(ciw):wpos[b,c]=ciw[c];morph[b,c]=mm[c];irab[b,c]=se[c]
        t=torch.from_numpy
        return {'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],'char_to_token':t(c2t),'char_mask':t(cmask),
                'char_ids':t(cids),'word_pos_ids':t(wpos),'morpheme_type_ids':t(morph),'is_irab_pos':t(irab),
                'lengths':torch.tensor([len(x) for x in texts])}

    def labels(self,target,length):
        """Per bare character: class id; -100 = unknown (Arabic letter in a fully unmarked word, or an
        unsupported mark combination); 0 for non-letters. Same marked-word rule as Shakkelha fine-tuning."""
        u=units(target);bare=''.join(c for c,_ in u);marked=set()
        for m in re.finditer(r'[ء-ي]+',bare):
            if any(u[i][1] for i in range(m.start(),m.end())):marked.update(range(m.start(),m.end()))
        y=[-100]*length
        for i,(c,d) in enumerate(u):
            if c in self.letters:y[i]=self.key_to_id.get(''.join(sorted(set(d))),-100) if i in marked else -100
            else:y[i]=0
        return y

    def collate(self,tok,rows):
        texts=[strip(r['target']) for r in rows];f=self.features(tok,texts);L=f['char_mask'].shape[1]
        y=torch.full((len(rows),L),-100,dtype=torch.long)
        for b,r in enumerate(rows):y[b,:len(texts[b])]=torch.tensor(self.labels(r['target'],len(texts[b])))
        f['labels']=y;f['repos']=[r['repo'] for r in rows];f['texts']=texts
        return f

    def decode_text(self,text,path):
        out=[]
        for i,c in enumerate(text):
            out.append(c)
            if c in self.letters and i<len(path) and self.classes[path[i]]:out.append(self.classes[path[i]])
        return self.mod._add_hamza_below_kasra(''.join(out))

def forward(model,f,device,amp=True):
    """Publisher forward, but the BiLSTM is packed so padding never leaks into the backward direction."""
    g=lambda k:f[k].to(device,non_blocking=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp and torch.device(device).type=='cuda'):
        th=model.encoder(input_ids=g('input_ids'),attention_mask=g('attention_mask')).last_hidden_state
    th=th.float();c2t=torch.clamp(g('char_to_token'),0,max(th.size(1)-1,0))
    feats=torch.gather(th,1,c2t.unsqueeze(-1).expand(-1,-1,th.size(-1)))
    x=torch.cat([feats,model.char_embed(torch.clamp(g('char_ids'),0,255)),model.word_pos_embed(torch.clamp(g('word_pos_ids'),0,31)),
                 model.morpheme_embed(torch.clamp(g('morpheme_type_ids'),0,7))],-1)
    model.lstm.flatten_parameters()
    packed=pack_padded_sequence(x,f['lengths'],batch_first=True,enforce_sorted=False)
    out,_=model.lstm(packed);out,_=pad_packed_sequence(out,batch_first=True,total_length=x.size(1))
    h=model.layer_norm(out+model.proj_residual(x))
    stem,irab=model.stem_head(h),model.irab_head(h)
    return torch.where(g('is_irab_pos').unsqueeze(-1),0.3*stem+0.7*irab,stem)

def crf_log_partition(crf,em,mask):
    alpha=crf.start_transitions+em[:,0]
    for t in range(1,em.size(1)):
        nxt=torch.logsumexp(alpha.unsqueeze(2)+crf.transitions.unsqueeze(0)+em[:,t].unsqueeze(1),dim=1)
        alpha=torch.where(mask[:,t].unsqueeze(1),nxt,alpha)
    return torch.logsumexp(alpha+crf.end_transitions,dim=1)

def partial_crf_nll(crf,em,labels,mask):
    """-log P(any path consistent with the known labels); unknown positions are marginalised."""
    known=labels>=0
    allowed=torch.where(known.unsqueeze(-1),F.one_hot(labels.clamp(min=0),em.size(-1)).bool(),torch.ones_like(em,dtype=torch.bool))
    both=crf_log_partition(crf,torch.cat([em,em.masked_fill(~allowed,-1e4)]),torch.cat([mask,mask]))
    B=em.size(0);return (both[:B]-both[B:]).sum()

@torch.inference_mode()
def viterbi(crf,em,mask):
    """Same recursion as LinearChainCRF.decode, with backtracking done on CPU arrays."""
    em=em.float();vit=crf.start_transitions+em[:,0];bps=[]
    for t in range(1,em.size(1)):
        best,idx=(vit.unsqueeze(2)+crf.transitions.unsqueeze(0)).max(1)
        vit=torch.where(mask[:,t].unsqueeze(1),best+em[:,t],vit);bps.append(idx)
    last=(vit+crf.end_transitions).argmax(1).cpu().numpy()
    bps=torch.stack(bps,1).cpu().numpy() if bps else np.zeros((em.size(0),0,em.size(2)),np.int64)
    paths=[]
    for b,L in enumerate(mask.sum(1).tolist()):
        tag=int(last[b]);path=[tag]
        for s in range(L-2,-1,-1):tag=int(bps[b,s,tag]);path.append(tag)
        paths.append(path[::-1])
    return paths

@torch.inference_mode()
def predict(model,tok,v4,texts,device,batch_size=64,amp=False):
    """Bare texts in, diacritized texts out. Length-sorted batches; order preserved."""
    model.eval();order=sorted(range(len(texts)),key=lambda i:len(texts[i]));out=[None]*len(texts)
    for s in range(0,len(order),batch_size):
        idx=order[s:s+batch_size];batch=[texts[i] for i in idx];f=v4.features(tok,batch)
        paths=viterbi(model.crf,forward(model,f,device,amp),f['char_mask'].to(device))
        for i,t,p in zip(idx,batch,paths):out[i]=v4.decode_text(t,p)
    return out

def evaluate_texts(refs,preds,repos):
    """Error rates on supervised letters, scored on output text so any model can be compared.
    labelled: all supervised letters; internal: non-final letters; marked_internal: non-final letters with an explicit mark (primary)."""
    totals=collections.defaultdict(collections.Counter)
    for ref,pred,repo in zip(refs,preds,repos):
        y,internal,annotated=encode_target(ref)
        pu=units(pred);assert ''.join(c for c,_ in pu)==strip(ref),(ref,pred)
        p=[LABELS.get(''.join(sorted(set(d))),-1) for _,d in pu]
        for i,label in enumerate(y):
            if label==-100:continue
            wrong=p[i]!=label
            for name,ok in (('labelled',True),('internal',internal[i]),('marked_internal',internal[i] and annotated[i])):
                if not ok:continue
                for g in ('all',repo):totals[g][name+'_n']+=1;totals[g][name+'_errors']+=wrong
    return {g:{**dict(c),**{n+'_der':c[n+'_errors']/max(c[n+'_n'],1) for n in ['labelled','internal','marked_internal']}} for g,c in totals.items()}

def read(split,data_dir=DATA):
    return [json.loads(l) for l in (Path(data_dir)/(split+'.jsonl')).open()]
