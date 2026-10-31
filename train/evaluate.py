"""Score a checkpoint (hub id or local directory) on a split with the same evaluator used for selection and the reported results."""
import argparse,json
import torch
from common_v4 import DATA,V4,load,snapshot,predict,evaluate_texts,read,strip

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Rabe3/Tashkeel-v4-Saudi');p.add_argument('--revision',help='hub revision; default latest')
    p.add_argument('--split',default='test');p.add_argument('--data-dir',default=str(DATA));p.add_argument('--limit',type=int)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--out',help='write full per-source metrics here');a=p.parse_args()
    model,tok,mod=load(snapshot(a.model,a.revision),a.device);v4=V4(mod)
    rows=read(a.split,a.data_dir)[:a.limit]
    preds=predict(model,tok,v4,[strip(r['target']) for r in rows],a.device,a.batch_size)
    res=evaluate_texts([r['target'] for r in rows],preds,[r['repo'] for r in rows])
    for g,m in sorted(res.items(),key=lambda x:x[0]!='all'):
        print(f"{g:42s} marked_internal {m['marked_internal_der']:.2%}  internal {m['internal_der']:.2%}  labelled {m['labelled_der']:.2%}")
    if a.out:open(a.out,'w').write(json.dumps({'model':a.model,'split':a.split,'rows':len(rows),'results':res},indent=2))
if __name__=='__main__':main()
