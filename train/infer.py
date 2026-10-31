"""Diacritize text with a Tashkeel-v4 checkpoint. Input marks are removed first; letters and punctuation are preserved.

  python infer.py --text 'وش أخبارك؟'
  python infer.py --model run/best < sentences.txt"""
import argparse,sys
import torch
from common_v4 import V4,load,snapshot,predict,strip

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Rabe3/Tashkeel-v4-Saudi',help='hub id or local directory');p.add_argument('--revision')
    p.add_argument('--text',action='append',help='repeatable; otherwise newline-delimited stdin')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
    texts=a.text or [l.rstrip('\n') for l in sys.stdin]
    model,tok,mod=load(snapshot(a.model,a.revision),a.device);v4=V4(mod)
    keep=[i for i,t in enumerate(texts) if strip(t).strip()];out=list(texts)
    for i,pred in zip(keep,predict(model,tok,v4,[strip(texts[i]) for i in keep],a.device,a.batch_size)):out[i]=pred
    print('\n'.join(out))
if __name__=='__main__':main()
