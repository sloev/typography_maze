# prototype: one stroke per sentence, length = words in sentence, turn = sign of sentence sentiment
import re,sys,json
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
D=[(1,0),(0,1),(-1,0),(0,-1)]
PW,PH,M=148,210,12  # A5 mm, margin
for fn in sys.argv[1:]:
    t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
    sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
    x=y=0;d=0;pts=[(0,0)];turns=[0,0,0]
    for s in sents:
        ws=re.findall(r"[a-z']+",s.lower())
        k=sum(lex.get(w,0) for w in ws); tr=(k>0)-(k<0); turns[tr+1]+=1
        d=(d+tr)%4; dx,dy=D[d]; x+=dx*len(ws); y+=dy*len(ws); pts.append((x,y))
    xs=[p[0] for p in pts];ys=[p[1] for p in pts]
    W,H=max(xs)-min(xs),max(ys)-min(ys)
    rot = (W>H)  # put long side vertical
    if rot: pts=[(p[1],-p[0]) for p in pts]; W,H=H,W
    xs=[p[0] for p in pts];ys=[p[1] for p in pts]
    sc=min((PW-2*M)/W,(PH-2*M)/H)
    ox=(PW-W*sc)/2-min(xs)*sc; oy=(PH-H*sc)/2-min(ys)*sc
    path=" ".join(f"{p[0]*sc+ox:.2f},{p[1]*sc+oy:.2f}" for p in pts)
    name=fn.split('/')[-1].split('.')[0]
    open(f"{name}.svg","w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{PW}mm" height="{PH}mm" viewBox="0 0 {PW} {PH}"><rect width="100%" height="100%" fill="white"/><polyline points="{path}" fill="none" stroke="black" stroke-width="0.08" stroke-linejoin="miter" shape-rendering="crispEdges"/></svg>')
    print(f"{name}: {len(sents)} sentences, neg/neutral/pos turns {turns}, scale {sc:.3f} mm/word")
