# crossing statistics for the sentence walk (axis-aligned, integer coords => exact)
import re,sys,collections
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
D=[(1,0),(0,1),(-1,0),(0,-1)]
for fn in sys.argv[1:]:
    t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
    sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
    x=y=0;d=0;H=collections.defaultdict(list);V=collections.defaultdict(list);segs=[]
    for i,s in enumerate(sents):
        ws=re.findall(r"[a-z']+",s.lower());k=sum(lex.get(w,0) for w in ws)
        d=(d+(k>0)-(k<0))%4;n=max(1,len(ws));dx,dy=D[d]
        nx,ny=x+dx*n,y+dy*n
        if dy==0: H[y].append((min(x,nx),max(x,nx),i))
        else: V[x].append((min(y,ny),max(y,ny),i))
        x,y=nx,ny
    # proper crossings H x V (interior points only), via brute force over buckets
    cross=0;gaps=[];per=collections.Counter()
    vs=[(vx,a,b,i) for vx,l in V.items() for a,b,i in l]
    vs.sort()
    import bisect
    keys=[v[0] for v in vs]
    for hy,l in H.items():
        for a,b,i in l:
            lo=bisect.bisect_right(keys,a);hi=bisect.bisect_left(keys,b)
            for vx,c,e,j in vs[lo:hi]:
                if c<hy<e and abs(i-j)>1:
                    cross+=1;gaps.append(abs(i-j));per[i]+=1;per[j]+=1
    # collinear overlaps (sentences written on top of each other)
    ov=0
    for dct in (H,V):
        for l in dct.values():
            l=sorted(l)
            for k in range(len(l)):
                for m in range(k+1,len(l)):
                    if l[m][0]>=l[k][1]: break
                    ov+=min(l[k][1],l[m][1])-l[m][0]
    gaps.sort()
    hub=per.most_common(1)[0]
    print(f"{fn}: {len(sents)} sentences, {cross} crossings ({cross/len(sents):.1f}/sentence), median gap {gaps[len(gaps)//2]} sentences, "
          f"overlap length {ov} words, most-crossed sentence #{hub[0]} x{hub[1]}: \"{sents[hub[0]][:90]}...\"")
