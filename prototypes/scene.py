# Ground view v4: walls whose height comes from the text they line.
#  - every sentence stroke is a PATHWAY, the sentence written along it as seen from above
#    (baseline on the right-hand wall, tops of the letters towards the left-hand wall)
#  - both sides of every word are lined by a wall segment whose HEIGHT is set by that word:
#        h(word) = H0 + H_LEN * letters + H_SENT * |sentiment|
#    so each sentence raises its own skyline; emotionally charged words stand up as towers
#  - crossing pathways cut openings through each other's walls
#  - camera always inside a pathway; sun through haze, exact shadows; sky visible above the walls
#  - rendering: exact grid traversal (DDA) of a height field, front-to-back with a y-buffer; 1-bit error diffusion
import re,sys,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
from numba import njit
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
arg=lambda i,d:float(sys.argv[i]) if len(sys.argv)>i else d
fn,out=sys.argv[1],sys.argv[2]; pick=int(arg(3,0)); SUN_AZ=arg(4,40.0); SUN_EL=arg(5,50.0); HAZE=arg(6,12.0)
MODE=sys.argv[7] if len(sys.argv)>7 else 'walk'
H0,H_LEN,H_SENT=0.35,0.14,0.55
t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
D=[(1,0),(0,1),(-1,0),(0,-1)]
def wscore(w): return sum(lex.get(v,0) for v in re.findall(r"[a-z']+",w.lower()))
def wheight(w): return H0+H_LEN*len(re.sub(r'\W','',w))+H_SENT*abs(wscore(w))
x=y=0;d=0;segs=[]
for i,s in enumerate(sents):
    toks=s.split(); k=sum(wscore(w) for w in toks)
    d=(d+(k>0)-(k<0))%4; n=max(1,len(toks)); dx,dy=D[d]
    segs.append(dict(x0=x,y0=y,x1=x+dx*n,y1=y+dy*n,d=d,i=i,text=s,n=n,words=toks)); x+=dx*n; y+=dy*n
HW=0.26; TW=0.30; TH=0.40; EYE=0.30
def rect(s,pad=HW):
    return (min(s['x0'],s['x1'])-pad,min(s['y0'],s['y1'])-pad,max(s['x0'],s['x1'])+pad,max(s['y0'],s['y1'])+pad)
def inter(a,b): return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
# --- viewpoint: always where the most pathways cross the stretch ahead
# exact crossings: every horizontal stroke against every vertical stroke (corridor width included)
AHEAD=10
Hs=[s for s in segs if s['d']%2==0]; Vs=[s for s in segs if s['d']%2==1]
def arr(L,keys): return np.array([[s[k] for k in keys] for s in L],float)
hA=arr(Hs,['x0','x1','y0','i']); vA=arr(Vs,['y0','y1','x0','i'])
hlo,hhi=np.minimum(hA[:,0],hA[:,1]),np.maximum(hA[:,0],hA[:,1]); vlo,vhi=np.minimum(vA[:,0],vA[:,1]),np.maximum(vA[:,0],vA[:,1])
cross={s['i']:[] for s in segs}
m=HW*2
for k in range(len(Hs)):
    hit=(vA[:,2]>hlo[k]-m)&(vA[:,2]<hhi[k]+m)&(hA[k,2]>vlo-m)&(hA[k,2]<vhi+m)&(np.abs(vA[:,3]-hA[k,3])>1)
    for j in np.nonzero(hit)[0]:
        hs,vs=Hs[k],Vs[j]
        cross[hs['i']].append(abs(vs['x0']-hs['x0']))   # distance along the horizontal stroke
        cross[vs['i']].append(abs(hs['y0']-vs['y0']))   # distance along the vertical stroke
best=[]
for s in segs:
    c=np.array(sorted(cross[s['i']]))
    if len(c)==0 or s['n']<4: continue
    for p in range(0,s['n']):
        cnt=int(((c>p+0.6)&(c<=p+AHEAD)).sum())
        if not np.any(np.abs(c-(p+0.3))<HW+0.05): best.append((cnt,s['i'],p))
best.sort(key=lambda b:(-b[0],b[1]))
seen=set();picks=[]
for b_ in best:                      # one viewpoint per sentence
    if b_[1] in seen: continue
    seen.add(b_[1]);picks.append(b_)
cnt,si,p=picks[pick]; cam=segs[si]; dx,dy=D[cam['d']]
ex,ey=cam['x0']+dx*(p+0.3)+0.0137,cam['y0']+dy*(p+0.3)+0.0113; yaw=math.atan2(dy,dx)
print("crossings ahead:",cnt)
print("camera on sentence",cam['i'],"|",cam['text'][:140])
R=26
near=[s for s in segs if inter(rect(s,HW+TW),(ex-R-2,ey-R-2,ex+R+2,ey+R+2))]
# --- height field: walls painted around every word tile, corridors carved out afterwards
PH=100; NH=2*R*PH
def gh(px,py): return int(round((px-(ex-R))*PH)),int(round(((ey+R)-py)*PH))
HM=np.zeros((NH,NH),np.float32)
for s in near:
    ddx,ddy=D[s['d']]
    for k,w in enumerate(s['words']):
        ax,ay=s['x0']+ddx*k,s['y0']+ddy*k; bx,by=ax+ddx,ay+ddy
        p=HW+TW
        x0_,y0_=gh(min(ax,bx)-(p if ddx==0 else 0),max(ay,by)+(p if ddy==0 else 0))
        x1_,y1_=gh(max(ax,bx)+(p if ddx==0 else 0),min(ay,by)-(p if ddy==0 else 0))
        x0_,y0_=max(0,x0_),max(0,y0_); x1_,y1_=min(NH,x1_),min(NH,y1_)
        if x0_<x1_ and y0_<y1_: HM[y0_:y1_,x0_:x1_]=np.maximum(HM[y0_:y1_,x0_:x1_],wheight(w))
for s in near:   # carve pathways (including a half-width cap at each end so turns connect)
    a,b,c,e=rect(s); x0_,y0_=gh(a,e); x1_,y1_=gh(c,b)
    HM[max(0,y0_):max(0,y1_),max(0,x0_):max(0,x1_)]=0
# --- lettering texture (floor)
ppu=200;N=2*R*ppu
def tw(px,py): return ((px-(ex-R))*ppu,((ey+R)-py)*ppu)
tex=np.full((N,N),255,np.uint8)
font=ImageFont.truetype('fonts/EBGaramond.ttf',200)
try: font.set_variation_by_axes([600])
except Exception: pass
asc,desc=font.getmetrics()
for s in near:
    L=s['n']; txt=s['text']
    im=Image.new('L',(int(font.getlength(txt))+20,asc+desc),255); ImageDraw.Draw(im).text((10,0),txt,font=font,fill=0)
    im=im.resize((max(1,int(L*ppu)),int(TH*ppu)),Image.LANCZOS).rotate(90*s['d'],expand=True,fillcolor=255)
    px,py=tw((s['x0']+s['x1'])/2,(s['y0']+s['y1'])/2); x0_,y0_=int(px-im.width/2),int(py-im.height/2)
    a=np.asarray(im); xa,ya=max(0,x0_),max(0,y0_); xb,yb=min(N,x0_+im.width),min(N,y0_+im.height)
    if xa<xb and ya<yb: tex[ya:yb,xa:xb]=np.minimum(tex[ya:yb,xa:xb],a[ya-y0_:yb-y0_,xa-x0_:xb-x0_])
