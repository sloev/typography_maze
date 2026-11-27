# Ground view, corrected model:
#   every sentence stroke is a PATHWAY; its words are written on the floor, one word per unit of length,
#   set across the path like road markings so they read upright as you walk;
#   infinitely tall walls line both sides of each pathway; crossing pathways open into each other.
#   The camera always stands inside a pathway, looking along it.
import re,sys,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
fn,out=sys.argv[1],sys.argv[2]; pick=int(sys.argv[3]) if len(sys.argv)>3 else 0
t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
D=[(1,0),(0,1),(-1,0),(0,-1)]
x=y=0;d=0;segs=[]
for i,s in enumerate(sents):
    toks=s.split(); ws=re.findall(r"[a-z']+",s.lower()); k=sum(lex.get(w,0) for w in ws)
    d=(d+(k>0)-(k<0))%4; n=max(1,len(toks)); dx,dy=D[d]
    segs.append(dict(x0=x,y0=y,x1=x+dx*n,y1=y+dy*n,d=d,i=i,words=toks)); x+=dx*n; y+=dy*n
HW=0.62   # half width of a pathway (word units)
def rect(s,pad=HW):
    return (min(s['x0'],s['x1'])-pad,min(s['y0'],s['y1'])-pad,max(s['x0'],s['x1'])+pad,max(s['y0'],s['y1'])+pad)
def inter(a,b): return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
# --- viewpoint: a long pathway that many other pathways cross (side openings), standing at its start
rs=[rect(s,0) for s in segs]
LOOK=9   # stand this many words before the sentence ends: see openings and the turn ahead
def tail(j):
    s=segs[j];dx,dy=D[s['d']];n=len(s['words'])
    px,py=s['x1']-dx*LOOK,s['y1']-dy*LOOK
    return (min(px,s['x1']),min(py,s['y1']),max(px,s['x1']),max(py,s['y1']))
def ncross(j):
    r=tail(j); return sum(1 for k,r2 in enumerate(rs) if abs(k-j)>1 and segs[k]['d']%2!=segs[j]['d']%2 and inter(r,r2))
def clean(j):
    s=segs[j];r=rect(s,0.1)
    return not any(segs[k]['d']%2==s['d']%2 and inter(r,rect(segs[k],0.1)) for k in range(len(segs)) if k!=j)
cands=[j for j,s in enumerate(segs) if len(s['words'])>=LOOK+6 and clean(j)]
scored=sorted(cands,key=lambda j:-ncross(j))
cam=segs[scored[pick]]
dx,dy=D[cam['d']]; n=len(cam['words'])
ex,ey=cam['x1']-dx*(LOOK+1.5)+0.0137,cam['y1']-dy*(LOOK+1.5)+0.0113; yaw=math.atan2(dy,dx)
print("camera on sentence",cam['i'],"crossings ahead",ncross(scored[pick]),":",' '.join(cam['words'][-LOOK-2:]))
# --- floor plan around the camera
R=26;ppu=190;N=2*R*ppu
def tw(px,py): return (int((px-(ex-R))*ppu),int(((ey+R)-py)*ppu))
near=[s for s in segs if inter(rect(s),(ex-R,ey-R,ex+R,ey+R))]
walk=Image.new('1',(N,N),0); dw=ImageDraw.Draw(walk)
for s in near:
    a,b,c,e=rect(s); p0=tw(a,e);p1=tw(c,b); dw.rectangle([p0,p1],fill=1)
WALK=np.asarray(walk)
tex=Image.new('L',(N,N),255)
font=ImageFont.truetype('fonts/EBGaramond.ttf',160)
try: font.set_variation_by_axes([560])
except Exception: pass
cache={}
def wordimg(w):
    if w in cache: return cache[w]
    l,tp,r,b=font.getbbox(w); im=Image.new('L',(r-l+8,b-tp+8),255)
    ImageDraw.Draw(im).text((4-l,4-tp),w,font=font,fill=0)
    # fit across the pathway (width) and along it (height), stretched like a road marking
    bw,bh=int(2*HW*0.86*ppu),int(0.9*ppu)
    sc=min(bw/im.width, bh/(im.height*1.9)); im=im.resize((max(1,int(im.width*sc)),max(1,int(im.height*sc*1.9))),Image.LANCZOS)
    cache[w]=im; return im
for s in near:
    ddx,ddy=D[s['d']]
    for k,w in enumerate(s['words']):
        cx,cy=s['x0']+ddx*(k+0.5),s['y0']+ddy*(k+0.5)
        if abs(cx-ex)>R-1 or abs(cy-ey)>R-1: continue
        im=wordimg(w).rotate(90*s['d']-90,expand=True,fillcolor=255)
        px,py=tw(cx,cy)
        reg=(px-im.width//2,py-im.height//2)
        tex.paste(Image.composite(Image.new('L',im.size,0),tex.crop((reg[0],reg[1],reg[0]+im.width,reg[1]+im.height)),Image.eval(im,lambda v:255-v)),reg)
TEX=np.asarray(tex)
# --- render (supersampled 2x, then thresholded to 1-bit)
SS=2;W,H=2400*SS,1600*SS;fov=math.radians(74);f=(W/2)/math.tan(fov/2);hor=int(H*0.30);h=0.42
cols=np.arange(W);lat=(cols-W/2)/f;ang=yaw-np.arctan(lat)
step=0.5/ppu;ts=np.arange(step,R-1,step)
zw=np.empty(W);nrm=np.empty(W,int);along=np.empty(W)
for c0 in range(0,W,100):
    a=ang[c0:c0+100];jj=np.arange(len(a))
    X=ex+np.outer(ts,np.cos(a));Y=ey+np.outer(ts,np.sin(a))
    ix=((X-(ex-R))*ppu).astype(np.int32);iy=(((ey+R)-Y)*ppu).astype(np.int32)
    inside=WALK[np.clip(iy,0,N-1),np.clip(ix,0,N-1)]
    hit=(~inside).any(axis=0);first=np.argmax(~inside,axis=0)
    zw[c0:c0+100]=np.where(hit,ts[first],1e6)*np.cos(a-yaw)
    pf=np.maximum(first-1,0)
    xs_wall=~WALK[np.clip(iy[pf,jj],0,N-1),np.clip(ix[first,jj],0,N-1)]   # moving in x alone hits the wall -> x-facing
    n_=xs_wall.astype(int)
    nrm[c0:c0+100]=n_
    along[c0:c0+100]=np.where(n_==1,Y[first,jj],X[first,jj])   # position along the wall, in word units
img=np.full((H,W),255,np.uint8)
rows=np.arange(hor+1,H);z=f*h/(rows-hor)
Z=z[:,None];Xf=ex+Z*math.cos(yaw)+Z*lat[None,:]*math.sin(yaw);Yf=ey+Z*math.sin(yaw)-Z*lat[None,:]*math.cos(yaw)
fxp=(Xf-(ex-R))*ppu;fyp=((ey+R)-Yf)*ppu;tx=np.floor(fxp).astype(np.int32);ty=np.floor(fyp).astype(np.int32)
ok=(Z<zw[None,:])&(tx>=0)&(tx<N-1)&(ty>=0)&(ty<N-1)
fl=np.full(ok.shape,255.0);u=(fxp-tx)[ok];v=(fyp-ty)[ok];a_,b_=ty[ok],tx[ok];T=TEX.astype(np.float32)
fl[ok]=T[a_,b_]*(1-u)*(1-v)+T[a_,b_+1]*u*(1-v)+T[a_+1,b_]*(1-u)*v+T[a_+1,b_+1]*u*v
img[hor+1:]=fl.astype(np.uint8)
yb=np.minimum(H-1,(hor+f*h/np.maximum(zw,1e-3)).astype(int))
for c in range(W):
    img[:yb[c],c]=255
    img[max(0,yb[c]-3*SS):yb[c]+1,c]=0                                  # floor line
    if zw[c]<1e5:
        gap=f*0.3/zw[c]                                                    # ruled courses every 0.3 units of height
        if gap>=7*SS:
            k=1
            while True:
                yy=int(hor-f*(k*0.3-h)/zw[c])
                if yy<0: break
                if yy<yb[c]: img[max(0,yy-SS//2):yy+SS//2+1,c]=0
                k+=1
last=-99
for c in range(1,W):
    top=min(yb[c],yb[c-1])
    if (nrm[c]!=nrm[c-1] or abs(zw[c]-zw[c-1])>0.6) and c-last>5*SS:      # corners and openings, culled when dense
        img[:top,c-SS:c+SS]=0; last=c
    elif math.floor(along[c])!=math.floor(along[c-1]) and c-last>14*SS:   # faint panel joint per word, culled when dense
        img[:top:3,c]=0; last=c
small=Image.fromarray(img).resize((W//SS,H//SS),Image.BOX)
small.point(lambda v:0 if v<140 else 255).convert('1').save(out)
