# Ground view v3
#  - every sentence stroke is a PATHWAY; the sentence is written along it as seen from above:
#    baseline along the right-hand wall, tops of the letters towards the left-hand wall
#  - tall walls on both sides (solid mass between pathways); crossing pathways open into each other
#  - camera always inside a pathway, looking along it
#  - lighting: the sun through haze. walls cast shadows on floor and walls; sunlit haze glows (light shafts);
#    distance fades into haze. 1-bit output via error diffusion.
import re,sys,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
from numba import njit
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
fn,out=sys.argv[1],sys.argv[2]; pick=int(sys.argv[3]) if len(sys.argv)>3 else 0
SUN_AZ=float(sys.argv[4]) if len(sys.argv)>4 else 35.0   # degrees, relative to view direction (+ = left)
SUN_EL=float(sys.argv[5]) if len(sys.argv)>5 else 52.0
t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
D=[(1,0),(0,1),(-1,0),(0,-1)]
x=y=0;d=0;segs=[]
for i,s in enumerate(sents):
    toks=s.split(); ws=re.findall(r"[a-z']+",s.lower()); k=sum(lex.get(w,0) for w in ws)
    d=(d+(k>0)-(k<0))%4; n=max(1,len(toks)); dx,dy=D[d]
    segs.append(dict(x0=x,y0=y,x1=x+dx*n,y1=y+dy*n,d=d,i=i,text=s,n=n)); x+=dx*n; y+=dy*n
HW=0.26      # half width of a pathway (1 unit = 1 word of length)
TH=0.40      # height of the lettering across the pathway
WALL=float(sys.argv[7]) if len(sys.argv)>7 else 1.2   # wall height used for sunlight only (walls are drawn infinitely tall)
EYE=0.30     # eye height
def rect(s,pad=HW):
    return (min(s['x0'],s['x1'])-pad,min(s['y0'],s['y1'])-pad,max(s['x0'],s['x1'])+pad,max(s['y0'],s['y1'])+pad)
def inter(a,b): return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
# --- viewpoint: a few words before a sentence ends, where other pathways cross the stretch ahead
rs=[rect(s,0) for s in segs]; LOOK=7
def tail(j):
    s=segs[j];dx,dy=D[s['d']]; px,py=s['x1']-dx*LOOK,s['y1']-dy*LOOK
    return (min(px,s['x1']),min(py,s['y1']),max(px,s['x1']),max(py,s['y1']))
def ncross(j):
    r=tail(j); return sum(1 for k,r2 in enumerate(rs) if abs(k-j)>1 and segs[k]['d']%2!=segs[j]['d']%2 and inter(r,r2))
def clean(j):
    s=segs[j];r=rect(s,0.1)
    return not any(segs[k]['d']%2==s['d']%2 and inter(r,rect(segs[k],0.1)) for k in range(len(segs)) if k!=j)
cands=[j for j,s in enumerate(segs) if s['n']>=LOOK+6 and clean(j)]
scored=sorted(cands,key=lambda j:-ncross(j))
cam=segs[scored[pick]]
dx,dy=D[cam['d']]
ex,ey=cam['x1']-dx*(LOOK+2.2)+0.0137,cam['y1']-dy*(LOOK+2.2)+0.0113; yaw=math.atan2(dy,dx)
print("camera on sentence",cam['i'],"| crossings ahead",ncross(scored[pick]),"|",cam['text'][:140])
# --- floor plan + lettering
R=20;ppu=220;N=2*R*ppu
def tw(px,py): return ((px-(ex-R))*ppu,((ey+R)-py)*ppu)
near=[s for s in segs if inter(rect(s),(ex-R-2,ey-R-2,ex+R+2,ey+R+2))]
walk=Image.new('1',(N,N),0); dw=ImageDraw.Draw(walk)
for s in near:
    a,b,c,e=rect(s); dw.rectangle([tw(a,e),tw(c,b)],fill=1)
WALK=np.asarray(walk)
tex=np.full((N,N),255,np.uint8)
font=ImageFont.truetype('fonts/EBGaramond.ttf',200)
try: font.set_variation_by_axes([600])
except Exception: pass
asc,desc=font.getmetrics()
for s in near:
    L=s['n']; txt=s['text']
    wpx=int(font.getlength(txt))+20; im=Image.new('L',(wpx,asc+desc),255)
    ImageDraw.Draw(im).text((10,0),txt,font=font,fill=0)
    im=im.resize((max(1,int(L*ppu)),int(TH*ppu)),Image.LANCZOS)   # justified to the stroke length
    im=im.rotate(90*s['d'],expand=True,fillcolor=255)                # glyph tops point to the left of travel
    cx,cy=(s['x0']+s['x1'])/2,(s['y0']+s['y1'])/2; px,py=tw(cx,cy)
    x0_,y0_=int(px-im.width/2),int(py-im.height/2)
    a=np.asarray(im); xa,ya=max(0,x0_),max(0,y0_); xb,yb_=min(N,x0_+im.width),min(N,y0_+im.height)
    if xa<xb and ya<yb_:
        tex[ya:yb_,xa:xb]=np.minimum(tex[ya:yb_,xa:xb],a[ya-y0_:yb_-y0_,xa-x0_:xb-x0_])
# --- sun: distance from every floor point to the first wall in the sun's (horizontal) direction
az=yaw+math.radians(SUN_AZ); el=math.radians(SUN_EL); tanel=math.tan(el)
sun=np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
# exact shadow test: march from the point towards the sun through the floor plan,
# only as far as the wall height can matter ((WALL - z) / tan(elevation))
@njit(cache=True)
def _lit(X,Y,Z,walk,ox,oy,ppu,sx,sy,tanel,wall,out):
    n=X.size; N_=walk.shape[0]; st=0.5/ppu
    for k in range(n):
        z=Z[k]
        if z>=wall: out[k]=True; continue
        smax=(wall-z)/tanel; t=st; ok=True
        while t<smax:
            px=X[k]+sx*t; py=Y[k]+sy*t
            ix=int((px-ox)*ppu); iy=int((oy-py)*ppu)
            if ix<0 or iy<0 or ix>=N_ or iy>=N_ or not walk[iy,ix]: ok=False; break
            t+=st
        out[k]=ok
def lit(X,Y,Z):
    X,Y,Z=np.broadcast_arrays(X,Y,Z); sh=X.shape
    o=np.empty(X.size,np.bool_)
    _lit(np.ascontiguousarray(X,np.float64).ravel(),np.ascontiguousarray(Y,np.float64).ravel(),np.ascontiguousarray(Z,np.float64).ravel(),
         WALK,ex-R,ey+R,float(ppu),math.cos(az),math.sin(az),tanel,WALL,o)
    return o.reshape(sh)
# --- camera rays
W,H=3600,2400;fov=math.radians(72);f=(W/2)/math.tan(fov/2);hor=int(H*0.34)
cols=np.arange(W);lat=(cols-W/2)/f;ang=yaw-np.arctan(lat)
step=0.6/ppu;ts=np.arange(step,R-0.5,step,dtype=np.float32)
zw=np.empty(W,np.float32);hx=np.empty(W,np.float32);hy=np.empty(W,np.float32);nx=np.empty(W);ny=np.empty(W)
for c0 in range(0,W,60):
    a=ang[c0:c0+60];jj=np.arange(len(a))
    X=ex+np.outer(ts,np.cos(a));Y=ey+np.outer(ts,np.sin(a))
    ix=((X-(ex-R))*ppu).astype(np.int32);iy=(((ey+R)-Y)*ppu).astype(np.int32)
    inside=WALK[np.clip(iy,0,N-1),np.clip(ix,0,N-1)]
    hit=(~inside).any(axis=0);first=np.argmax(~inside,axis=0);pf=np.maximum(first-1,0)
    th_=np.where(hit,ts[first],R)
    zw[c0:c0+60]=th_*np.cos(a-yaw)
    hx[c0:c0+60]=X[pf,jj];hy[c0:c0+60]=Y[pf,jj]
    xface=~WALK[np.clip(iy[pf,jj],0,N-1),np.clip(ix[first,jj],0,N-1)]
    nx[c0:c0+60]=np.where(xface,-np.sign(np.cos(a)),0);ny[c0:c0+60]=np.where(xface,0,-np.sign(np.sin(a)))
HAZE=float(sys.argv[6]) if len(sys.argv)>6 else 9.0;AMB=0.16;HAZEC=0.93
img=np.zeros((H,W),np.float32)
rows=np.arange(H,dtype=np.float32)
# walls (everything above the wall foot in each column)
yb=np.minimum(H-1,(hor+f*EYE/np.maximum(zw,1e-3)).astype(int))
lam=np.maximum(0,nx*sun[0]+ny*sun[1])
for c in range(W):
    r=rows[:yb[c]]; Zh=EYE+zw[c]*(hor-r)/f
    L_=lit(np.full(r.size,hx[c]),np.full(r.size,hy[c]),Zh)
    col=0.95*(AMB+2.2*lam[c]*L_)
    dd=np.sqrt((zw[c]*zw[c])*(1+lat[c]**2)+(Zh-EYE)**2); T=np.exp(-dd/HAZE)
    img[:yb[c],c]=col*T+HAZEC*(1-T)
# floor
fr=np.arange(hor+1,H);Zf=(f*EYE/(fr-hor)).astype(np.float32)
Xf=ex+Zf[:,None]*(math.cos(yaw)+lat[None,:]*math.sin(yaw));Yf=ey+Zf[:,None]*(math.sin(yaw)-lat[None,:]*math.cos(yaw))
fxp=(Xf-(ex-R))*ppu;fyp=((ey+R)-Yf)*ppu;tx=np.clip(np.floor(fxp).astype(np.int32),0,N-2);ty=np.clip(np.floor(fyp).astype(np.int32),0,N-2)
u=fxp-tx;v=fyp-ty;T_=TEX=tex.astype(np.float32)/255
alb=T_[ty,tx]*(1-u)*(1-v)+T_[ty,tx+1]*u*(1-v)+T_[ty+1,tx]*(1-u)*v+T_[ty+1,tx+1]*u*v
alb=0.06+0.88*alb
Lf=lit(Xf,Yf,np.zeros_like(Xf))
colf=alb*(AMB+1.05*sun[2]*Lf)
dd=np.sqrt(Zf[:,None]**2*(1+lat[None,:]**2)+EYE**2);T=np.exp(-dd/HAZE)
colf=colf*T+HAZEC*(1-T)
mask=Zf[:,None]<zw[None,:]
img[hor+1:]=np.where(mask,colf,img[hor+1:])
# light shafts: in-scattering of sunlight by the haze along each view ray (coarse grid, upsampled)
cs,rs_=W//6,H//6
cc=np.arange(cs)*6+3;rr=np.arange(rs_)*6+3
la=(cc-W/2)/f; va=(hor-rr)/f
dirx=(math.cos(yaw)+la*math.sin(yaw));diry=(math.sin(yaw)-la*math.cos(yaw))
S=np.zeros((rs_,cs),np.float32);NS=90
for k in range(NS):
    z=(k+0.5)/NS*R*0.9                      # forward depth
    X=ex+z*dirx[None,:];Y=ey+z*diry[None,:];Z=EYE+z*va[:,None]
    inside=(z<zw[cc][None,:])
    Lk=lit(np.broadcast_to(X,(rs_,cs)),np.broadcast_to(Y,(rs_,cs)),Z)
    vd=np.stack([np.broadcast_to(dirx,(rs_,cs)),np.broadcast_to(diry,(rs_,cs)),np.broadcast_to(va[:,None],(rs_,cs))])
    cosg=(vd[0]*sun[0]+vd[1]*sun[1]+vd[2]*sun[2])/np.linalg.norm(vd,axis=0)
    phase=(1-0.5**2)/(1+0.5**2-2*0.5*cosg)**1.5             # forward-scattering haze
    S+=inside*Lk*np.exp(-z/HAZE)*phase*(R*0.9/NS)
S=np.asarray(Image.fromarray(S).resize((W,H),Image.BILINEAR))
img=img*(1-0.10*S.clip(0,6))+0.10*S
img=np.clip(img,0,1)**1.15
@njit(cache=True)
def fs(a):
    h,w=a.shape;o=np.zeros((h,w),np.uint8)
    for y in range(h):
        rng=range(w) if y%2==0 else range(w-1,-1,-1);sd=1 if y%2==0 else -1
        for x in rng:
            old=a[y,x];new=1.0 if old>=0.5 else 0.0;o[y,x]=255 if new>0 else 0;e=old-new
            if 0<=x+sd<w: a[y,x+sd]+=e*7/16
            if y+1<h:
                if 0<=x-sd<w: a[y+1,x-sd]+=e*3/16
                a[y+1,x]+=e*5/16
                if 0<=x+sd<w: a[y+1,x+sd]+=e*1/16
    return o
Image.fromarray(fs(img.astype(np.float64))).convert('1').save(out)
