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
fn,out=sys.argv[1],sys.argv[2]; pick=int(arg(3,0)); SUN_AZ=arg(4,40.0); SUN_EL=arg(5,38.0); HAZE=arg(6,9.0)
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
cam=segs[scored[pick]]; dx,dy=D[cam['d']]
ex,ey=cam['x1']-dx*(LOOK+2.2)+0.0137,cam['y1']-dy*(LOOK+2.2)+0.0113; yaw=math.atan2(dy,dx)
print("camera on sentence",cam['i'],"|",cam['text'][:140])
R=22
near=[s for s in segs if inter(rect(s,HW+TW),(ex-R-2,ey-R-2,ex+R+2,ey+R+2))]
# --- height field: walls painted around every word tile, corridors carved out afterwards
PH=60; NH=2*R*PH
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
# --- renderer
W,H=3600,2400;fov=math.radians(72);f=(W/2)/math.tan(fov/2);hor=int(H*0.40)
@njit(cache=True)
def trace(HM,ox,oy,PH,ex,ey,yaw,f,W,H,hor,eye,R,PX,PY,PZ,DEP,KIND,NX,NY):
    NH=HM.shape[0]
    for c in range(W):
        lat=(c-W/2)/f; a=yaw-math.atan(lat); dx=math.cos(a); dy=math.sin(a); cth=math.cos(a-yaw)
        gx=(ex-ox)*PH; gy=(oy-ey)*PH; vx=dx; vy=-dy           # grid space (y down)
        ix=int(math.floor(gx)); iy=int(math.floor(gy))
        sx=1 if vx>0 else -1; sy=1 if vy>0 else -1
        tdx=abs(1.0/vx) if vx!=0 else 1e30; tdy=abs(1.0/vy) if vy!=0 else 1e30
        tmx=((ix+1-gx) if vx>0 else (gx-ix))*tdx; tmy=((iy+1-gy) if vy>0 else (gy-iy))*tdy
        ybuf=H; prevh=HM[iy,ix]; tin=0.0; axis=-1
        while ybuf>0:
            if tmx<tmy: tout=tmx; nax=0
            else: tout=tmy; nax=1
            h=HM[iy,ix]; t0=tin/PH; t1=tout/PH
            # vertical face at entry (only if this cell is higher than the one we came from)
            if h>prevh and t0>1e-6:
                zp=t0*cth; ytop=hor-f*(h-eye)/zp
                r0=int(math.ceil(max(ytop,0.0)))
                for r in range(r0,ybuf):
                    PX[r,c]=ex+dx*t0; PY[r,c]=ey+dy*t0; PZ[r,c]=eye+(hor-r)*zp/f; DEP[r,c]=t0
                    KIND[r,c]=3
                    if axis==0: NX[r,c]=-sx; NY[r,c]=0
                    else: NX[r,c]=0; NY[r,c]=sy      # grid y is flipped
                if r0<ybuf: ybuf=r0
            # horizontal surface of the cell (floor if h==0, otherwise a wall top)
            if h<eye and ybuf>0:
                zp1=max(t1*cth,1e-6); ytop=hor+f*(eye-h)/zp1
                r0=int(math.ceil(max(ytop,0.0)))
                for r in range(r0,ybuf):
                    zz=f*(eye-h)/(r-hor) if r>hor else 1e9; tt=zz/cth
                    PX[r,c]=ex+dx*tt; PY[r,c]=ey+dy*tt; PZ[r,c]=h; DEP[r,c]=tt
                    KIND[r,c]=1 if h==0 else 2; NX[r,c]=0; NY[r,c]=0
                if r0<ybuf: ybuf=r0
            prevh=h; tin=tout; axis=nax
            if nax==0: ix+=sx; tmx+=tdx
            else: iy+=sy; tmy+=tdy
            if ix<0 or iy<0 or ix>=NH or iy>=NH or tin/PH>R: break
        for r in range(0,ybuf): KIND[r,c]=0; DEP[r,c]=1e9
PX=np.zeros((H,W),np.float32);PY=np.zeros((H,W),np.float32);PZ=np.zeros((H,W),np.float32);DEP=np.zeros((H,W),np.float32)
KIND=np.zeros((H,W),np.int8);NX=np.zeros((H,W),np.int8);NY=np.zeros((H,W),np.int8)
trace(HM,ex-R,ey+R,float(PH),ex,ey,yaw,f,W,H,hor,EYE,float(R),PX,PY,PZ,DEP,KIND,NX,NY)
az=yaw+math.radians(SUN_AZ); el=math.radians(SUN_EL); tanel=math.tan(el); HMAX=float(HM.max())
sun=np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
@njit(cache=True)
def litk(X,Y,Z,HM,ox,oy,PH,sx,sy,tanel,hmax,out):
    NH=HM.shape[0];st=0.4/PH
    for k in range(X.size):
        z=Z[k]; t=st; ok=True
        while z+t*tanel<hmax:
            ix=int((X[k]+sx*t-ox)*PH); iy=int((oy-Y[k]-sy*t)*PH)
            if ix<0 or iy<0 or ix>=NH or iy>=NH: break
            if HM[iy,ix]>z+t*tanel+1e-4: ok=False; break
            t+=st
        out[k]=ok
def lit(X,Y,Z):
    X,Y,Z=[np.ascontiguousarray(v,np.float64).ravel() for v in np.broadcast_arrays(X,Y,Z)]
    o=np.empty(X.size,np.bool_); litk(X,Y,Z,HM,ex-R,ey+R,float(PH),math.cos(az),math.sin(az),tanel,HMAX,o); return o
# nudge surface points off their surface before the shadow test
ox_=PX+NX*0.004; oy_=PY+NY*0.004; oz_=PZ+np.where(KIND<3,0.002,0)
L=lit(ox_,oy_,oz_).reshape(H,W)
# albedo: lettering on the floor, plain walls
fx=(PX-(ex-R))*ppu; fy=((ey+R)-PY)*ppu; tx=np.clip(fx.astype(np.int32),0,N-2); ty=np.clip(fy.astype(np.int32),0,N-2)
u=np.clip(fx-tx,0,1);v=np.clip(fy-ty,0,1);T_=tex.astype(np.float32)/255
lettering=T_[ty,tx]*(1-u)*(1-v)+T_[ty,tx+1]*u*(1-v)+T_[ty+1,tx]*(1-u)*v+T_[ty+1,tx+1]*u*v
alb=np.where(KIND==1,0.05+0.88*lettering,0.90)
lam=np.where(KIND==3,np.maximum(0,NX*sun[0]+NY*sun[1]),sun[2])
AMB=0.20
col=alb*(AMB+1.25*lam*L)
# sky: bright haze, glowing towards the sun
r_=np.arange(H)[:,None];c_=np.arange(W)[None,:]
vx_=np.cos(yaw-np.arctan((c_-W/2)/f));vy_=np.sin(yaw-np.arctan((c_-W/2)/f));vz_=(hor-r_)/f
nrm=np.sqrt(vx_**2+vy_**2+vz_**2);cosg=(vx_*sun[0]+vy_*sun[1]+vz_*sun[2])/nrm
sky=0.86+0.14*np.clip(cosg,0,1)**8
col=np.where(KIND==0,sky,col)
dist=np.where(KIND==0,1e9,DEP*np.sqrt(1+((c_-W/2)/f)**2))
Tr=np.exp(-dist/HAZE); HAZEC=0.93
col=col*Tr+HAZEC*(1-Tr)
# light shafts: in-scattering where the haze is sunlit (coarse, upsampled)
S6=6;cs,rs2=W//S6,H//S6;cc=np.arange(cs)*S6+3;rr=np.arange(rs2)*S6+3
la=(cc-W/2)/f;va=(hor-rr)/f
dxv=np.cos(yaw-np.arctan(la));dyv=np.sin(yaw-np.arctan(la));cth=np.cos(np.arctan(la))
dep=DEP[rr][:,cc]
Sc=np.zeros((rs2,cs),np.float32);NS=80
cg=(dxv[None,:]*sun[0]+dyv[None,:]*sun[1]+va[:,None]*cth[None,:]*sun[2])/np.sqrt(1+(va[:,None]*cth[None,:])**2)
phase=(1-0.45**2)/(1+0.45**2-2*0.45*cg)**1.5
for k in range(NS):
    tt=(k+0.5)/NS*R
    X=ex+tt*dxv[None,:];Y=ey+tt*dyv[None,:];Z=EYE+tt*cth[None,:]*va[:,None]
    X,Y,Z=np.broadcast_arrays(X,Y,Z)
    Lk=lit(X,Y,Z).reshape(rs2,cs)
    Sc+=(tt<dep)*Lk*np.exp(-tt/HAZE)*phase*(R/NS)
Sc=np.asarray(Image.fromarray(Sc).resize((W,H),Image.BILINEAR))
col=col*(1-0.07*np.clip(Sc,0,8))+0.07*np.clip(Sc,0,8)*0.98
img=np.clip(col,0,1)**1.1
@njit(cache=True)
def fs(a):
    h,w=a.shape;o=np.zeros((h,w),np.uint8)
    for y in range(h):
        sd=1 if y%2==0 else -1
        for i in range(w):
            x=i if sd==1 else w-1-i
            old=a[y,x];new=1.0 if old>=0.5 else 0.0;o[y,x]=255 if new>0 else 0;e=old-new
            if 0<=x+sd<w: a[y,x+sd]+=e*7/16
            if y+1<h:
                if 0<=x-sd<w: a[y+1,x-sd]+=e*3/16
                a[y+1,x]+=e*5/16
                if 0<=x+sd<w: a[y+1,x+sd]+=e*1/16
    return o
Image.fromarray(fs(img.astype(np.float64))).convert('1').save(out)
Image.fromarray((img*255).astype(np.uint8)).resize((900,600),Image.LANCZOS).save(out.replace('.png','_tone.png'))
