# Standing letters: no walls, no floor text. Every sentence stands on the ground along its stroke as
# big upright 3D letters (glyphs extruded to a slab), baseline on the ground, reading in the walking
# direction when seen from the right-hand side. Crossing sentences pass through each other.
# Stroke length = the sentence's natural set width, so letters keep their true proportions.
import re,sys,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
from numba import njit
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
arg=lambda i,d:float(sys.argv[i]) if len(sys.argv)>i else d
fn,out=sys.argv[1],sys.argv[2]; pick=int(arg(3,0)); SUN_AZ=arg(4,40.0); SUN_EL=arg(5,40.0)
MODE=sys.argv[6] if len(sys.argv)>6 else 'stand'
EM=1.5; PPU=150; THK=0.22                       # em size (units), mask pixels per unit, letter thickness
font=ImageFont.truetype('fonts/EBGaramond.ttf',int(EM*PPU))
try: font.set_variation_by_axes([620])
except Exception: pass
asc,desc=font.getmetrics(); ASC=asc/PPU
t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
D=[(1,0),(0,1),(-1,0),(0,-1)]
x=y=0;d=0;segs=[]
for i,s in enumerate(sents):
    k=sum(lex.get(w,0) for w in re.findall(r"[a-z']+",s.lower()))
    d=(d+(k>0)-(k<0))%4; dx,dy=D[d]
    L=font.getlength(s)/PPU                    # natural width of the sentence in units
    segs.append(dict(x0=x,y0=y,x1=x+dx*L,y1=y+dy*L,d=d,i=i,text=s,L=L)); x+=dx*L; y+=dy*L
# --- the busiest place: most distinct sentences passing within a radius
P=np.array([[s['x0'],s['y0'],s['x1'],s['y1']] for s in segs])
def segdist(px,py):
    ax,ay,bx,by=P.T; vx,vy=bx-ax,by-ay; tt=np.clip(((px-ax)*vx+(py-ay)*vy)/np.maximum(vx*vx+vy*vy,1e-9),0,1)
    return np.hypot(ax+tt*vx-px,ay+tt*vy-py)
cand=[]
for s in segs[::3]:
    for f_ in (0.25,0.5,0.75):
        px,py=s['x0']+(s['x1']-s['x0'])*f_,s['y0']+(s['y1']-s['y0'])*f_
        cand.append((int((segdist(px,py)<8).sum()),px,py,s['i']))
cand.sort(key=lambda c:-c[0]); seen=[];picks=[]
for c in cand:
    if all(math.hypot(c[1]-q[1],c[2]-q[2])>40 for q in picks): picks.append(c)
cnt,jx,jy,si=picks[pick]; base=segs[si]
print("junction with",cnt,"sentences within 8 units, on sentence",si,":",base['text'][:100])
# --- scene: sentences near the junction, glyph masks in one atlas
RAD=70
near=[s for s in segs if segdist(jx,jy)[s['i']]<RAD]
HP=int((asc+desc))+2
offs=[];W_=0
for s in near: offs.append(W_); W_+=int(s['L']*PPU)+4
atlas=np.zeros((HP,W_),np.uint8)
for s,o in zip(near,offs):
    im=Image.new('L',(int(s['L']*PPU)+4,HP),0); ImageDraw.Draw(im).text((0,0),s['text'],font=font,fill=255)
    atlas[:,o:o+im.width]=np.asarray(im)
ATL=(atlas>127)
# per stroke: origin, direction, left normal, length, atlas offset
B=np.array([[s['x0'],s['y0'],D[s['d']][0],D[s['d']][1],s['L'],o] for s,o in zip(near,offs)],np.float64)
print(len(near),"sentences in scene, atlas",ATL.shape)
@njit(cache=True)
def hit_scene(ox,oy,oz,dx,dy,dz,B,ATL,PPU,ASC,THK,tmax,HP,anyhit):
    # letters are flat black cut-outs with no thickness, standing in the vertical plane of their stroke.
    # returns t, kind (0 sky, 1 ground, 2 letter), normal
    best=tmax; kind=0
    if dz<0:
        tg=-oz/dz
        if tg<best: best=tg; kind=1
    for b in range(B.shape[0]):
        x0=B[b,0];y0=B[b,1];ux=B[b,2];uy=B[b,3];L=B[b,4];off=int(B[b,5])
        lx=-uy;ly=ux
        dp=dx*lx+dy*ly
        if abs(dp)<1e-12: continue
        rp=(ox-x0)*lx+(oy-y0)*ly
        t=-rp/dp
        if t<=1e-6 or t>=best: continue
        s_=(ox-x0)*ux+(oy-y0)*uy+(dx*ux+dy*uy)*t; z_=oz+dz*t
        if s_<0 or s_>=L or z_<0 or z_>ASC: continue
        ix=int(s_*PPU); iz=int((ASC-z_)*PPU)
        if iz<0 or iz>=HP: continue
        if ATL[iz,off+ix]:
            if anyhit: return t,2,0.0,0.0,0.0
            best=t; kind=2
    return best,kind,0.0,0.0,1.0
@njit(cache=True)
def render(cx,cy,cz,fwd,right,up,f,W,H,B,ATL,PPU,ASC,THK,HP,tmax,sx,sy,sz,OUT,DEP,KND,LIT,NDS):
    for r in range(H):
        for c in range(W):
            u=(c-W/2+0.5)/f; v=(H/2-r-0.5)/f
            dx=fwd[0]+u*right[0]+v*up[0]; dy=fwd[1]+u*right[1]+v*up[1]; dz=fwd[2]+u*right[2]+v*up[2]
            n=math.sqrt(dx*dx+dy*dy+dz*dz); dx/=n;dy/=n;dz/=n
            t,k,nx,ny,nz=hit_scene(cx,cy,cz,dx,dy,dz,B,ATL,PPU,ASC,THK,tmax,HP,False)
            KND[r,c]=k; DEP[r,c]=t
            NDS[r,c]=dx*sx+dy*sy+dz*sz
            if k==0 or k==2: continue
            px=cx+dx*t;py=cy+dy*t;pz=cz+dz*t
            if k==1: nx=0.0;ny=0.0;nz=1.0
            lam=nx*sx+ny*sy+nz*sz
            if lam<=0: LIT[r,c]=0.0; OUT[r,c]=nz*0.1+abs(nx)*0.05; continue
            # shadow ray
            e=0.03
            t2,k2,a_,b_,c_=hit_scene(px+nx*e,py+ny*e,pz+nz*e,sx,sy,sz,B,ATL,PPU,ASC,THK,60.0,HP,True)
            LIT[r,c]=lam if k2==0 else 0.0
            OUT[r,c]=nz*0.1+abs(nx)*0.05
# --- camera
bx_,by_=D[base['d']]; lx_,ly_=-by_,bx_
H0=0.62
if MODE=='stand':     # on the readable (right-hand) side of the sentence, looking along it into the junction
    cx,cy=jx-bx_*7-lx_*3.2,jy-by_*7-ly_*3.2; cz=H0; tx,ty,tz=jx+bx_*2,jy+by_*2,0.6; fv=66
elif MODE=='among':   # standing in the junction among the letters
    cx,cy=jx-lx_*1.2,jy-ly_*1.2; cz=H0; tx,ty,tz=jx+bx_*10-lx_*1.5,jy+by_*10-ly_*1.5,0.7; fv=74
elif MODE=='high':    # from above and behind, looking down on the junction
    cx,cy=jx-bx_*11-lx_*9,jy-by_*11-ly_*9; cz=9.0; tx,ty,tz=jx,jy,0.0; fv=58
elif MODE=='low':     # eye on the ground, letters towering
    cx,cy=jx-bx_*4-lx_*1.4,jy-by_*4-ly_*1.4; cz=0.12; tx,ty,tz=jx+bx_*4,jy+by_*4,1.1; fv=80
fwd=np.array([tx-cx,ty-cy,tz-cz]); fwd/=np.linalg.norm(fwd)
right=np.cross(fwd,[0,0,1.0]); right/=np.linalg.norm(right); up=np.cross(right,fwd)
W,H=2400,1600; f=(W/2)/math.tan(math.radians(fv)/2)
az=math.atan2(fwd[1],fwd[0])+math.radians(SUN_AZ); el=math.radians(SUN_EL)
sun=np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
OUT,DEP,LIT,NDS=[np.zeros((H,W),np.float64) for _ in range(4)]; KND=np.zeros((H,W),np.int8)
render(cx,cy,cz,fwd,right,up,f,W,H,B,ATL,float(PPU),ASC,THK,HP,200.0,sun[0],sun[1],sun[2],OUT,DEP,KND,LIT,NDS)
# --- shading: letters pure black; the ground is lit by sky + sun and carries the letters' shadows
HAZE=55.0
AMB=0.55
col=np.where(KND==1,0.98*(AMB+0.62*LIT),0.0)
col=np.where(KND==0,0.92+0.08*np.clip(NDS,0,1)**8,col)
Tr=np.exp(-np.where(KND==0,1e9,DEP)/HAZE); col=col*Tr+0.96*(1-Tr)
img=np.clip(col,0,1)
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
Image.fromarray(fs(img.copy())).convert('1').save(out)
Image.fromarray((img*255).astype(np.uint8)).save(out.replace('.png','_tone.png'))
