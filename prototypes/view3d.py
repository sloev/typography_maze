# prototype: first-person view inside the maze. walls = sentence strokes (infinitely tall), sentences written on the floor
import re,sys,math,numpy as np
from PIL import Image,ImageDraw,ImageFont
lex={}
for l in open('texts/AFINN-en-165.txt'):
    w,s=l.rstrip('\n').rsplit('\t',1); lex[w]=int(s)
def body(t):
    s=t.find('*** START'); s=t.find('\n',s); e=t.find('*** END'); return t[s:e]
fn=sys.argv[1]; out=sys.argv[2]
t=re.sub(r'\s+',' ',body(open(fn,encoding='utf-8').read()))
sents=[s for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]
D=[(1,0),(0,1),(-1,0),(0,-1)]
x=y=0;d=0;segs=[]
for i,s in enumerate(sents):
    ws=re.findall(r"[a-z']+",s.lower()); k=sum(lex.get(w,0) for w in ws)
    d=(d+(k>0)-(k<0))%4; n=max(1,len(ws)); dx,dy=D[d]
    segs.append((x,y,x+dx*n,y+dy*n,d,i)); x+=dx*n; y+=dy*n
S=np.array([s[:4] for s in segs],float)
# --- crossing hotspot: count distinct strokes per 1x1 cell, then densest 24x24 window
xmin,ymin=S[:,[0,2]].min(),S[:,[1,3]].min(); xmax,ymax=S[:,[0,2]].max(),S[:,[1,3]].max()
G=np.zeros((int(ymax-ymin)+2,int(xmax-xmin)+2),np.int32)
for x0,y0,x1,y1 in S:
    a,b=sorted((x0,x1));c,e=sorted((y0,y1))
    G[int(c-ymin):int(e-ymin)+1,int(a-xmin):int(b-xmin)+1]+=1
from numpy.lib.stride_tricks import sliding_window_view
win=24; ii=np.cumsum(np.cumsum(np.pad(G,((1,0),(1,0))),0),1)
dens=ii[win:,win:]-ii[:-win,win:]-ii[win:,:-win]+ii[:-win,:-win]
cy,cx=np.unravel_index(dens.argmax(),dens.shape)
hx,hy=cx+xmin+win/2,cy+ymin+win/2
print("hotspot",hx,hy,"stroke-cells",dens.max())
near=S[(np.minimum(S[:,0],S[:,2])<hx+90)&(np.maximum(S[:,0],S[:,2])>hx-90)&(np.minimum(S[:,1],S[:,3])<hy+90)&(np.maximum(S[:,1],S[:,3])>hy-90)]
nearidx=[i for i,s in enumerate(segs) if (min(s[0],s[2])<hx+90 and max(s[0],s[2])>hx-90 and min(s[1],s[3])<hy+90 and max(s[1],s[3])>hy-90)]
def cast(ox,oy,ang):
    rx,ry=math.cos(ang),math.sin(ang); best=1e9;bid=-1
    x0,y0,x1,y1=near.T
    # segments axis aligned: param intersection
    den=rx*(y1-y0)-ry*(x1-x0)
    with np.errstate(divide='ignore',invalid='ignore'):
        tt=((x0-ox)*(y1-y0)-(y0-oy)*(x1-x0))/den
        uu=((x0-ox)*ry-(y0-oy)*rx)/den
    ok=(den!=0)&(tt>1e-6)&(uu>=0)&(uu<=1)
    if ok.any():
        k=np.where(ok,tt,1e9).argmin(); return tt[k],k
    return 1e9,-1
# --- choose viewpoint: stand beside a long stroke inside the hotspot, look along it
cands=[sg for sg in segs if abs(sg[2]-sg[0])+abs(sg[3]-sg[1])>=14 and abs((sg[0]+sg[2])/2-hx)<win and abs((sg[1]+sg[3])/2-hy)<win]
best=None
for sg in cands:
    x0,y0,x1,y1,dd,_=sg; dx,dy=D[dd]
    for side in (1,-1):
        lx,ly=-dy*side,dx*side
        ox,oy=x0+dx*1.5+lx*0.75,y0+dy*1.5+ly*0.75; yaw=math.atan2(dy,dx)
        hits=[cast(ox,oy,yaw+math.radians(a_)) for a_ in range(-40,41,4)]
        ids={h[1] for h in hits}; vista=min(cast(ox,oy,yaw)[0],40)
        sc=len(ids)*vista
        if best is None or sc>best[0]: best=(sc,ox,oy,yaw)
_,ex,ey,yaw=best; print('eye',ex,ey,math.degrees(yaw))
# --- floor texture: sentence text along each stroke, on the left side of travel
R=70;ppu=48;T=Image.new('L',(2*R*ppu,2*R*ppu),255)
font=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',int(ppu*0.34))
for i in nearidx:
    x0,y0,x1,y1,dd,si=segs[i]; L=abs(x1-x0)+abs(y1-y0)
    txt=sents[si]; tw=int(font.getlength(txt))+2; th=int(ppu*0.42)
    st=Image.new('L',(tw,th),255); ImageDraw.Draw(st).text((0,0),txt,font=font,fill=0)
    st=st.resize((int(L*ppu),th))
    st=st.rotate(90*dd,expand=True,fillcolor=255)
    # same sentence on both faces of the wall, reading in the direction of travel
    for side,img_ in ((1,st),(-1,st.rotate(180))):
        lx,ly=-D[dd][1]*side,D[dd][0]*side
        bx=[x0+lx*0.06,x1+lx*0.06,x0+lx*0.48,x1+lx*0.48]; by=[y0+ly*0.06,y1+ly*0.06,y0+ly*0.48,y1+ly*0.48]
        px=int((min(bx)-(ex-R))*ppu); py=int(((ey+R)-max(by))*ppu)
        T.paste(0,(px,py),Image.eval(img_,lambda v:255-v))
Tn=np.asarray(T)
# --- render
W,H=2400,1500;fov=math.radians(84);f=(W/2)/math.tan(fov/2);hor=int(H*0.42);h=0.45
img=np.full((H,W),255,np.uint8)
fx,fy=math.cos(yaw),math.sin(yaw);rx_,ry_=math.sin(yaw),-math.cos(yaw)
zw=np.zeros(W);wid=np.zeros(W,int)
for c in range(W):
    a=yaw-math.atan((c-W/2)/f); dist,k=cast(ex,ey,a); zw[c]=dist*math.cos(a-yaw); wid[c]=k
rows=np.arange(hor+1,H); z=f*h/(rows-hor)
cols=np.arange(W); lat=(cols-W/2)/f
Z=z[:,None]; X=ex+Z*fx+Z*lat[None,:]*rx_; Y=ey+Z*fy+Z*lat[None,:]*ry_
tx=((X-(ex-R))*ppu).astype(int); ty=(((ey+R)-Y)*ppu).astype(int)
ok=(tx>=0)&(tx<Tn.shape[1])&(ty>=0)&(ty<Tn.shape[0])&(Z<zw[None,:])
fl=np.full(ok.shape,255,np.uint8); fl[ok]=Tn[ty[ok],tx[ok]]
img[hor+1:]=fl
# walls: white, black base line, black edges where wall changes, sparse hatching on one wall orientation
for c in range(W):
    yb=int(min(H-1,hor+f*h/max(zw[c],1e-3)))
    img[:yb,c]=255
    img[max(0,yb-2):yb+1,c]=0
    seg=near[wid[c]]
    if seg[1]==seg[3] and (c%7==0): img[:yb:,c]=np.where(np.arange(yb)%2==0,0,255)  # x-parallel walls get a dotted hatch
for c in range(1,W):
    if wid[c]!=wid[c-1] or abs(zw[c]-zw[c-1])>0.05*zw[c]:
        yb=int(min(H-1,hor+f*h/max(min(zw[c],zw[c-1]),1e-3))); img[:yb,c-1:c+2]=0
Image.fromarray(np.where(img<140,0,255).astype(np.uint8)).convert('1').save(out)
