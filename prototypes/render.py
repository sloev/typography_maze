# general camera: per-pixel 3D ray casting through the word-height field (any position, yaw, pitch)
exec(open('scene.py').read())
tx_,ty_=ex,ey                     # the junction: the camera point chosen by crossing count
def hm_at(px,py):
    ix,iy=int((px-(ex-R))*PH),int(((ey+R)-py)*PH); return float(HM[iy,ix])
dxw,dyw=D[cam['d']]; lx,ly=-dyw,dxw     # left of the reading direction
if MODE=='walk':
    cx,cy,cz,yw,pt,fv=ex,ey,EYE,yaw,0.0,72
elif MODE=='wall':      # standing on top of the left-hand wall a few words back, looking down into the junction
    bx,by=ex-dxw*2.5+lx*(HW+TW/2),ey-dyw*2.5+ly*(HW+TW/2)
    cx,cy,cz=bx,by,hm_at(bx,by)+0.35; yw=yaw-math.radians(16); pt=-40.0; fv=70
elif MODE=='oblique':   # a bird's view from behind and above
    dist=6.5; pt=-40.0; yw=yaw-math.radians(30)
    cx,cy,cz=ex-math.cos(yw)*dist*math.cos(math.radians(pt)),ey-math.sin(yw)*dist*math.cos(math.radians(pt)),0.2+dist*math.sin(-math.radians(pt)); fv=62
elif MODE=='aerial':
    dist=11; pt=-62.0; yw=yaw-math.radians(35)
    cx,cy,cz=ex-math.cos(yw)*dist*math.cos(math.radians(pt)),ey-math.sin(yw)*dist*math.cos(math.radians(pt)),dist*math.sin(-math.radians(pt)); fv=58
elif MODE=='plan':      # straight down
    cx,cy,cz,yw,pt,fv=ex,ey,8.0,yaw,-89.9,64
elif MODE=='worm':      # eye almost on the floor, looking up at the towers
    cx,cy,cz,yw,pt,fv=ex,ey,0.06,yaw,14.0,84
elif MODE=='corner':    # in the junction, looking down the crossing pathway, slightly downwards
    cx,cy,cz,yw,pt,fv=ex+dxw*0.9,ey+dyw*0.9,0.55,yaw+math.pi/2,-14.0,76
HAZE=HAZE*{'oblique':2.5,'aerial':4,'plan':5,'wall':1.6}.get(MODE,1)
print("mode",MODE,"camera",round(cx,2),round(cy,2),round(cz,2),"pitch",pt)
W,H=2400,1600
p=math.radians(pt)
fwd=np.array([math.cos(p)*math.cos(yw),math.cos(p)*math.sin(yw),math.sin(p)])
right=np.array([math.sin(yw),-math.cos(yw),0.0]); up=np.cross(right,fwd)
f=(W/2)/math.tan(math.radians(fv)/2)
@njit(cache=True)
def cast(HM,X0,Y1,PH,ox,oy,oz,fwd,right,up,f,W,H,hmax,rmax,PX,PY,PZ,DEP,KIND,NX,NY,DX,DY,DZ):
    NH=HM.shape[0]
    for r in range(H):
        for c in range(W):
            u=(c-W/2+0.5)/f; v=(H/2-r-0.5)/f
            dx=fwd[0]+u*right[0]+v*up[0]; dy=fwd[1]+u*right[1]+v*up[1]; dz=fwd[2]+u*right[2]+v*up[2]
            n=math.sqrt(dx*dx+dy*dy+dz*dz); dx/=n; dy/=n; dz/=n
            DX[r,c]=dx; DY[r,c]=dy; DZ[r,c]=dz
            gx=(ox-X0)*PH; gy=(Y1-oy)*PH; vx=dx; vy=-dy
            ix=int(math.floor(gx)); iy=int(math.floor(gy))
            sx=1 if vx>0 else -1; sy=1 if vy>0 else -1
            tdx=1.0/(PH*abs(vx)) if vx!=0 else 1e30; tdy=1.0/(PH*abs(vy)) if vy!=0 else 1e30
            tmx=(((ix+1-gx) if vx>0 else (gx-ix))/PH)/abs(vx) if vx!=0 else 1e30
            tmy=(((iy+1-gy) if vy>0 else (gy-iy))/PH)/abs(vy) if vy!=0 else 1e30
            t=0.0; axis=-1; kind=0
            while True:
                if ix<0 or iy<0 or ix>=NH or iy>=NH or t>rmax: break
                h=HM[iy,ix]
                tout=min(tmx,tmy)
                zin=oz+dz*t; zout=oz+dz*tout
                if zin<=h and t>0:
                    kind=3; th=t; break
                if dz<0 and zout<=h:
                    th=(h-oz)/dz; kind=1 if h==0 else 2; break
                if dz>=0 and zin>hmax: break
                t=tout
                if tmx<tmy: ix+=sx; tmx+=tdx; axis=0
                else: iy+=sy; tmy+=tdy; axis=1
            KIND[r,c]=kind
            if kind==0: DEP[r,c]=1e9; continue
            PX[r,c]=ox+dx*th; PY[r,c]=oy+dy*th; PZ[r,c]=oz+dz*th; DEP[r,c]=th
            if kind==3:
                if axis==0: NX[r,c]=-sx; NY[r,c]=0
                else: NX[r,c]=0; NY[r,c]=sy
            else: NX[r,c]=0; NY[r,c]=0
Z32=lambda:np.zeros((H,W),np.float32); Z8=lambda:np.zeros((H,W),np.int8)
PX,PY,PZ,DEP,DX,DY,DZ=[Z32() for _ in range(7)]; KIND,NX,NY=Z8(),Z8(),Z8()
HMAX=float(HM.max())
cast(HM,ex-R,ey+R,float(PH),cx,cy,cz,fwd,right,up,f,W,H,HMAX,2*R,PX,PY,PZ,DEP,KIND,NX,NY,DX,DY,DZ)
az=yaw+math.radians(SUN_AZ); el=math.radians(SUN_EL); tanel=math.tan(el)
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
L=lit(PX+NX*0.004,PY+NY*0.004,PZ+np.where(KIND<3,0.002,0)).reshape(H,W)
fx=(PX-(ex-R))*ppu; fy=((ey+R)-PY)*ppu; txi=np.clip(fx.astype(np.int32),0,N-2); tyi=np.clip(fy.astype(np.int32),0,N-2)
u=np.clip(fx-txi,0,1);v=np.clip(fy-tyi,0,1);T_=tex.astype(np.float32)/255
lettering=T_[tyi,txi]*(1-u)*(1-v)+T_[tyi,txi+1]*u*(1-v)+T_[tyi+1,txi]*(1-u)*v+T_[tyi+1,txi+1]*u*v
alb=np.where(KIND==1,0.03+0.95*lettering,0.96)
lam=np.where(KIND==3,np.maximum(0,NX*sun[0]+NY*sun[1]),sun[2])
sky_face=np.where(KIND==3,np.where(NX!=0,0.86,0.74),np.where(KIND==2,1.05,1.0))
AMB=0.62
col=alb*(AMB*sky_face+0.75*lam*L)
cosg=DX*sun[0]+DY*sun[1]+DZ*sun[2]
sky=0.88+0.12*np.clip(cosg,0,1)**8
col=np.where(KIND==0,sky,col)
Tr=np.exp(-np.where(KIND==0,1e9,DEP)/HAZE); col=col*Tr+0.97*(1-Tr)
# light shafts (coarse)
S6=6;rr=np.arange(H//S6)*S6+3;cc=np.arange(W//S6)*S6+3
dX,dY,dZ,dep=DX[rr][:,cc],DY[rr][:,cc],DZ[rr][:,cc],np.minimum(DEP[rr][:,cc],2*R)
cg=dX*sun[0]+dY*sun[1]+dZ*sun[2]; phase=(1-0.45**2)/(1+0.45**2-2*0.45*cg)**1.5
Sc=np.zeros(dX.shape,np.float32);NS=80;tmax=dep.max()
for k in range(NS):
    tt=(k+0.5)/NS*min(tmax,2*R)
    Lk=lit(cx+tt*dX,cy+tt*dY,cz+tt*dZ).reshape(dX.shape)
    Sc+=(tt<dep)*Lk*np.exp(-tt/HAZE)*phase*(min(tmax,2*R)/NS)
Sc=np.asarray(Image.fromarray(Sc).resize((W,H),Image.BILINEAR))
col=col*(1-0.05*np.clip(Sc,0,8))+0.05*np.clip(Sc,0,8)*0.98
img=np.clip(col,0,1)**0.85
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
