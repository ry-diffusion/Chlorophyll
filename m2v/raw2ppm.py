# m2v dump (.raw: "M2VD" w h fmt row, texels) -> .ppm, plus stats; formats 80/81 BGRA8, 70/71 RGBA8, 115 RGBA16F
import struct, sys, array
def half(h):
    s=(h>>15)&1; e=(h>>10)&31; m=h&1023
    if e==0: v=m/1024*2**-14
    elif e==31: v=float('inf') if m==0 else float('nan')
    else: v=(1+m/1024)*2**(e-15)
    return -v if s else v
for p in sys.argv[1:]:
    d=open(p,'rb').read()
    mag,w,h,fmt,row=struct.unpack('<5I',d[:20]); px=d[20:]
    out=bytearray(); stats=[]
    for y in range(h):
        for x in range(w):
            o=y*row
            if fmt in (80,81): b,g,r,a=px[o+4*x:o+4*x+4]; c=(r,g,b,a)
            elif fmt in (70,71): c=tuple(px[o+4*x:o+4*x+4])
            elif fmt==115:
                hv=struct.unpack('<4H',px[o+8*x:o+8*x+8]); f=[half(v) for v in hv]
                stats.append(f); c=tuple(max(0,min(255,int((v if v==v else 0)*255))) for v in f)
            else: c=(255,0,255,255)
            out+=bytes(c[:3])
    open(p[:-4]+'.ppm','wb').write(b'P6\n%d %d\n255\n'%(w,h)+out)
    # mean colour of a few spots
    def at(x,y):
        i=(y*w+x)*3; return tuple(out[i:i+3])
    print(p.split('/')[-1], w,h,'fmt',fmt,'centre',at(w//2,h//2),'tl',at(w//8,h//8),'br',at(7*w//8,7*h//8))
    if stats:
        flat=[v for s in stats for v in s[:3]]
        print('   rgb16f min %.3f max %.3f nan %d inf %d'%(min(v for v in flat if v==v), max(v for v in flat if v==v and v!=float('inf')), sum(1 for v in flat if v!=v), sum(1 for v in flat if v==float('inf'))))
