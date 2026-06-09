"""make_lottie.py — author the honest hero Lottie (flywheel spins, meter plateaus).
Authored for lottie-web embedding in the paper HTML (inline values, no player slots).
Follows the Skottie shape rules (gr-wrapped groups, tr transforms, 0-1 RGBA, array keyframes)."""
import json, math, os

W, H, FR, OP = 800, 450, 60, 360
BLUE=[0.184,0.427,0.941,1]; GREEN=[0.173,0.627,0.420,1]; AMBER=[0.910,0.639,0.239,1]
RED=[0.878,0.337,0.247,1]; TRACK=[0.875,0.902,0.933,1]; BG=[0.973,0.984,0.992,1]; INK=[0.082,0.125,0.169,1]

def tr(p=(0,0), a=(0,0), s=(100,100), r=0, o=100):
    return {"ty":"tr","p":{"a":0,"k":list(p)},"a":{"a":0,"k":list(a)},
            "s":{"a":0,"k":list(s)},"r":{"a":0,"k":r},"o":{"a":0,"k":o}}
def ell(p,s,color=None,stroke=None,sw=8):
    it=[{"ty":"el","p":{"a":0,"k":list(p)},"s":{"a":0,"k":list(s)}}]
    if color is not None: it.append({"ty":"fl","c":{"a":0,"k":color},"o":{"a":0,"k":100}})
    if stroke is not None: it.append({"ty":"st","c":{"a":0,"k":stroke},"o":{"a":0,"k":100},"w":{"a":0,"k":sw},"lc":2,"lj":2})
    it.append(tr()); return {"ty":"gr","it":it}
def rect(p,s,color,rad=0):
    return {"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":list(p)},"s":{"a":0,"k":list(s)},"r":{"a":0,"k":rad}},
            {"ty":"fl","c":{"a":0,"k":color},"o":{"a":0,"k":100}},tr()]}
def layer(nm, shapes, ks=None, op=OP):
    base={"o":{"a":0,"k":100},"p":{"a":0,"k":[W/2,H/2,0]},"a":{"a":0,"k":[0,0,0]},
          "s":{"a":0,"k":[100,100,100]},"r":{"a":0,"k":0}}
    if ks: base.update(ks)
    return {"ty":4,"nm":nm,"ip":0,"op":op,"st":0,"ks":base,"shapes":shapes}

# ── flywheel: ring + 3 stage-dots, drawn around (0,0), layer rotates 0->360 (loop) ──
CX, CY, RAD = 225, 225, 95
fw_shapes=[ell((0,0),(2*RAD,2*RAD),stroke=BLUE,sw=9)]
for i,col in enumerate([BLUE,GREEN,AMBER]):
    ang=math.radians(i*120)
    fw_shapes.append(ell((RAD*math.cos(ang),RAD*math.sin(ang)),(26,26),color=col))
fw_shapes.append(ell((0,0),(30,30),color=BLUE))  # hub
flywheel=layer("flywheel", fw_shapes, ks={
    "p":{"a":0,"k":[CX,CY,0]},
    "r":{"a":1,"k":[{"t":0,"s":[0],"i":{"x":[1],"y":[1]},"o":{"x":[0],"y":[0]}},{"t":OP,"s":[360]}]}})

# ── meter: track + fill (rises to 58% then PLATEAUS) + dashed-ish target line ──
# track spans y=80..380 (300px = 100%). fill bottom fixed at 380.
def h_of(pct): return 300*pct/100.0
def py_of(h): return 380 - h/2.0
FILLX=610; FW=58
fill_kf_s=[]; fill_kf_p=[]
seq=[(0,0),(40,51),(120,58),(OP,58)]  # frame, percent  -> fill to 58, then flat
for idx,(t,pct) in enumerate(seq):
    h=h_of(pct)
    s_kf={"t":t,"s":[FW,h]}; p_kf={"t":t,"s":[FILLX,py_of(h),0] if False else [FILLX,py_of(h)]}
    if idx<len(seq)-1:
        s_kf.update({"i":{"x":[0.6],"y":[1]},"o":{"x":[0.4],"y":[0]}})
        p_kf.update({"i":{"x":[0.6],"y":[1]},"o":{"x":[0.4],"y":[0]}})
    fill_kf_s.append(s_kf); fill_kf_p.append(p_kf)
meter_fill=layer("meter_fill",[{"ty":"gr","it":[
    {"ty":"rc","p":{"a":1,"k":fill_kf_p},"s":{"a":1,"k":fill_kf_s},"r":{"a":0,"k":8}},
    {"ty":"fl","c":{"a":0,"k":AMBER},"o":{"a":0,"k":100}},tr()]}],
    ks={"p":{"a":0,"k":[0,0,0]}})
meter_track=layer("meter_track",[rect((FILLX,230),(70,300),TRACK,rad=12)], ks={"p":{"a":0,"k":[0,0,0]}})
# target line at 65% -> y = 380 - 195 = 185 (three dashes, honest "never reached")
target=layer("target",[rect((FILLX-22,185),(20,4),RED,rad=2),rect((FILLX+2,185),(20,4),RED,rad=2),
                       rect((FILLX+26,185),(12,4),RED,rad=2)], ks={"p":{"a":0,"k":[0,0,0]}})

bg=layer("background",[rect((W/2,H/2),(W,H),BG)])

doc={"v":"5.7.0","fr":FR,"ip":0,"op":OP,"w":W,"h":H,"assets":[],
     "layers":[target,meter_fill,meter_track,flywheel,bg]}

out=os.path.join(os.path.dirname(__file__),"figures","flywheel_hero.json")
json.dump(doc, open(out,"w"))
json.loads(open(out).read())  # validate parse
print("wrote + validated", out, "(", os.path.getsize(out), "bytes )")
