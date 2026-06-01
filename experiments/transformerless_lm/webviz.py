#!/usr/bin/env python3
"""webviz.py — VISUALIZE the web. You can't render 110M edges (no browser can; it'd be a hairball). So this
draws an explorable NEIGHBORHOOD: BFS from a seed concept, top-PMI grounded links per node, capped — plus a
scale panel with the TRUE totals so the size is felt. Emits ONE self-contained graph.html (embedded data +
canvas force-graph, zero deps, opens offline). Click a node to expand it; node color = source field; edge
thickness = co-occurrence weight.

Uses indexed `edges WHERE a=?` lookups (cheap even mid-fold, unlike GROUP BY). Knowledge edges only
(translation/parallel/memory excluded), stopwords dropped.

  python webviz.py gravity --depth 2 --per 10
  python webviz.py substrate --depth 2          # visualize the web's knowledge of its own design
"""
import sys, math, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kdb import KnowledgeDB, load_embedding

DB = Path(__file__).parent / "knowledge.db"
OUT = Path(__file__).parent / "graph.html"


def build(seed, depth=2, per=10, cap=320):
    s, E, n, e = load_embedding()
    k = KnowledgeDB(str(DB), s, E, n, e)
    seed = seed.lower()
    if seed not in k.stoi:
        print(f"'{seed}' not a node"); return None
    nodes = {seed: {"id": seed, "field": "seed", "depth": 0}}
    edges = []
    frontier = [seed]
    for d in range(1, depth + 1):
        nxt = []
        for u in frontier:
            rows = k.db.execute(
                "SELECT b,src,w FROM edges WHERE a=? AND src NOT LIKE 'align-%' AND src NOT LIKE 'parallel-%' "
                "AND src!='memory' ORDER BY w DESC LIMIT 200", (u,)).fetchall()
            ranked = sorted(((k.assoc(u, b, w) * math.log1p(w), w, b, src)
                             for b, src, w in rows if b not in k.stop and b != u), reverse=True)[:per]
            for sc, w, b, src in ranked:
                if len(nodes) >= cap and b not in nodes:
                    continue
                if b not in nodes:
                    nodes[b] = {"id": b, "field": src, "depth": d}
                    nxt.append(b)
                edges.append({"s": u, "t": b, "w": int(w)})
        frontier = nxt
        if len(nodes) >= cap:
            break
    p, ne, f = k.stats()
    return {"seed": seed, "nodes": list(nodes.values()), "edges": edges,
            "totals": {"passages": p, "edges": ne, "fields": f, "nodes": len(k.nodes)}}


HTML = """<!doctype html><html><head><meta charset=utf-8><title>OMC web — {seed}</title>
<style>
 html,body{margin:0;background:#0a0e14;color:#cdd9e5;font:13px ui-monospace,monospace;overflow:hidden}
 #c{display:block} #hud{position:fixed;top:0;left:0;padding:12px 16px;pointer-events:none}
 #hud b{color:#7ee787;font-size:15px} .k{color:#79c0ff} .dim{color:#768390}
 #tip{position:fixed;background:#161b22;border:1px solid #30363d;padding:4px 8px;border-radius:4px;display:none;pointer-events:none}
</style></head><body>
<canvas id=c></canvas>
<div id=hud><b>THE WEB</b> · seed: <span class=k>{seed}</span><br>
<span class=dim>showing</span> <span id=shown></span> <span class=dim>of</span>
<span class=k>{passages:,} passages · {edges:,} edges · {nodes:,} nodes · {fields} fields</span><br>
<span class=dim>click a node to expand · drag to pan · scroll to zoom</span></div>
<div id=tip></div>
<script>
const DATA={data};
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
let W,H;function size(){W=cv.width=innerWidth;H=cv.height=innerHeight;}size();onresize=size;
const fields=[...new Set(DATA.nodes.map(n=>n.field))];
const hue=f=>f==='seed'?140:(fields.indexOf(f)*47)%360;
let N=DATA.nodes.map(n=>({...n,x:W/2+(Math.random()-.5)*400,y:H/2+(Math.random()-.5)*400,vx:0,vy:0}));
const idx=()=>Object.fromEntries(N.map((n,i)=>[n.id,i]));
let L=DATA.edges.map(e=>({s:e.s,t:e.t,w:e.w}));
document.getElementById('shown').textContent=N.length+' nodes, '+L.length+' edges';
let view={x:0,y:0,z:1},drag=null;
function step(){const I=idx();
 for(const n of N){n.vx*=.85;n.vy*=.85;}
 for(let i=0;i<N.length;i++)for(let j=i+1;j<N.length;j++){
   let a=N[i],b=N[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.hypot(dx,dy)||1,f=900/(d*d);
   a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f;}
 for(const e of L){let a=N[I[e.s]],b=N[I[e.t]];if(!a||!b)continue;
   let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-90)*.01;
   a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f;}
 for(const n of N){n.x+=n.vx;n.y+=n.vy;}}
function draw(){ctx.setTransform(view.z,0,0,view.z,view.x,view.y);ctx.clearRect(-view.x/view.z,-view.y/view.z,W/view.z,H/view.z);
 const I=idx();
 for(const e of L){let a=N[I[e.s]],b=N[I[e.t]];if(!a||!b)continue;
   ctx.strokeStyle='rgba(120,150,180,'+Math.min(.5,.08+Math.log(e.w+1)/12)+')';
   ctx.lineWidth=Math.min(3,.4+Math.log(e.w+1)/3);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
 for(const n of N){let r=n.depth===0?9:Math.max(3,6-n.depth);
   ctx.fillStyle='hsl('+hue(n.field)+',70%,'+(n.depth===0?65:55)+'%)';
   ctx.beginPath();ctx.arc(n.x,n.y,r,0,7);ctx.fill();
   if(view.z>.7||n.depth===0){ctx.fillStyle='#cdd9e5';ctx.font=(n.depth===0?13:10)+'px monospace';ctx.fillText(n.id,n.x+r+2,n.y+3);}}}
function loop(){for(let i=0;i<2;i++)step();draw();requestAnimationFrame(loop);}loop();
cv.onmousedown=e=>drag={x:e.clientX-view.x,y:e.clientY-view.y};
onmouseup=()=>drag=null;
cv.onmousemove=e=>{if(drag){view.x=e.clientX-drag.x;view.y=e.clientY-drag.y;}
 const mx=(e.clientX-view.x)/view.z,my=(e.clientY-view.y)/view.z;let hit=null;
 for(const n of N){if(Math.hypot(n.x-mx,n.y-my)<8){hit=n;break;}}
 if(hit){tip.style.display='block';tip.style.left=e.clientX+8+'px';tip.style.top=e.clientY+8+'px';
   tip.textContent=hit.id+'  ['+hit.field+']';}else tip.style.display='none';};
cv.onwheel=e=>{e.preventDefault();let f=e.deltaY<0?1.1:.9;view.z*=f;};
</script></body></html>"""


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    args = sys.argv[1:]; depth, per = 2, 10
    for flag, var in (("--depth", "depth"), ("--per", "per")):
        if flag in args:
            i = args.index(flag);
            if var == "depth": depth = int(args[i+1])
            else: per = int(args[i+1])
            args = args[:i] + args[i+2:]
    g = build(" ".join(args), depth, per)
    if not g:
        return
    t = g["totals"]
    out_html = (HTML
                .replace("{seed}", html.escape(g["seed"]))
                .replace("{data}", json.dumps(g))
                .replace("{passages:,}", f"{t['passages']:,}")
                .replace("{edges:,}", f"{t['edges']:,}")
                .replace("{nodes:,}", f"{t['nodes']:,}")
                .replace("{fields}", str(t["fields"])))
    OUT.write_text(out_html)
    print(f"[webviz] {len(g['nodes'])} nodes, {len(g['edges'])} edges → {OUT}")
    print(f"[webviz] open in browser: file://{OUT}")
    print(f"[webviz] (scale panel shows the true {g['totals']['passages']:,} passages / {g['totals']['edges']:,} edges)")


if __name__ == "__main__":
    main()
