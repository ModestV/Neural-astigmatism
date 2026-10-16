# ruff: noqa: E501
"""HTML-отчет. Задача T4."""

from __future__ import annotations

from html import escape
from pathlib import Path

from symspot.core.types import Drawing, Result
from symspot.render._common import fmt, instance_ids
from symspot.render.palette import color_hex
from symspot.render.svg_overlay import build_svg

_CSS = """
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:13px/1.35 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
color:#222}
#app{display:flex;height:100vh}
#view{flex:1 1 auto;min-width:0;position:relative;background:#d8d8d8}
#map{width:100%;height:100%;display:block;cursor:grab;touch-action:none;user-select:none}
#map.drag{cursor:grabbing}
#view .tools{position:absolute;left:8px;top:8px;display:flex;gap:4px}
#view .tools button{padding:2px 8px}
#side{flex:0 0 560px;max-width:45vw;overflow:auto;border-left:1px solid #bbb;padding:10px 12px;
background:#fafafa}
h1{font-size:16px;margin:0 0 4px}
h2{font-size:13px;margin:14px 0 4px;border-bottom:1px solid #ddd;padding-bottom:2px}
.muted{color:#777}
table{border-collapse:collapse;width:100%}
th,td{padding:2px 5px;text-align:left;border-bottom:1px solid #eee;white-space:nowrap}
th{cursor:pointer;position:sticky;top:0;background:#eee}
td.n,th.n{text-align:right}
tbody tr{cursor:pointer}
tbody tr:hover{background:#eef4ff}
tr.sel{background:#fff3b0 !important}
tr.amb td.st{color:#b35c00;font-weight:600}
tr.amb{background:#fff7ec}
tr.hidden,.inst.hidden{display:none}
.sw{display:inline-block;width:12px;height:12px;border:1px solid #0004;vertical-align:-2px;
margin-right:4px}
.leg label{display:block;cursor:pointer}
.leg .cnt{color:#777}
#amb a{display:block;color:#b35c00;cursor:pointer;text-decoration:none;padding:1px 0}
#amb a:hover{text-decoration:underline}
.inst{pointer-events:visibleStroke}
.inst.amb{filter:none}
body.has-sel .inst:not(.sel){opacity:.3}
#hl{fill:none;stroke:#ff9800;stroke-width:2px;vector-effect:non-scaling-stroke;
stroke-dasharray:6 3;pointer-events:none}
"""

_JS = r"""
(function(){
const svg=document.getElementById('map');
const view=document.getElementById('view');
const base=svg.getAttribute('viewBox').split(/\s+/).map(Number);
let vb=base.slice();
function setVB(){svg.setAttribute('viewBox',vb.map(v=>+v.toFixed(4)).join(' '));}
function aspect(){const r=view.getBoundingClientRect();return Math.max(r.width,1)/Math.max(r.height,1);}
function fit(cx,cy,w,h){
  const a=aspect();
  let W=Math.max(w,h*a), H=W/a;
  vb=[cx-W/2,cy-H/2,W,H];setVB();
}
function reset(){fit(base[2]/2,base[3]/2,base[2],base[3]);}
function scalePx(){const r=svg.getBoundingClientRect();return Math.min(r.width/vb[2],r.height/vb[3]);}
function toUser(ev){
  const r=svg.getBoundingClientRect();const s=scalePx();
  const ox=(r.width-vb[2]*s)/2, oy=(r.height-vb[3]*s)/2;
  return [vb[0]+(ev.clientX-r.left-ox)/s, vb[1]+(ev.clientY-r.top-oy)/s];
}
svg.addEventListener('wheel',function(ev){
  ev.preventDefault();
  const f=Math.pow(1.0018,ev.deltaY);
  const w=vb[2]*f;
  if(w<base[2]*1e-4||w>base[2]*20)return;
  const p=toUser(ev);
  vb=[p[0]-(p[0]-vb[0])*f,p[1]-(p[1]-vb[1])*f,w,vb[3]*f];setVB();
},{passive:false});
let drag=null;
svg.addEventListener('pointerdown',function(ev){
  drag={x:ev.clientX,y:ev.clientY,vb:vb.slice(),moved:false,target:ev.target};
  svg.setPointerCapture(ev.pointerId);
});
svg.addEventListener('pointermove',function(ev){
  if(!drag)return;
  const dx=ev.clientX-drag.x, dy=ev.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>3){drag.moved=true;svg.classList.add('drag');}
  if(!drag.moved)return;
  const s=scalePx();
  vb=[drag.vb[0]-dx/s,drag.vb[1]-dy/s,drag.vb[2],drag.vb[3]];setVB();
});
svg.addEventListener('pointerup',function(ev){
  if(drag&&!drag.moved){
    const g=drag.target.closest&&drag.target.closest('.inst');
    if(g)select(g.dataset.id,false);else select(null,false);
  }
  drag=null;svg.classList.remove('drag');
});

const rows=Array.from(document.querySelectorAll('#tbl tbody tr'));
const insts=Array.from(svg.querySelectorAll('.inst'));
const hlLayer=document.getElementById('hl');
let selId=null;
function boxOf(el){return el.dataset.bbox.split(' ').map(Number);}
function restore(g){g.querySelectorAll('[data-w]').forEach(e=>{e.setAttribute('stroke-width',e.dataset.w);delete e.dataset.w;});
  if(g.dataset.gw){g.setAttribute('stroke-width',g.dataset.gw);delete g.dataset.gw;}}
function boost(g){
  const gw=g.getAttribute('stroke-width');
  if(gw){g.dataset.gw=gw;g.setAttribute('stroke-width',(parseFloat(gw)*2).toFixed(3));}
  g.querySelectorAll('[stroke-width]').forEach(e=>{e.dataset.w=e.getAttribute('stroke-width');
    e.setAttribute('stroke-width',(parseFloat(e.dataset.w)*2).toFixed(3));});
}
function select(id,center){
  if(selId!==null){
    const g=document.getElementById('inst-'+selId);
    if(g){g.classList.remove('sel');restore(g);}
    rows.forEach(r=>r.classList.remove('sel'));
  }
  selId=(id===null||id===undefined)?null:String(id);
  document.body.classList.toggle('has-sel',selId!==null);
  hlLayer.setAttribute('width',0);hlLayer.setAttribute('height',0);
  if(selId===null)return;
  const g=document.getElementById('inst-'+selId);
  if(!g){selId=null;document.body.classList.remove('has-sel');return;}
  g.classList.add('sel');boost(g);
  g.parentNode.appendChild(g);
  const row=rows.find(r=>r.dataset.id===selId);
  if(row){row.classList.add('sel');if(!center)row.scrollIntoView({block:'nearest'});}
  const b=boxOf(g);
  hlLayer.setAttribute('x',b[0]);hlLayer.setAttribute('y',b[1]);
  hlLayer.setAttribute('width',Math.max(b[2]-b[0],0));hlLayer.setAttribute('height',Math.max(b[3]-b[1],0));
  if(center){
    const w=Math.max(b[2]-b[0],base[2]*0.01), h=Math.max(b[3]-b[1],base[2]*0.01);
    fit((b[0]+b[2])/2,(b[1]+b[3])/2,w*4,h*4);
  }
}
rows.forEach(r=>r.addEventListener('click',()=>select(r.dataset.id,true)));
document.querySelectorAll('#amb a').forEach(a=>a.addEventListener('click',()=>select(a.dataset.id,true)));

const hiddenLegends=new Set();
let onlyAmb=false;
function applyFilter(){
  const vis=e=>!hiddenLegends.has(e.dataset.legend)&&(!onlyAmb||e.dataset.status==='ambiguous');
  insts.forEach(g=>g.classList.toggle('hidden',!vis(g)));
  rows.forEach(r=>r.classList.toggle('hidden',!vis(r)));
  document.querySelectorAll('#amb a').forEach(a=>a.style.display=hiddenLegends.has(a.dataset.legend)?'none':'');
}
document.querySelectorAll('.leg input[data-legend]').forEach(cb=>cb.addEventListener('change',()=>{
  if(cb.checked)hiddenLegends.delete(cb.dataset.legend);else hiddenLegends.add(cb.dataset.legend);
  applyFilter();
}));
function setAll(v){document.querySelectorAll('.leg input[data-legend]').forEach(cb=>{
  cb.checked=v;if(v)hiddenLegends.delete(cb.dataset.legend);else hiddenLegends.add(cb.dataset.legend);});applyFilter();}
document.getElementById('all').addEventListener('click',()=>setAll(true));
document.getElementById('none').addEventListener('click',()=>setAll(false));
document.getElementById('onlyamb').addEventListener('change',function(){onlyAmb=this.checked;applyFilter();});
document.getElementById('reset').addEventListener('click',()=>{select(null,false);reset();});
document.getElementById('zin').addEventListener('click',()=>{const c=[vb[0]+vb[2]/2,vb[1]+vb[3]/2];fit(c[0],c[1],vb[2]/1.5,vb[3]/1.5);});
document.getElementById('zout').addEventListener('click',()=>{const c=[vb[0]+vb[2]/2,vb[1]+vb[3]/2];fit(c[0],c[1],vb[2]*1.5,vb[3]*1.5);});

const ths=Array.from(document.querySelectorAll('#tbl th'));
ths.forEach((th,i)=>th.addEventListener('click',()=>{
  const dir=th.dataset.dir==='asc'?'desc':'asc';ths.forEach(t=>delete t.dataset.dir);th.dataset.dir=dir;
  const num=th.classList.contains('n');
  const key=r=>{const v=r.children[i].dataset.v;return num?parseFloat(v):v;};
  const body=document.querySelector('#tbl tbody');
  rows.slice().sort((a,b)=>{const x=key(a),y=key(b);return (x<y?-1:x>y?1:0)*(dir==='asc'?1:-1);})
    .forEach(r=>body.appendChild(r));
}));
window.addEventListener('resize',()=>{});
reset();
})();
"""


def _legend_ids(result: Result) -> list[str]:
    seen: dict[str, None] = {}
    for q in result.queries:
        seen.setdefault(q.legend_id, None)
    for inst in result.instances:
        seen.setdefault(inst.legend_id, None)
    return sorted(seen)


def build_html(drawing: Drawing, result: Result) -> str:
    """Собрать текст HTML-отчета."""
    svg = build_svg(drawing, result, xml_header=False, responsive=True)
    svg = svg.replace("<svg ", '<svg id="map" ', 1)
    # слой подсветки рамки внутри viewport (после экземпляров)
    svg = svg.replace(
        "</g>\n</svg>", '<rect id="hl" x="0" y="0" width="0" height="0"/>\n</g>\n</svg>'
    )

    labels = {q.legend_id: (q.label or "") for q in result.queries}
    ids = instance_ids(result)
    legends = _legend_ids(result)
    counts: dict[str, int] = {k: 0 for k in legends}
    for inst in result.instances:
        counts[inst.legend_id] += 1
    n_amb = sum(1 for i in result.instances if i.status == "ambiguous")

    leg_html = []
    for lid in legends:
        label = labels.get(lid, "")
        title = f"{escape(lid)}" + (f" ({escape(label)})" if label else "")
        leg_html.append(
            f'<label><input type="checkbox" checked data-legend="{escape(lid, quote=True)}"> '
            f'<span class="sw" style="background:{color_hex(lid)}"></span>{title} '
            f'<span class="cnt">{counts[lid]}</span></label>'
        )

    rows = []
    amb_links = []
    for inst, eff in zip(result.instances, ids, strict=True):
        label = labels.get(inst.legend_id, "")
        amb = inst.status == "ambiguous"
        legend_attr = escape(inst.legend_id, quote=True)
        x0, y0, x1, y1 = inst.bbox
        rows.append(
            f'<tr data-id="{eff}" data-legend="{legend_attr}" data-status="{inst.status}"'
            f' data-bbox="{fmt(x0)} {fmt(y0)} {fmt(x1)} {fmt(y1)}"'
            f"{' class=amb' if amb else ''}>"
            f'<td class="n" data-v="{eff}">{eff}</td>'
            f'<td data-v="{legend_attr}"><span class="sw" style="background:'
            f'{color_hex(inst.legend_id)}"></span>{escape(inst.legend_id)}</td>'
            f'<td data-v="{escape(label, quote=True)}">{escape(label)}</td>'
            f'<td class="n" data-v="{inst.score:.6f}">{inst.score:.3f}</td>'
            f'<td class="n" data-v="{inst.coverage:.6f}">{inst.coverage:.3f}</td>'
            f'<td class="st" data-v="{inst.status}">{inst.status}</td>'
            f'<td data-v="{escape(inst.source, quote=True)}">{escape(inst.source)}</td></tr>'
        )
        if amb:
            rivals = f" (конкуренты: {escape(', '.join(inst.rivals))})" if inst.rivals else ""
            amb_links.append(
                f'<a data-id="{eff}" data-legend="{legend_attr}">#{eff} {escape(inst.legend_id)}'
                f", score {inst.score:.3f}{rivals}</a>"
            )

    timings = "".join(
        f"<tr><td>{escape(str(k))}</td><td class='n'>{v:.3f} с</td></tr>"
        for k, v in result.timings.items()
    )
    empty = "" if result.instances else '<p class="muted">Экземпляры не найдены.</p>'
    amb_block = "".join(amb_links) if amb_links else '<span class="muted">нет</span>'
    name = escape(Path(result.drawing).name or result.drawing)
    thead = (
        '<th class="n">id</th><th>legend_id</th><th>label</th><th class="n">score</th>'
        '<th class="n">coverage</th><th>status</th><th>source</th>'
    )
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>SymSpot: {name}</title>
<style>{_CSS}</style></head><body>
<div id="app">
<div id="view">
{svg}
<div class="tools"><button id="zin">+</button><button id="zout">-</button>
<button id="reset">Сброс</button></div>
</div>
<div id="side">
<h1>{name}</h1>
<div class="muted">экземпляров: {len(result.instances)}, неоднозначных: {n_amb},
размер: {fmt(drawing.width, 1)} x {fmt(drawing.height, 1)} ({drawing.kind})</div>
{empty}
<h2>Классы (фильтр)</h2>
<div class="leg">{"".join(leg_html)}
<div><button id="all">Все</button> <button id="none">Ничего</button>
<label><input type="checkbox" id="onlyamb"> только ambiguous</label></div></div>
<h2>Ambiguous ({n_amb})</h2>
<div id="amb">{amb_block}</div>
<h2>Экземпляры</h2>
<table id="tbl"><thead><tr>{thead}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Тайминги</h2>
<table>{timings or '<tr><td class="muted">нет данных</td></tr>'}</table>
</div></div>
<script>{_JS}</script>
</body></html>
"""


def render_html(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Один самодостаточный HTML: встроенный SVG-оверлей с zoom/pan, таблица экземпляров
    (id, класс, score, coverage, status), фильтр по классу, клик по строке -> подсветка и
    центрирование экземпляра, отдельный блок ``ambiguous``, тайминги."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(drawing, result), encoding="utf-8")
    return out
