'use strict';
const $ = id => document.getElementById(id);
const token = location.hash.slice(1);
let doc, pageId, currentSymbol, selectedId = null, mode = 'select';
let view = { z: 1, x: 0, y: 0 }, gesture = null, space = false;
let dirty = false, editVersion = 0, saving = null, saveTimer, imageUrl, loadIndex = 0;
let undoStack = [], redoStack = [];
const page = () => doc.pages.find(p => p.id === pageId);
const mark = () => page().annotations.find(a => a.id === selectedId);
const symbol = id => doc.symbols.find(s => s.id === id);
const clone = value => JSON.parse(JSON.stringify(value));
function message(text) { $('message').textContent = text; $('message').hidden = !text; }
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Authorization: 'Bearer ' + token, ...(body === undefined ? {} : {'Content-Type': 'application/json'}) },
    ...(body === undefined ? {} : {body: JSON.stringify(body)})
  });
  if (!response.ok) {
    let error; try { error = await response.json(); } catch { error = {}; }
    throw new Error(error.error || `Ошибка ${response.status}`);
  }
  return response;
}
function snapshot() { return JSON.stringify(doc); }
function checkpoint(before = snapshot()) {
  undoStack.push(before); if (undoStack.length > 60) undoStack.shift(); redoStack = [];
}
function touch() {
  dirty = true; editVersion++; $('save-status').textContent = 'Есть изменения…';
  clearTimeout(saveTimer); saveTimer = setTimeout(() => save().catch(e => message(e.message)), 900);
}
async function save() {
  clearTimeout(saveTimer);
  if (saving) { await saving; if (dirty) return save(); return; }
  if (!dirty) return;
  const version = editVersion, payload = clone(doc);
  $('save-status').textContent = 'Сохранение…';
  saving = (async () => {
    try {
      const result = await (await api('/api/save', payload)).json();
      doc.revision = result.revision;
      if (version === editVersion) { dirty = false; $('save-status').textContent = 'Сохранено на компьютере'; }
    } catch (e) { $('save-status').textContent = 'Не сохранено'; throw e; }
    finally { saving = null; }
  })();
  await saving;
  if (dirty) return save();
}
function invalidate(...ids) { page().reviewed_symbols = page().reviewed_symbols.filter(s => !ids.includes(s)); }
function change(action, affected = []) {
  checkpoint(); action(); invalidate(...affected); touch(); render();
}
function button(text, action, className = '') {
  const node = document.createElement('button'); node.textContent = text;
  node.className = className; node.onclick = action; return node;
}
function relevantSymbols() {
  const ids = new Set([...page().annotations.map(a => a.symbol), ...page().reviewed_symbols]);
  if (currentSymbol) ids.add(currentSymbol);
  for (const s of doc.symbols) if(s.id.startsWith("custom_")) ids.add(s.id);
  return doc.symbols.filter(s => ids.has(s.id));
}
function renderPages() {
  $('page-count').textContent = doc.pages.length; $('pages').replaceChildren();
  for (const p of doc.pages) {
    const row = button(p.file.split('/').at(-1), () => switchPage(p.id), 'page-row' + (p.id === pageId ? ' active' : ''));
    row.title = p.file;
    const small = document.createElement('small');
    small.textContent = `Лист ${p.page} · ${p.annotations.length} рамок · ${p.reviewed_symbols.length} классов проверено`;
    row.append(small); $('pages').append(row);
  }
}
function renderSymbols() {
  $('symbols').replaceChildren();
  const query = $('search').value.toLocaleLowerCase();
  for (const s of relevantSymbols()) {
    if (!s.name.toLocaleLowerCase().includes(query) && !s.id.toLocaleLowerCase().includes(query)) continue;
    const row = document.createElement('div'); row.className = 'symbol-row' + (s.id === currentSymbol ? ' active' : '');
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = s.enabled;
    check.setAttribute('aria-label', `Исследовать ${s.name}`);
    check.onchange = () => change(() => { s.enabled = check.checked; });
    const prefix = page().file.split('/').at(-1).replace(/\.pdf$/i,'')+' · ';
    const displayName = s.name.startsWith(prefix) ? s.name.slice(prefix.length) : s.name;
    const label = button(displayName, () => { currentSymbol = s.id; selectedId = null; render(); }); label.title = s.name;
    const count = document.createElement('small'); count.textContent = page().annotations.filter(a => a.symbol === s.id).length;
    row.append(check, label, count); $('symbols').append(row);
  }
}
function renderDetails() {
  const s = symbol(currentSymbol), a = mark();
  $('symbol-name').value = s?.name || ''; $('symbol-name').disabled = !s;
  $('rename-symbol').disabled = !s;
  $('reviewed').checked = page().reviewed_symbols.includes(currentSymbol); $('reviewed').disabled = !s;
  $('empty-selection').hidden = !!a; $('details').hidden = !a;
  if (a) {
    $('mark-symbol').replaceChildren();
    for (const entry of relevantSymbols()) {
      const option = document.createElement('option'); option.value = entry.id; option.textContent = entry.name;
      $('mark-symbol').append(option);
    }
    $('mark-symbol').value = a.symbol;
    for (const k of ['x','y','w','h']) $('box-' + k).value = Number(a.bbox[k].toFixed(3));
    $('legend').checked = a.legend; $('template').checked = a.template;
  }
  const marks = page().annotations.filter(a => a.symbol === currentSymbol);
  $('mark-count').textContent = marks.length; $('marks').replaceChildren();
  marks.forEach((a, i) => {
    const label = `${i+1}. ${a.legend ? 'Легенда' : 'На плане'}${a.template ? ' · образец' : ''}`;
    const row = button(label, () => { selectedId = a.id; focusBox(a.bbox); render(); }, a.id === selectedId ? 'active' : '');
    $('marks').append(row);
  });
  $('undo').disabled = !undoStack.length; $('redo').disabled = !redoStack.length;
}
function svgElement(name, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function drawBoxes() {
  const svg = $('boxes'); svg.replaceChildren();
  for (const a of page().annotations) {
    const box = gesture?.id === a.id && gesture.preview ? gesture.preview : a.bbox;
    const chosen = a.id === selectedId, active = a.symbol === currentSymbol;
    const color = chosen ? '#1d4ed8' : a.template ? '#7c3aed' : a.legend ? '#cf8617' : active ? '#087e6b' : '#8292a6';
    svg.append(svgElement('rect', {x:box.x,y:box.y,width:box.w,height:box.h,
      fill:chosen?'#285bba22':'transparent',stroke:color,'stroke-width':(chosen?2.5:1.2)/view.z,
      'vector-effect':'non-scaling-stroke','data-id':a.id}));
    if (chosen) {
      const size = 7/view.z;
      for (const [handle,x,y] of [['nw',box.x,box.y],['ne',box.x+box.w,box.y],['sw',box.x,box.y+box.h],['se',box.x+box.w,box.y+box.h]]) {
        svg.append(svgElement('rect',{x:x-size/2,y:y-size/2,width:size,height:size,fill:'#fff',stroke:color,
          'stroke-width':1/view.z,'vector-effect':'non-scaling-stroke','data-id':a.id,'data-handle':handle}));
      }
    }
  }
  if (gesture?.type === 'draw' && gesture.preview) {
    const b=gesture.preview; svg.append(svgElement('rect',{x:b.x,y:b.y,width:b.w,height:b.h,
      fill:'#285bba22',stroke:'#285bba','stroke-width':2/view.z,'vector-effect':'non-scaling-stroke'}));
  }
}
function positionStage() {
  $('stage').style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.z})`;
  $('zoom-label').textContent = Math.round(view.z*100)+'%'; drawBoxes();
}
function fit() {
  const rect = $('viewport').getBoundingClientRect(), p=page();
  view.z = Math.min(rect.width/p.width,rect.height/p.height)*0.96;
  view.x = (rect.width-p.width*view.z)/2; view.y=(rect.height-p.height*view.z)/2; positionStage();
}
function focusBox(b) {
  const rect=$('viewport').getBoundingClientRect();
  view.z=Math.min(20,rect.width/(b.w*5),rect.height/(b.h*5));
  view.x=rect.width/2-(b.x+b.w/2)*view.z; view.y=rect.height/2-(b.y+b.h/2)*view.z; positionStage();
}
function zoom(factor, x, y) {
  const next=Math.max(0.03,Math.min(25,view.z*factor)), ratio=next/view.z;
  view.x=x-(x-view.x)*ratio; view.y=y-(y-view.y)*ratio; view.z=next; positionStage();
}
function render() { renderPages(); renderSymbols(); renderDetails(); drawBoxes(); }
async function loadImage(reset = true) {
  const index=++loadIndex; $('loading').hidden=false;
  try {
    const response=await api(`/api/image?page=${encodeURIComponent(pageId)}&dpi=${$('dpi').value}`);
    const blob=await response.blob(); if(index!==loadIndex)return;
    const url=URL.createObjectURL(blob); if(imageUrl)URL.revokeObjectURL(imageUrl); imageUrl=url;
    const p=page(); $('stage').style.width=p.width+'px'; $('stage').style.height=p.height+'px';
    $('boxes').setAttribute('viewBox',`0 0 ${p.width} ${p.height}`);
    $('page-image').src=url;
    $('page-info').textContent=`${p.file.split('/').at(-1)} · ${Math.round(p.width)} × ${Math.round(p.height)} пунктов PDF`;
    if(reset)fit(); else positionStage(); $('loading').hidden=true;
  } catch(e) { if(index===loadIndex){message(e.message);$('loading').textContent='Не удалось подготовить страницу';} }
}
function switchPage(id) {
  pageId=id; selectedId=null; gesture=null;
  currentSymbol=page().annotations[0]?.symbol || page().reviewed_symbols[0] || null;
  $('search').value=''; render(); loadImage();
}
function setMode(value) {
  mode=value;
  document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===value));
  $('viewport').style.cursor=value==='pan'?'grab':value==='draw'?'crosshair':'default';
}
function point(e) {
  const r=$('viewport').getBoundingClientRect();
  return {x:(e.clientX-r.left-view.x)/view.z,y:(e.clientY-r.top-view.y)/view.z};
}
function clampPoint(p) { return {x:Math.max(0,Math.min(page().width,p.x)),y:Math.max(0,Math.min(page().height,p.y))}; }
function rectangle(a,b) { return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),w:Math.abs(b.x-a.x),h:Math.abs(b.y-a.y)}; }
$('viewport').addEventListener('pointerdown',e=>{
  if(!doc || !$('loading').hidden)return;
  if(e.button!==0&&e.button!==1)return;
  $('viewport').focus(); const start=point(e), id=e.target.dataset.id;
  if(mode==='pan'||space||e.button===1)gesture={type:'pan',x:e.clientX,y:e.clientY,v:{...view}};
  else if(mode==='draw') {
    if(!currentSymbol){message('Сначала выберите или создайте класс ИУО');return;}
    gesture={type:'draw',start:clampPoint(start),before:snapshot()};
  } else if(id) {
    selectedId=id; currentSymbol=mark().symbol;
    gesture={type:e.target.dataset.handle?'resize':'move',handle:e.target.dataset.handle,
      id,start,box:clone(mark().bbox),before:snapshot()}; render();
  } else {selectedId=null;render();return;}
  $('viewport').setPointerCapture(e.pointerId);e.preventDefault();
});
$('viewport').addEventListener('pointermove',e=>{
  if(!gesture)return; const g=gesture,p=clampPoint(point(e));
  if(g.type==='pan'){view.x=g.v.x+e.clientX-g.x;view.y=g.v.y+e.clientY-g.y;positionStage();return;}
  if(g.type==='draw')g.preview=rectangle(g.start,p);
  else if(g.type==='move')g.preview={...g.box,
    x:Math.max(0,Math.min(page().width-g.box.w,g.box.x+p.x-g.start.x)),
    y:Math.max(0,Math.min(page().height-g.box.h,g.box.y+p.y-g.start.y))};
  else {
    const b=g.box,opposite={x:g.handle.includes('w')?b.x+b.w:b.x,y:g.handle.includes('n')?b.y+b.h:b.y};
    g.preview=rectangle(opposite,p);
  }
  drawBoxes();
});
$('viewport').addEventListener('pointerup',()=>{
  if(!gesture)return;const g=gesture;gesture=null;
  if(g.preview&&g.preview.w>=0.2&&g.preview.h>=0.2){
    checkpoint(g.before);
    if(g.type==='draw'){
      const a={id:'manual_'+crypto.randomUUID(),symbol:currentSymbol,bbox:g.preview,legend:false,template:false,source:{}};
      page().annotations.push(a);selectedId=a.id;
    }else mark().bbox=g.preview;
    invalidate(currentSymbol);touch();
  }
  render();
});
$('viewport').addEventListener('pointercancel',()=>{gesture=null;drawBoxes();});
$('viewport').addEventListener('wheel',e=>{
  e.preventDefault();if(gesture)return;const r=$('viewport').getBoundingClientRect();
  zoom(Math.exp(-e.deltaY*0.002),e.clientX-r.left,e.clientY-r.top);
},{passive:false});
function deleteMark(){const a=mark();if(a)change(()=>{page().annotations=page().annotations.filter(v=>v.id!==a.id);selectedId=null;},[a.symbol]);}
function history(back) {
  const source=back?undoStack:redoStack,target=back?redoStack:undoStack;if(!source.length)return;
  target.push(snapshot());const revision=doc.revision;doc=JSON.parse(source.pop());doc.revision=revision;
  selectedId=null;touch();render();
}
$('save').onclick=()=>save().catch(e=>message(e.message));
$('download').onclick=()=>{
  const blob=new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;a.download='workspace_copy.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('export').onclick=async()=>{
  try {await save();const result=await(await api('/api/export',{page_id:pageId})).json();
    $('export-path').value=result.path;$('export-warnings').textContent=result.warnings.join('\n');$('export-dialog').showModal();
  }catch(e){message(e.message);}
};
$('close-dialog').onclick=()=>$('export-dialog').close();
$('search').oninput=renderSymbols;
$('new-symbol').onclick=()=>{
  const name=prompt('Название нового класса ИУО:');if(!name?.trim())return;
  if(name.trim().length>200){message('Название должно быть короче 200 символов');return;}
  change(()=>{const s={id:'custom_'+crypto.randomUUID(),name:name.trim(),enabled:false};doc.symbols.push(s);currentSymbol=s.id;selectedId=null;});
};
$('rename-symbol').onclick=()=>{
  const name=$('symbol-name').value.trim();if(!name||!symbol(currentSymbol))return;
  change(()=>{symbol(currentSymbol).name=name;});
};
$('reviewed').onchange=()=>change(()=>{
  page().reviewed_symbols=page().reviewed_symbols.filter(s=>s!==currentSymbol);
  if($('reviewed').checked)page().reviewed_symbols.push(currentSymbol);
});
$('mark-symbol').onchange=()=>{const a=mark();if(!a)return;const old=a.symbol,next=$('mark-symbol').value;
  change(()=>{a.symbol=next;currentSymbol=next;if(a.template)symbol(next).enabled=true;},[old,next]);};
$('legend').onchange=()=>{const a=mark();if(a)change(()=>{a.legend=$('legend').checked;},[a.symbol]);};
$('template').onchange=()=>{const a=mark();if(a)change(()=>{a.template=$('template').checked;if(a.template)symbol(a.symbol).enabled=true;});};
$('apply-box').onclick=()=>{
  const a=mark();if(!a)return;const b=Object.fromEntries(['x','y','w','h'].map(k=>[k,Number($('box-'+k).value)]));
  if(Object.values(b).some(v=>!Number.isFinite(v))||b.x<0||b.y<0||b.w<=0||b.h<=0||b.x+b.w>page().width+0.01||b.y+b.h>page().height+0.01){message('Рамка должна находиться внутри страницы и иметь положительный размер');return;}
  change(()=>{a.bbox=b;},[a.symbol]);
};
$('delete').onclick=deleteMark;$('undo').onclick=()=>history(true);$('redo').onclick=()=>history(false);
$('fit').onclick=fit;$('dpi').onchange=()=>loadImage(false);
for(const [id,factor] of [['zoom-in',1.25],['zoom-out',0.8]])$(id).onclick=()=>{const r=$('viewport').getBoundingClientRect();zoom(factor,r.width/2,r.height/2);};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
window.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save().catch(err=>message(err.message));return;}
  if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();history(!e.shiftKey);return;}
  if(e.code==='Space'){space=true;e.preventDefault();}
  if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();deleteMark();}
  if(e.key.toLowerCase()==='v')setMode('select');if(e.key.toLowerCase()==='r')setMode('draw');if(e.key.toLowerCase()==='h')setMode('pan');
  if(e.key==='Escape'){gesture=null;selectedId=null;render();}
});
window.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});
window.addEventListener('blur',()=>{space=false;gesture=null;});
window.addEventListener('beforeunload',e=>{if(dirty||saving){e.preventDefault();e.returnValue='';}});
new ResizeObserver(()=>{if(doc)positionStage();}).observe($('viewport'));
(async()=>{
  try{const data=await(await api('/api/state')).json();doc=data.workspace;
    $('save-status').textContent='Сохранено на компьютере';$('warnings').replaceChildren();
    for(const warning of doc.warnings){const li=document.createElement('li');li.textContent=warning;$('warnings').append(li);}
    switchPage(doc.pages[0].id);
  }catch(e){message(e.message);$('loading').textContent='Редактор не открыт';}
})();
