// Minimal DOM good enough to drive the parameter controls, so the
// lifecycle can be exercised without a browser.
class El {
  constructor(tag){ this.tag=tag; this.children=[]; this.attrs={}; this.parent=null;
    this.className=''; this.checked=false; this.value=''; this.type=''; this.textContent=''; }
  appendChild(c){ c.parent=this; this.children.push(c); return c; }
  before(n){ const i=this.parent.children.indexOf(this); this.parent.children.splice(i,0,n); n.parent=this.parent; }
  setAttribute(k,v){ this.attrs[k]=String(v); }
  getAttribute(k){ return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k){ delete this.attrs[k]; }
  closest(sel){ let n=this; while(n){ if(n.matches(sel)) return n; n=n.parent; } return null; }
  matches(sel){
    // :checked pseudo-class
    if(sel.endsWith(':checked')){
      return this.checked && this.matches(sel.slice(0,-':checked'.length));
    }
    // Parse: optional tag, optional .class, optional [attr] / [attr="v"]
    const m=sel.match(/^([\w-]+)?(?:\.([\w-]+))?(?:\[([^=\]]+)(?:="?([^"\]]*)"?)?\])?$/);
    if(!m) throw new Error('harness cannot parse selector: '+sel);
    const [,tag,cls,attr,val]=m;
    if(tag && this.tag!==tag) return false;
    if(cls && !(' '+this.className+' ').includes(' '+cls+' ')) return false;
    if(attr){
      const got=this.getAttribute(attr);
      if(val===undefined ? got===null : got!==val) return false;
    }
    return !!(tag||cls||attr);
  }
  walk(){ let out=[]; for(const c of this.children){ out.push(c); out=out.concat(c.walk()); } return out; }
  querySelectorAll(sel){ const r=this.walk().filter(n=>n.matches(sel)); r.forEach=Array.prototype.forEach.bind(r); return r; }
  querySelector(sel){ return this.querySelectorAll(sel)[0] || null; }
}
function mkRow(param, choices, opts){
  opts = opts || {};
  const row=new El('div'); row.className='param-row';
  row.setAttribute('data-param', param);
  row.setAttribute('data-sep', opts.sep || ',');
  if(opts.restarts) row.setAttribute('data-restarts','1');
  const menu=new El('details'); menu.className='param-menu'; row.appendChild(menu);
  const summary=new El('span'); summary.className='param-summary';
  summary.setAttribute('data-summary-for', param); menu.appendChild(summary);
  const optsEl=new El('div'); optsEl.className='param-options'; menu.appendChild(optsEl);
  const inh=new El('input'); inh.type='checkbox'; inh.className='param-inherit'; inh.checked=true;
  inh.setAttribute('data-param-inherit', param); optsEl.appendChild(inh);
  for(const c of choices){
    const cb=new El('input'); cb.type='checkbox'; cb.className='param-value'; cb.value=c;
    cb.setAttribute('data-param-value', param); optsEl.appendChild(cb);
  }
  if(opts.custom){
    const cwrap=new El('div'); cwrap.className='param-custom'; optsEl.appendChild(cwrap);
    const ci=new El('input'); ci.className='param-custom-input'; cwrap.appendChild(ci);
  }
  return row;
}
function mkCheck(name, value, checked){
  const cb=new El('input'); cb.type='checkbox'; cb.className='matrix-pick';
  cb.setAttribute('name', name); cb.value=value; cb.checked=!!checked;
  return cb;
}

// Free-text parameter row (tensor_split): a plain input, no dropdown.
function mkTextRow(param, sep){
  const row=new El('div'); row.className='param-row';
  row.setAttribute('data-param', param);
  row.setAttribute('data-sep', sep || ',');
  row.setAttribute('data-restarts','1');
  const input=new El('input'); input.className='param-text';
  input.setAttribute('data-param-text', param);
  row.appendChild(input);
  return row;
}

const form=new El('form');
form.setAttribute('data-max-cells','500');
// Matrix selections, without which updateMatrixCount bails immediately.
form.appendChild(mkCheck('model','m1',true));
form.appendChild(mkCheck('model','m2',false));
form.appendChild(mkCheck('build','b1',true));
form.appendChild(mkCheck('preset','internal-quick',true));
form.appendChild(mkRow('ubatch_size',['64','128','256','512','1024','2048'],{restarts:1,custom:1}));
form.appendChild(mkRow('gpu_assign',['all','0','1','0-1'],{restarts:1,custom:1}));
form.appendChild(mkRow('spec_type',['draft','draft-mtp'],{restarts:1}));         // no custom box
form.appendChild(mkRow('temperature',['0','0.7','1.0'],{custom:1}));
form.appendChild(mkTextRow('tensor_split','|'));
const _byId={};
global.document={
  getElementById:(id)=>{ if(id==='new-job-form') return form;
    if(!_byId[id]) _byId[id]=new El('div'); return _byId[id]; },
  createElement:(t)=>new El(t),
  createTextNode:(t)=>{ const n=new El('#text'); n.textContent=t; return n; } };
// form.elements: named-control access, as browsers expose it.
Object.defineProperty(form, 'elements', {
  get(){
    const map={};
    for(const n of form.walk()){
      const nm=n.getAttribute('name');
      if(nm && !(nm in map)) map[nm]=n;
    }
    return map;
  }
});
// Named inputs prefillJobForm writes directly.
['name','description'].forEach(function(n){
  const el=new El('input'); el.setAttribute('name', n); form.appendChild(el);
});

global.form=form;
