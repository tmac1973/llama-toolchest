// Minimal DOM stub shared by the tests in web/jstest, good enough to run
// the page scripts without a browser.
//
//   const { El, makeDocument, buildJobForm } = require('./dom.js');
//
// El is an element; makeDocument() sets global.document to a document
// whose body is an El; buildJobForm() builds the benchmark job form that
// the parameter controls work on and makes it global.form.
class El {
  constructor(tag){ this.tag=tag||'div'; this.children=[]; this.attrs={}; this.parent=null;
    this.id=''; this.className=''; this.checked=false; this.value=''; this.type=''; this.textContent='';
    this.hidden=false; this.dataset={}; this.style={}; this.listeners={}; this.isConnected=true;
    this.rect={top:0, bottom:100}; this.scrolled=null;
    this.scrollTop=0; this.scrollHeight=0; this.clientHeight=0; }
  get tagName(){ return this.tag.toUpperCase(); }
  get parentNode(){ return this.parent; }
  get parentElement(){ return this.parent; }
  get firstChild(){ return this.children[0] || null; }
  get nextSibling(){ if(!this.parent) return null; const s=this.parent.children; return s[s.indexOf(this)+1] || null; }
  // detach takes the element out of its parent's children, as a browser
  // does before it inserts a node somewhere else.
  detach(){ if(this.parent){ const s=this.parent.children, i=s.indexOf(this); if(i>=0) s.splice(i,1); } this.parent=null; }
  appendChild(c){ c.detach(); c.parent=this; this.children.push(c); return c; }
  insertBefore(n, ref){ n.detach(); n.parent=this; const i=ref?this.children.indexOf(ref):-1;
    if(i<0) this.children.push(n); else this.children.splice(i,0,n); return n; }
  removeChild(c){ if(c.parent!==this) throw new Error('not a child'); c.detach(); return c; }
  remove(){ this.detach(); }
  before(n){ n.detach(); const i=this.parent.children.indexOf(this); this.parent.children.splice(i,0,n); n.parent=this.parent; }
  contains(n){ while(n){ if(n===this) return true; n=n.parent; } return false; }
  setAttribute(k,v){ this.attrs[k]=String(v); }
  getAttribute(k){ return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k){ delete this.attrs[k]; }
  addEventListener(t,f){ (this.listeners[t]=this.listeners[t]||[]).push(f); }
  // dispatch calls the listeners for type with ev (default: an event with
  // only a type and a target).
  dispatch(type, ev){ ev=ev||{type, target:this}; (this.listeners[type]||[]).forEach(f=>f(ev)); }
  click(){ this.dispatch('click'); }
  select(){}
  // Layout: the test sets rect; scrollIntoView records its options.
  getBoundingClientRect(){ return this.rect; }
  scrollIntoView(o){ this.scrolled=o; }
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


// A spec_type row as the job form renders it: each mode choice carries a
// data-mode checkbox and an expandable settings box, and a draft choice's
// box also carries the n-gram assist dropdown with every assist mode's
// settings, rendered hidden and revealed by that dropdown.
function mkSpecRow(){
  const row=new El('div'); row.className='param-row';
  row.setAttribute('data-param','spec_type');
  row.setAttribute('data-sep',';');
  row.setAttribute('data-restarts','1');
  const menu=new El('details'); menu.className='param-menu'; row.appendChild(menu);
  const summary=new El('span'); summary.className='param-summary';
  summary.setAttribute('data-summary-for','spec_type'); menu.appendChild(summary);
  const optsEl=new El('div'); optsEl.className='param-options'; menu.appendChild(optsEl);
  const inh=new El('input'); inh.type='checkbox'; inh.className='param-inherit'; inh.checked=true;
  inh.setAttribute('data-param-inherit','spec_type'); optsEl.appendChild(inh);

  const assistParams={'ngram-mod':[['assist_n_max','64'],['assist_n_min','48'],['assist_n_match','24']],
                      'ngram-simple':[['assist_size_n','12'],['assist_size_m','48'],['assist_min_hits','1']],
                      'ngram-cache':[]};
  const modes=[['none','',[],null],
               ['draft-mtp','draft-mtp',[['draft_max','6'],['draft_min','0']],'draft'],
               ['ngram-mod:assist_n_max=64,assist_n_min=48,assist_n_match=24','ngram-mod',
                [['assist_n_max','64'],['assist_n_min','48'],['assist_n_match','24']],'assist'],
               ['ngram-cache','ngram-cache',[],'assist']];

  for(const [value,mode,params,slot] of modes){
    const cb=new El('input'); cb.type='checkbox'; cb.className='param-value'; cb.value=value;
    cb.setAttribute('data-param-value','spec_type');
    if(mode) cb.setAttribute('data-mode',mode);
    optsEl.appendChild(cb);
    if(!mode) continue;
    const box=new El('details'); box.className='spec-params';
    box.setAttribute('data-mode',mode); optsEl.appendChild(box);
    for(const [key,def] of params){
      const inp=new El('input'); inp.className='spec-param-input';
      inp.setAttribute('data-key',key); inp.value=def; box.appendChild(inp);
    }
    if(slot!=='draft') continue;
    const sel=new El('select'); sel.className='spec-assist-select'; sel.value='';
    box.appendChild(sel);
    for(const am of Object.keys(assistParams)){
      const d=new El('div'); d.className='spec-assist-params';
      d.setAttribute('data-assist',am); d.hidden=true; box.appendChild(d);
      for(const [key,def] of assistParams[am]){
        const inp=new El('input'); inp.className='spec-param-input';
        inp.setAttribute('data-key',key); inp.value=def; d.appendChild(inp);
      }
    }
  }
  return row;
}

// makeDocument sets global.document to a stub document and returns it.
// Its body is an El, and getElementById and querySelector search the
// body. fire(type, ev) calls the listeners added with addEventListener.
// A test can replace any member it needs to answer differently.
function makeDocument(){
  const body=new El('body');
  const listeners={};
  const doc={
    body, listeners,
    addEventListener(t,f){ (listeners[t]=listeners[t]||[]).push(f); },
    fire(type, ev){ (listeners[type]||[]).forEach(f=>f(ev||{type})); },
    createElement:(t)=>new El(t),
    createTextNode:(t)=>{ const n=new El('#text'); n.textContent=t; return n; },
    getElementById(id){ return body.walk().find(n=>n.id===id || n.getAttribute('id')===id) || null; },
    querySelector(sel){ return body.querySelector(sel); },
    querySelectorAll(sel){ return body.querySelectorAll(sel); },
  };
  global.document=doc;
  return doc;
}

// buildJobForm builds the benchmark job form for the parameter controls,
// sets global.form to it and global.document to a document that returns
// it for "new-job-form" (and a new div for any other id). Returns the form.
function buildJobForm(){
  const form=new El('form');
  form.setAttribute('data-max-cells','500');
  // Matrix selections, without which updateMatrixCount bails immediately.
  form.appendChild(mkCheck('model','m1',true));
  form.appendChild(mkCheck('model','m2',false));
  form.appendChild(mkCheck('build','b1',true));
  form.appendChild(mkCheck('preset','internal-quick',true));
  form.appendChild(mkRow('ubatch_size',['64','128','256','512','1024','2048'],{restarts:1,custom:1}));
  form.appendChild(mkRow('gpu_assign',['all','0','1','0-1'],{restarts:1,custom:1}));
  form.appendChild(mkSpecRow());                                                   // no custom box
  form.appendChild(mkRow('temperature',['0','0.7','1.0'],{custom:1}));
  form.appendChild(mkTextRow('tensor_split','|'));
  const doc=makeDocument();
  const byId={};
  doc.getElementById=(id)=>{ if(id==='new-job-form') return form;
    if(!byId[id]) byId[id]=new El('div'); return byId[id]; };
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
  return form;
}

module.exports={ El, makeDocument, buildJobForm };
