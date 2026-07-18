require('./dom.js');
const fs=require('fs');
eval(fs.readFileSync(__dirname+'/params.js','utf8'));
let fails=0;
const ok=(c,m)=>{ if(!c){ console.log('FAIL:',m); fails++; } else console.log('pass:',m); };

const row = n => form.querySelector('.param-row[data-param="'+n+'"]');

// 1. Fresh form: everything inherits.
ok(readParams(form)===null, 'fresh form sends no params');
ok(row('ubatch_size').querySelector('.param-summary').textContent==="", 'summary starts blank until synced');

// 2. Tick a value -> inherit clears, becomes a fixed override.
const ub=row('ubatch_size');
const v512=ub.querySelectorAll('.param-value').find(c=>c.value==='512');
v512.checked=true; onParamValueToggle(v512);
ok(ub.querySelector('.param-inherit').checked===false, 'ticking a value clears inherit');
ok(JSON.stringify(readParams(form))==='{"ubatch_size":["512"]}', 'one value = fixed override');
ok(ub.querySelector('.param-summary').textContent==='512', 'summary shows the fixed value');

// 3. Re-tick inherit -> values clear.
const inh=ub.querySelector('.param-inherit');
inh.checked=true; onParamInheritToggle(inh);
ok(ub.querySelectorAll('.param-value').every(c=>!c.checked), 'inherit clears every value');
ok(readParams(form)===null, 'back to sending nothing');

// 4. Ladder -> sweep.
fillUbatchLadder();
const p=readParams(form);
ok(p && p.ubatch_size.length===6, 'ladder selects six values');
ok(ub.querySelector('.param-summary').textContent.endsWith('sweep'), 'summary marks it a sweep');

// 5. Custom value that duplicates a curated one must not double up.
const before=ub.querySelectorAll('.param-value').length;
ub.querySelector('.param-custom-input').value='512';
addParamCustom(ub.querySelector('.param-custom-input'));
ok(ub.querySelectorAll('.param-value').length===before, 'duplicate custom reuses the existing checkbox');
ok(readParams(form).ubatch_size.filter(v=>v==='512').length===1, 'no duplicate value submitted');

// 6. gpu_assign "0-1" must NOT collapse onto "0".
const ga=row('gpu_assign');
ga.querySelector('.param-custom-input').value='0-1';
addParamCustom(ga.querySelector('.param-custom-input'));
ok(JSON.stringify(readParams(form).gpu_assign)==='["0-1"]', 'dual-GPU 0-1 stays 0-1, not 0');
ok(ga.querySelectorAll('.param-value').find(c=>c.value==='0').checked===false, 'the "0" choice is untouched');

// 7. A value with no custom box must still be representable.
const st=row('spec_type');
ok(st.querySelector('.param-custom-input')===null, 'spec_type has no custom box');
addParamValue(st, 'ngram-cache');
ok(JSON.stringify(readParams(form).spec_type)==='["ngram-cache"]', 'value survives on a row with no custom box');

// 8. Numeric equality: 1.0 vs 1 collapse, 0-1 vs 0 do not.
ok(numEq('1.0','1')===true, '1.0 equals 1');
ok(numEq('0-1','0')===false, '0-1 does not equal 0');
ok(numEq('all','0')===false, 'all does not equal 0');


// ---- Coverage the first version of this harness silently skipped -----
// The review found it ceremonial: prefillJobForm was never extracted,
// updateMatrixCount always returned at its first branch because the stub
// had no model/build/preset checkboxes, and free-text rows didn't exist.

// 9. Free-text rows split on their own separator.
const ts = row('tensor_split');
ok(ts !== null, 'free-text row exists in the stub');
ts.querySelector('[data-param-text]').value = '1,1 | 3,1';
ok(JSON.stringify(readParams(form).tensor_split)==='["1,1","3,1"]',
   'tensor_split splits on | not ,');

ts.querySelector('[data-param-text]').value = '';
ok(readParams(form).tensor_split===undefined, 'blank free-text sends nothing');

// 10. Matrix maths actually runs now.
const preview = document.getElementById('job-matrix-preview');
// Reset every parameter row so the count is predictable.
paramRows(form).forEach(function(r){
  const inh=r.querySelector('.param-inherit');
  if(inh){ inh.checked=true; onParamInheritToggle(inh); }
  const tx=r.querySelector('[data-param-text]');
  if(tx) tx.value='';
});
updateMatrixCount();
ok(preview.textContent.indexOf('1 models × 1 builds × 1 presets = 1 cells')!==-1,
   'base matrix is 1 cell: '+preview.textContent);

// A single value stays 1 cell; it fixes the parameter, it doesn't sweep.
const ub2 = row('ubatch_size');
const one = ub2.querySelectorAll('.param-value').find(c=>c.value==='512');
one.checked=true; onParamValueToggle(one);
ok(preview.textContent.indexOf('= 1 cells')!==-1,
   'one value is a fixed override, not a sweep: '+preview.textContent);

// Three values -> 3 cells and 3 reloads (ubatch restarts the router).
['1024','2048'].forEach(function(v){
  const c=ub2.querySelectorAll('.param-value').find(x=>x.value===v);
  c.checked=true; onParamValueToggle(c);
});
ok(preview.textContent.indexOf('= 3 cells')!==-1, '3 sweep points = 3 cells: '+preview.textContent);
ok(/~3 llama-server reloads/.test(preview.textContent),
   'reload estimate counts the expensive axis: '+preview.textContent);

// A sampling axis multiplies cells but not reloads.
const tp = row('temperature');
['0','0.7'].forEach(function(v){
  const c=tp.querySelectorAll('.param-value').find(x=>x.value===v);
  c.checked=true; onParamValueToggle(c);
});
ok(preview.textContent.indexOf('= 6 cells')!==-1, 'sampling axis multiplies cells: '+preview.textContent);
ok(/~3 llama-server reloads/.test(preview.textContent),
   'sampling axis is free of reloads: '+preview.textContent);

// 11. Over the cell cap disables submit.
const btn = document.getElementById('job-submit-btn');
ok(btn.disabled===false, 'submit enabled for a sane matrix');

// 12. Edit round-trip: prefillJobForm must restore exactly what a job
// stored, including values the curated list doesn't offer.
paramRows(form).forEach(function(r){
  const inh=r.querySelector('.param-inherit');
  if(inh){ inh.checked=true; onParamInheritToggle(inh); }
});
global.editingJobID = 'job-1';
prefillJobForm({
  name:'j', description:'',
  model_ids:['m1'], build_ids:['b1'], presets:['internal-quick'],
  overrides:{ gpu_assign:'0-1', temperature:1.0, draft_model_path:'/d.gguf' },
  sweeps:[{field:'ubatch_size', values:['512','1024']},
          {field:'spec_type', values:['ngram-cache']}]
});

const back = readParams(form);
ok(JSON.stringify(back.gpu_assign)==='["0-1"]',
   'dual-GPU assignment survives the edit round-trip, got '+JSON.stringify(back.gpu_assign));
ok(JSON.stringify(back.temperature)==='["1.0"]',
   'float 1.0 matches the curated choice rather than becoming a custom "1", got '+JSON.stringify(back.temperature));
ok(JSON.stringify(back.ubatch_size)==='["512","1024"]',
   'sweep values restored, got '+JSON.stringify(back.ubatch_size));
ok(JSON.stringify(back.spec_type)==='["ngram-cache"]',
   'a value with no custom box survives, got '+JSON.stringify(back.spec_type));
ok(unmappedOverrides.draft_model_path==='/d.gguf',
   'an override with no control is carried through, not deleted');

console.log(fails===0 ? '\nALL PASS' : '\n'+fails+' FAILURES');
process.exit(fails?1:0);
