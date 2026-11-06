// Drives the benchmark job form's parameter controls (extracted from
// benchmarks.html into params.js) against the job form in dom.js. Run by
// internal/api/js_params_test.go.
const assert = require('assert');
const fs = require('fs');
const { buildJobForm } = require('./dom.js');

const form = buildJobForm();
eval(fs.readFileSync(__dirname + '/params.js', 'utf8'));

const row = n => form.querySelector('.param-row[data-param="'+n+'"]');

// 1. Fresh form: everything inherits.
assert.strictEqual(readParams(form), null, 'fresh form sends no params');
assert.strictEqual(row('ubatch_size').querySelector('.param-summary').textContent, "", 'summary starts blank until synced');

// 2. Tick a value -> inherit clears, becomes a fixed override.
const ub=row('ubatch_size');
const v512=ub.querySelectorAll('.param-value').find(c=>c.value==='512');
v512.checked=true; onParamValueToggle(v512);
assert.strictEqual(ub.querySelector('.param-inherit').checked, false, 'ticking a value clears inherit');
assert.strictEqual(JSON.stringify(readParams(form)), '{"ubatch_size":["512"]}', 'one value = fixed override');
assert.strictEqual(ub.querySelector('.param-summary').textContent, '512', 'summary shows the fixed value');

// 3. Re-tick inherit -> values clear.
const inh=ub.querySelector('.param-inherit');
inh.checked=true; onParamInheritToggle(inh);
assert.ok(ub.querySelectorAll('.param-value').every(c=>!c.checked), 'inherit clears every value');
assert.strictEqual(readParams(form), null, 'back to sending nothing');

// 4. Ladder -> sweep.
fillUbatchLadder();
const p=readParams(form);
assert.strictEqual(p.ubatch_size.length, 6, 'ladder selects six values');
assert.ok(ub.querySelector('.param-summary').textContent.endsWith('sweep'), 'summary marks it a sweep');

// 5. Custom value that duplicates a curated one must not double up.
const before=ub.querySelectorAll('.param-value').length;
ub.querySelector('.param-custom-input').value='512';
addParamCustom(ub.querySelector('.param-custom-input'));
assert.strictEqual(ub.querySelectorAll('.param-value').length, before, 'duplicate custom reuses the existing checkbox');
assert.strictEqual(readParams(form).ubatch_size.filter(v=>v==='512').length, 1, 'no duplicate value submitted');

// 6. gpu_assign "0-1" must NOT collapse onto "0".
const ga=row('gpu_assign');
ga.querySelector('.param-custom-input').value='0-1';
addParamCustom(ga.querySelector('.param-custom-input'));
assert.strictEqual(JSON.stringify(readParams(form).gpu_assign), '["0-1"]', 'dual-GPU 0-1 stays 0-1, not 0');
assert.strictEqual(ga.querySelectorAll('.param-value').find(c=>c.value==='0').checked, false, 'the "0" choice is untouched');

// 7. A value with no custom box must still be representable.
const st=row('spec_type');
assert.strictEqual(st.querySelector('.param-custom-input'), null, 'spec_type has no custom box');
addParamValue(st, 'ngram-cache');
assert.strictEqual(JSON.stringify(readParams(form).spec_type), '["ngram-cache"]', 'value survives on a row with no custom box');

// 8. Numeric equality: 1.0 vs 1 collapse, 0-1 vs 0 do not.
assert.strictEqual(numEq('1.0','1'), true, '1.0 equals 1');
assert.strictEqual(numEq('0-1','0'), false, '0-1 does not equal 0');
assert.strictEqual(numEq('all','0'), false, 'all does not equal 0');


// ---- Coverage the first version of this harness silently skipped -----
// The review found it ceremonial: prefillJobForm was never extracted,
// updateMatrixCount always returned at its first branch because the stub
// had no model/build/preset checkboxes, and free-text rows didn't exist.

// 9. Free-text rows split on their own separator.
const ts = row('tensor_split');
assert.notStrictEqual(ts, null, 'free-text row exists in the stub');
ts.querySelector('[data-param-text]').value = '1,1 | 3,1';
assert.strictEqual(JSON.stringify(readParams(form).tensor_split), '["1,1","3,1"]',
   'tensor_split splits on | not ,');

ts.querySelector('[data-param-text]').value = '';
assert.strictEqual(readParams(form).tensor_split, undefined, 'blank free-text sends nothing');

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
assert.ok(preview.textContent.indexOf('1 models × 1 builds × 1 presets = 1 cells')!==-1,
   'base matrix is 1 cell: '+preview.textContent);

// A single value stays 1 cell; it fixes the parameter, it doesn't sweep.
const ub2 = row('ubatch_size');
const one = ub2.querySelectorAll('.param-value').find(c=>c.value==='512');
one.checked=true; onParamValueToggle(one);
assert.ok(preview.textContent.indexOf('= 1 cells')!==-1,
   'one value is a fixed override, not a sweep: '+preview.textContent);

// Three values -> 3 cells and 3 reloads (ubatch restarts the router).
['1024','2048'].forEach(function(v){
  const c=ub2.querySelectorAll('.param-value').find(x=>x.value===v);
  c.checked=true; onParamValueToggle(c);
});
assert.ok(preview.textContent.indexOf('= 3 cells')!==-1, '3 sweep points = 3 cells: '+preview.textContent);
assert.ok(/~3 llama-server reloads/.test(preview.textContent),
   'reload estimate counts the expensive axis: '+preview.textContent);

// A sampling axis multiplies cells but not reloads.
const tp = row('temperature');
['0','0.7'].forEach(function(v){
  const c=tp.querySelectorAll('.param-value').find(x=>x.value===v);
  c.checked=true; onParamValueToggle(c);
});
assert.ok(preview.textContent.indexOf('= 6 cells')!==-1, 'sampling axis multiplies cells: '+preview.textContent);
assert.ok(/~3 llama-server reloads/.test(preview.textContent),
   'sampling axis is free of reloads: '+preview.textContent);

// 11. Over the cell cap disables submit.
const btn = document.getElementById('job-submit-btn');
assert.strictEqual(btn.disabled, false, 'submit enabled for a sane matrix');

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
assert.strictEqual(JSON.stringify(back.gpu_assign), '["0-1"]',
   'dual-GPU assignment survives the edit round-trip, got '+JSON.stringify(back.gpu_assign));
assert.strictEqual(JSON.stringify(back.temperature), '["1.0"]',
   'float 1.0 matches the curated choice rather than becoming a custom "1", got '+JSON.stringify(back.temperature));
assert.strictEqual(JSON.stringify(back.ubatch_size), '["512","1024"]',
   'sweep values restored, got '+JSON.stringify(back.ubatch_size));
assert.strictEqual(JSON.stringify(back.spec_type), '["ngram-cache"]',
   'a value with no custom box survives, got '+JSON.stringify(back.spec_type));
assert.strictEqual(unmappedOverrides.draft_model_path, '/d.gguf',
   'an override with no control is carried through, not deleted');

// 12. Speculative decoding: a draft method and an n-gram assist combine
// inside one checkbox, which is the only way all twenty-five pairings are
// reachable on a row that has no custom-entry box.
const sp=row('spec_type');
sp.querySelector('.param-inherit').checked=true;
sp.querySelectorAll('.param-value').forEach(c=>{ c.checked=false; });

const mtpBox=sp.querySelectorAll('.spec-params').find(b=>b.getAttribute('data-mode')==='draft-mtp');
const mtpSel=mtpBox.querySelector('.spec-assist-select');
const mtpMax=mtpBox.querySelectorAll('.spec-param-input').find(i=>i.getAttribute('data-key')==='draft_max');

// Editing a setting selects the mode, and encodes only that slot.
mtpMax.value='3'; updateSpecValue(mtpMax);
assert.strictEqual(JSON.stringify(readParams(form).spec_type), '["draft-mtp:draft_max=3,draft_min=0"]',
   'draft slot alone encodes without a "+", got '+JSON.stringify(readParams(form).spec_type));

// Choosing an assist appends it and reveals only that mode's settings.
mtpSel.value='ngram-mod'; updateSpecValue(mtpSel);
assert.strictEqual(readParams(form).spec_type[0], 'draft-mtp+ngram-mod:draft_max=3,draft_min=0,assist_n_max=64,assist_n_min=48,assist_n_match=24',
   'assist joins with "+" and contributes its own keys, got '+JSON.stringify(readParams(form).spec_type));
assert.strictEqual(mtpBox.querySelectorAll('.spec-assist-params').find(d=>d.getAttribute('data-assist')==='ngram-simple').hidden, true,
   'a non-selected assist stays hidden so its defaults never reach the value');

// Switching the assist swaps which keys appear.
mtpSel.value='ngram-simple'; updateSpecValue(mtpSel);
assert.ok(readParams(form).spec_type[0].indexOf('assist_size_n=12')!==-1 &&
   readParams(form).spec_type[0].indexOf('assist_n_max')===-1,
   'switching the assist swaps its settings, got '+JSON.stringify(readParams(form).spec_type));

// Back to None drops the segment and every assist key.
mtpSel.value=''; updateSpecValue(mtpSel);
assert.strictEqual(readParams(form).spec_type[0], 'draft-mtp:draft_max=3,draft_min=0',
   'clearing the assist drops the "+" segment, got '+JSON.stringify(readParams(form).spec_type));

// 13. A saved combined value restores onto the checkbox, the dropdown and
// both slots' inputs.
sp.querySelectorAll('.param-value').forEach(c=>{ c.checked=false; });
addParamValue(sp, 'draft-mtp+ngram-mod:draft_max=5,assist_n_max=32');
assert.strictEqual(readParams(form).spec_type[0], 'draft-mtp+ngram-mod:draft_max=5,assist_n_max=32',
   'a combined value round-trips, got '+JSON.stringify(readParams(form).spec_type));
assert.strictEqual(mtpSel.value, 'ngram-mod', 'restore selects the assist dropdown, got '+mtpSel.value);
assert.strictEqual(mtpMax.value, '5', 'restore fills the draft input, got '+mtpMax.value);
const modBox=mtpBox.querySelectorAll('.spec-assist-params').find(d=>d.getAttribute('data-assist')==='ngram-mod');
assert.strictEqual(modBox.querySelectorAll('.spec-param-input').find(i=>i.getAttribute('data-key')==='assist_n_max').value, '32',
   'restore fills the assist input');
assert.strictEqual(modBox.querySelectorAll('.spec-param-input').find(i=>i.getAttribute('data-key')==='assist_n_min').value, '',
   'a key the saved value omits is cleared, not left at its default');

console.log('ALL PASS');
