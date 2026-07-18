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

console.log(fails===0 ? '\nALL PASS' : '\n'+fails+' FAILURES');
process.exit(fails?1:0);
