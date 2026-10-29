// What the Measure dialog says beside each video size: the measured peak,
// a prediction, or that the size will not fit, from the server's needs
// entries (sizeNeed in gen_instances.go).
const fs = require('fs');
eval(fs.readFileSync(__dirname + '/measure.js', 'utf8'));

let fails = 0;
const ok = (c, m) => { if (!c) { console.log('FAIL:', m); fails++; } else console.log('pass:', m); };

ok(!genSizeNeedText({}).text, 'nothing known: nothing shown');

let t = genSizeNeedText({gib: 23.6, measured: true, fit: 'fits'});
ok(t.text === 'measured 23.6 GiB: fits' && !t.over, 'a measured size that fits');

t = genSizeNeedText({gib: 29.4, measured: true, fit: 'over'});
ok(t.text === 'needed more than 29.4 GiB: will not fit' && t.over, 'a measured size stopped for going past the room');

t = genSizeNeedText({gib: 26.4, fit: 'close'});
ok(t.text === 'about 26.4 GiB predicted: close to the limit, measure to be sure', 'a prediction near the limit');
ok(t.title.indexOf('Only Measure grants a size') >= 0, 'a prediction says it grants nothing');

t = genSizeNeedText({fit: 'over'});
ok(t.text === 'will not fit' && t.over, 'above a size already too big, with no prediction');

console.log(fails === 0 ? 'ALL PASS' : fails + ' FAILURES');
process.exit(fails === 0 ? 0 : 1);
