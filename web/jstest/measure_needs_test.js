// What the Measure dialog says beside each video size: the measured peak,
// a prediction, or that the size will not fit, from the server's needs
// entries (sizeNeed in gen_instances.go).
const assert = require('assert');
const fs = require('fs');
eval(fs.readFileSync(__dirname + '/measure.js', 'utf8'));

assert.ok(!genSizeNeedText({}).text, 'nothing known: nothing shown');

let t = genSizeNeedText({gib: 23.6, measured: true, fit: 'fits'});
assert.strictEqual(t.text, 'measured 23.6 GiB: fits', 'a measured size that fits');
assert.ok(!t.over, 'a measured size that fits');

t = genSizeNeedText({gib: 29.4, measured: true, fit: 'over'});
assert.strictEqual(t.text, 'needed more than 29.4 GiB: will not fit', 'a measured size stopped for going past the room');
assert.ok(t.over, 'a measured size stopped for going past the room');

t = genSizeNeedText({gib: 26.4, fit: 'close'});
assert.strictEqual(t.text, 'about 26.4 GiB predicted: close to the limit, measure to be sure', 'a prediction near the limit');
assert.ok(t.title.indexOf('Only Measure grants a size') >= 0, 'a prediction says it grants nothing');

t = genSizeNeedText({gib: 30.1, fit: 'maybe'});
assert.strictEqual(t.text, 'about 30.1 GiB predicted: may not fit, measure to check', 'a prediction past the limit only says it may not fit');
assert.ok(!t.over, 'a prediction past the limit only says it may not fit');

t = genSizeNeedText({fit: 'over'});
assert.strictEqual(t.text, 'will not fit', 'above a size already too big, with no prediction');
assert.ok(t.over, 'above a size already too big, with no prediction');

console.log('ALL PASS');
