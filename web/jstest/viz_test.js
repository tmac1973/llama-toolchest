// Which points a chart may draw. Not every run carries every metric: a
// run measured before memory was recorded, or under a router below log
// verbosity 4, has timings and no footprint. Drawing it at zero would
// claim it used no memory, and Plotly's formatter throws on undefined,
// so the page has to leave those runs out.
require('./dom.js');
const fs = require('fs');
eval(fs.readFileSync(__dirname + '/viz.js', 'utf8'));

let fails = 0;
const ok = (c, m) => { if (!c) { console.log('FAIL:', m); fails++; } else console.log('pass:', m); };

DATA = {
    points: [
        { run_id: 'a', metrics: { gen: 40, mem_gpu: 23 } },
        { run_id: 'b', metrics: { gen: 55 } },
        { run_id: 'c', metrics: { gen: 60, mem_gpu: 0 } }
    ]
};

const speed = pointsWith({ key: 'gen' });
ok(speed.length === 3, 'every run has a speed');

const mem = pointsWith({ key: 'mem_gpu' });
ok(mem.length === 2, 'the run with no footprint is left out of a memory chart');
ok(mem.every(p => p.run_id !== 'b'), 'and it is the right one that is left out');

// Zero is a measurement, not an absence — a model whose weights are all
// on the CPU really does use no GPU memory.
ok(mem.some(p => p.run_id === 'c'), 'a measured zero is still plotted');

ok(pointsWith(null).length === 3, 'a chart with no metric keeps every point');

console.log(fails === 0 ? 'ALL PASS' : fails + ' FAILURES');
process.exit(fails === 0 ? 0 : 1);
