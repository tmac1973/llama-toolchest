// The video upscaler Measure dialog's limit check must agree with the
// server's (checkVUpscaleClip in gen_vupscale_jobs.go): cases.json holds
// each case with whether the server refuses it.
const assert = require('assert');
const fs = require('fs');
eval(fs.readFileSync(__dirname + '/vupscale.js', 'utf8'));

const cases = JSON.parse(fs.readFileSync(__dirname + '/cases.json', 'utf8'));
for (const c of cases) {
    const t = vupscaleClipLimitText(c.script, c.scale, c.w, c.h, c.frames, c.window);
    assert.strictEqual(t !== '', c.refused, JSON.stringify(c) + ': JS says "' + t + '"');
}
console.log('ALL PASS');
