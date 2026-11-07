// node js/filmlook-style/bench/bench.js <dir of .rgba/.json 2x-intermediates>
'use strict';
const fs = require('fs'), path = require('path');
const S = require('../dist/filmlook-style.js');
const dir = process.argv[2];
const names = fs.readdirSync(dir).filter(n => n.endsWith('.rgba')).sort();
const imgs = names.map(n => { const m = JSON.parse(fs.readFileSync(path.join(dir, n.replace('.rgba', '.json'))));
  return { width: m.w, height: m.h, data: new Uint8ClampedArray(fs.readFileSync(path.join(dir, n))) }; });
const t = [];
let heapPeak = 0;
const h0 = process.memoryUsage().heapUsed;
for (let rep = 0; rep < 3; rep++) {
  for (const im of imgs) {
    const s = process.hrtime.bigint();
    S.analyzeImageData(S.box2x(im), 'zh');
    t.push(Number(process.hrtime.bigint() - s) / 1e6);
    heapPeak = Math.max(heapPeak, process.memoryUsage().heapUsed - h0);
  }
}
const first = t[0]; t.sort((a, b) => a - b);
console.log(JSON.stringify({ n: t.length, input: `${imgs[0].width}x${imgs[0].height} (2x intermediate)`,
  first_call_ms: +first.toFixed(1), p50_ms: +t[Math.floor(t.length / 2)].toFixed(1),
  p95_ms: +t[Math.floor(t.length * 0.95)].toFixed(1), max_ms: +t[t.length - 1].toFixed(1),
  heap_growth_mb: +(heapPeak / 2 ** 20).toFixed(1), node: process.version }));
