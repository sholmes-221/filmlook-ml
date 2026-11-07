// node tests/js/parity_runner.js <dir>  : reads *.rgba (+ .json {w,h}) and prints JS features as JSON lines
'use strict';
const fs = require('fs'), path = require('path');
const F = require('../../js/filmlook-style/features.js');
const dir = process.argv[2];
for (const name of fs.readdirSync(dir).filter(n => n.endsWith('.rgba')).sort()) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, name.replace('.rgba', '.json'))));
  const data = new Uint8ClampedArray(fs.readFileSync(path.join(dir, name)));
  const t0 = process.hrtime.bigint();
  const f = F.styleFeatures({ width: meta.w, height: meta.h, data });
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(JSON.stringify({ name, ms, f }));
}
