// Run: node --test js/filmlook-style/test/   (Node >= 18). Fixtures: python scripts/style_make_fixtures.py
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const F = require(path.join(ROOT, 'features.js'));
const E = require(path.join(ROOT, 'engine.js'));
const M = require(path.join(ROOT, 'model-data.js'));
const BUNDLE = path.join(ROOT, 'dist', 'filmlook-style.js');
const G = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'golden.json')));

function img(c) {
  return { width: c.w, height: c.h, data: new Uint8ClampedArray(fs.readFileSync(path.join(__dirname, 'fixtures', c.name + '.rgba'))) };
}

function close(a, b, tol, where) {
  if (typeof a === 'number' && typeof b === 'number') {
    assert.ok(Math.abs(a - b) <= tol, `${where}: ${a} vs ${b}`);
  } else if (Array.isArray(a)) {
    assert.strictEqual(a.length, b.length, where);
    a.forEach((x, i) => close(x, b[i], tol, `${where}[${i}]`));
  } else if (a && typeof a === 'object') {
    assert.deepStrictEqual(Object.keys(a).sort(), Object.keys(b).sort(), where);
    for (const k of Object.keys(a)) close(a[k], b[k], tol, `${where}.${k}`);
  } else {
    assert.strictEqual(a, b, where);
  }
}

test('feature parity with Python/OpenCV reference (1e-9)', () => {
  for (const c of G.cases) {
    const f = F.styleFeatures(img(c));
    for (const k of F.FEATURE_NAMES) close(f[k], c.features[k], 1e-9, `${c.name}.${k}`);
  }
});

test('box2x is bit-identical to Python', () => {
  const B = G.box2x;
  const out = F.box2x({ width: B.w, height: B.h, data: new Uint8ClampedArray(fs.readFileSync(path.join(__dirname, 'fixtures', 'box_in.rgba'))) });
  assert.strictEqual(out.width, Math.floor(B.w / 2));
  for (let y = 0; y < out.height; y++) for (let x = 0; x < out.width; x++) for (let c = 0; c < 3; c++)
    assert.strictEqual(out.data[(y * out.width + x) * 4 + c], B.expected[y][x][c]);
});

test('engine output identical to Python reference (en + zh)', () => {
  for (const c of G.cases) {
    for (const lang of ['en', 'zh']) {
      const r = E.analyzeFeatures(c.features, M, lang, F.featureVector);
      close(r, c[lang], 1e-9, `${c.name}.${lang}`);
    }
  }
});

test('learned-model inference parity: linear, forest, mlp', () => {
  for (const kind of ['linear', 'forest', 'mlp']) {
    const L = G.learned[kind];
    G.cases.forEach((c, i) => {
      const p = E.predictLearned(F.featureVector(c.features), L.model);
      close(p, L.expected[i], 1e-9, `${kind}.${c.name}`);
    });
  }
});

test('bundle: single classic script, global API, deterministic', () => {
  const S = require(BUNDLE);
  const c = G.cases[0];
  const a = S.analyzeImageData(img(c), 'en'), b = S.analyzeImageData(img(c), 'en');
  assert.deepStrictEqual(a, b);
  assert.strictEqual(a.matches.length, 3);
  assert.ok(['match', 'uncertain'].includes(a.status));
  assert.match(a.disclaimer, /does not identify the photographer/);
});

test('bundle is ES2017-safe and sandbox-safe (no eval/Function/WASM/network/modern syntax)', () => {
  const raw = fs.readFileSync(BUNDLE, 'utf8');
  const src = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');  // code only, no comments
  const forbidden = [/\?\./, /\?\?/, /\beval\s*\(/, /new\s+Function\b/, /WebAssembly/, /\bfetch\s*\(/,
    /XMLHttpRequest/, /\bimport\s*\(/, /^\s*import\s/m, /^\s*export\s/m, /\bclass\s+\w+\s*\{[^}]*#/, /\.at\(/,
    /\bstructuredClone\b/, /Object\.hasOwn\(/, /\.flatMap\(/, /\.flat\(/, /\.matchAll\(/, /BigInt/];
  for (const re of forbidden) assert.ok(!re.test(src), `forbidden construct ${re}`);
  assert.ok(Buffer.byteLength(raw) < 80 * 1024, 'bundle should stay under 80 KB');
});

test('monochrome images never get colour evidence or colour photographers as reliable matches', () => {
  const w = 200, h = 150, d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { const v = (i * 37) % 256; d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = v; d[4 * i + 3] = 255; }
  for (const lang of ['en', 'zh']) {
    const r = E.analyzeFeatures(F.styleFeatures({ width: w, height: h, data: d }), M, lang, F.featureVector);
    assert.strictEqual(r.monochrome, true);
    const colourWords = lang === 'en' ? /colou?r|palette|warm|cool|saturat/i : /色/;
    for (const m of r.matches) {
      for (const e of m.evidence) assert.ok(!colourWords.test(e), `colour phrase on mono: ${e}`);
      if (m.reliable) assert.ok(['daido_moriyama', 'fan_ho'].includes(m.id), m.id);
    }
  }
});

test('flat / low-information image returns "uncertain"', () => {
  const w = 120, h = 80, d = new Uint8ClampedArray(w * h * 4).fill(128);
  const r = E.analyzeFeatures(F.styleFeatures({ width: w, height: h, data: d }), M, 'en', F.featureVector);
  assert.strictEqual(r.status, 'uncertain');
  assert.ok(r.matches.every(m => !m.reliable));
});

test('rejects bad input', () => {
  assert.throws(() => F.styleFeatures({ width: 8, height: 8, data: new Uint8ClampedArray(256) }));
  assert.throws(() => E.analyzeFeatures(G.cases[0].features, M, 'fr', F.featureVector));
});
