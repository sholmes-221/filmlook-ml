/* FilmLook style features v1: pure JavaScript (ES2017, no dependencies, no eval/WebAssembly).
 * Mirrors src/filmlook_ml/style/features.py exactly (see tests/parity). Input: {width, height, data}
 * where data is RGBA uint8 (Canvas getImageData) with the long side <= WORK_SIDE.
 */
'use strict';

var FEATURE_VERSION = 'style-features-v1';
var WORK_SIDE = 384;
var EPS = 1e-6;
var FEATURE_NAMES = [
  'y_mean', 'y_std', 'p05', 'p25', 'p50', 'p75', 'p95', 'clip_hi', 'clip_lo', 'dark_frac', 'bright_frac',
  'sat_mean', 'sat_std', 'sat_p90', 'sat_p95', 'mono', 'accent_frac', 'muted_frac', 'warmth', 'tint', 'colorfulness',
  'hue_red', 'hue_yellow', 'hue_green', 'hue_cyan', 'hue_blue', 'hue_magenta', 'hue_entropy',
  'grad_mean', 'edge_frac', 'lap_logvar', 'noise', 'hv_balance', 'diag_frac',
  'local_contrast', 'lc_ratio',
  'grid_y_std', 'sat_concentration', 'dark_cells', 'accent_cells', 'centroid_x', 'centroid_y',
  'dark_centroid_x', 'dark_centroid_y', 'center_ratio', 'top_bottom', 'sharp_center_ratio', 'edge_center_ratio'
];

function q8(x) {
  var v = Math.floor(x * 255 + 0.5);
  return v < 0 ? 0 : (v > 255 ? 255 : v);
}

function histPct(h, n, q) {
  var t = q * n, c = 0;
  for (var b = 0; b < 256; b++) { c += h[b]; if (c >= t) return b / 255; }
  return 1;
}

function meanStd(arr, len) {
  var s = 0, i;
  for (i = 0; i < len; i++) s += arr[i];
  var m = s / len, v = 0;
  for (i = 0; i < len; i++) { var d = arr[i] - m; v += d * d; }
  return [m, Math.sqrt(v / len)];
}

/* mean over the (2r+1)^2 window clipped to the image, via an integral image */
function boxMean(img, H, W, r) {
  var W1 = W + 1, ii = new Float64Array((H + 1) * W1), y, x;
  for (y = 0; y < H; y++) {
    var row = 0;
    for (x = 0; x < W; x++) {
      row += img[y * W + x];
      ii[(y + 1) * W1 + x + 1] = ii[y * W1 + x + 1] + row;
    }
  }
  var out = new Float64Array(H * W);
  for (y = 0; y < H; y++) {
    var y0 = Math.max(y - r, 0), y1 = Math.min(y + r + 1, H);
    for (x = 0; x < W; x++) {
      var x0 = Math.max(x - r, 0), x1 = Math.min(x + r + 1, W);
      var s = ii[y1 * W1 + x1] - ii[y0 * W1 + x1] - ii[y1 * W1 + x0] + ii[y0 * W1 + x0];
      out[y * W + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

function styleFeatures(image) {
  var W = image.width, H = image.height, data = image.data;
  if (W < 16 || H < 16) throw new Error('image too small');
  var n = W * H, i, x, y, k;
  var R = new Float64Array(n), G = new Float64Array(n), B = new Float64Array(n), Y = new Float64Array(n);
  var S = new Float64Array(n), MX = new Float64Array(n), D = new Float64Array(n);
  var hy = new Float64Array(256), hs = new Float64Array(256);
  for (i = 0; i < n; i++) {
    var r = data[4 * i] / 255, g = data[4 * i + 1] / 255, b = data[4 * i + 2] / 255;
    R[i] = r; G[i] = g; B[i] = b;
    var yy = 0.299 * r + 0.587 * g + 0.114 * b;
    Y[i] = yy;
    hy[q8(yy)] += 1;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    MX[i] = mx; D[i] = d;
    var s = mx > 0 ? d / mx : 0;
    S[i] = s;
    hs[q8(s)] += 1;
  }
  var f = {};
  // ---------------- tone
  var ms = meanStd(Y, n);
  f.y_mean = ms[0]; f.y_std = ms[1];
  f.p05 = histPct(hy, n, 0.05); f.p25 = histPct(hy, n, 0.25); f.p50 = histPct(hy, n, 0.5);
  f.p75 = histPct(hy, n, 0.75); f.p95 = histPct(hy, n, 0.95);
  var chi = 0, clo = 0, dk = 0, br = 0;
  for (k = 0; k < 256; k++) {
    if (k >= 250) chi += hy[k];
    if (k <= 5) clo += hy[k];
    if (k < 51) dk += hy[k];
    if (k > 204) br += hy[k];
  }
  f.clip_hi = chi / n; f.clip_lo = clo / n; f.dark_frac = dk / n; f.bright_frac = br / n;
  // ---------------- colour
  ms = meanStd(S, n);
  f.sat_mean = ms[0]; f.sat_std = ms[1];
  f.sat_p90 = histPct(hs, n, 0.90); f.sat_p95 = histPct(hs, n, 0.95);
  f.mono = f.sat_p95 < 0.08 ? 1 : 0;
  var acc = 0, mut = 0, warm = 0, tint = 0;
  var RG = new Float64Array(n), YB = new Float64Array(n);
  var hw = [0, 0, 0, 0, 0, 0];
  for (i = 0; i < n; i++) {
    if (S[i] > 0.55 && Y[i] > 0.15 && Y[i] < 0.9) acc++;
    if (S[i] < 0.25) mut++;
    warm += R[i] - B[i];
    tint += G[i] - 0.5 * (R[i] + B[i]);
    RG[i] = R[i] - G[i];
    YB[i] = 0.5 * (R[i] + G[i]) - B[i];
    if (S[i] > 0.15 && MX[i] > 0.1 && D[i] > 0) {
      var dd = D[i], h;
      if (MX[i] === R[i]) { h = (G[i] - B[i]) / dd; if (h < 0) h += 6; }
      else if (MX[i] === G[i]) h = (B[i] - R[i]) / dd + 2;
      else h = (R[i] - G[i]) / dd + 4;
      var sec = Math.floor(h);
      if (sec < 0) sec = 0; if (sec > 5) sec = 5;
      hw[sec] += S[i];
    }
  }
  f.accent_frac = acc / n; f.muted_frac = mut / n; f.warmth = warm / n; f.tint = tint / n;
  var a1 = meanStd(RG, n), a2 = meanStd(YB, n);
  f.colorfulness = Math.sqrt(a1[1] * a1[1] + a2[1] * a2[1]) + 0.3 * Math.sqrt(a1[0] * a1[0] + a2[0] * a2[0]);
  var tot = hw[0] + hw[1] + hw[2] + hw[3] + hw[4] + hw[5];
  var hn = ['hue_red', 'hue_yellow', 'hue_green', 'hue_cyan', 'hue_blue', 'hue_magenta'], ent = 0;
  for (k = 0; k < 6; k++) {
    var p = tot > 0 ? hw[k] / tot : 0;
    f[hn[k]] = p;
    if (p > 0) ent -= p * Math.log(p);
  }
  f.hue_entropy = tot > 0 ? ent / Math.log(6) : 0;
  // ---------------- texture (interior pixels)
  var Hi = H - 2, Wi = W - 2, ni = Hi * Wi;
  var GM = new Float64Array(ni), LAP = new Float64Array(ni), GX = new Float64Array(ni), GY = new Float64Array(ni);
  var EDG = new Uint8Array(ni);
  var gsum = 0, esum = 0;
  for (y = 1; y < H - 1; y++) {
    for (x = 1; x < W - 1; x++) {
      var j = (y - 1) * Wi + (x - 1), c = y * W + x;
      var gx = (Y[c + 1] - Y[c - 1]) / 2, gy = (Y[c + W] - Y[c - W]) / 2;
      var gm = Math.sqrt(gx * gx + gy * gy);
      GX[j] = gx; GY[j] = gy; GM[j] = gm;
      gsum += gm;
      if (gm > 0.08) { EDG[j] = 1; esum++; }
      LAP[j] = Y[c - W] + Y[c + W] + Y[c - 1] + Y[c + 1] - 4 * Y[c];
    }
  }
  f.grad_mean = gsum / ni; f.edge_frac = esum / ni;
  var lv = meanStd(LAP, ni)[1];
  f.lap_logvar = Math.log(lv * lv + 1e-8);
  var sx = 0, sy = 0, diag = 0;
  for (j = 0; j < ni; j++) {
    if (!EDG[j]) continue;
    var ax = Math.abs(GX[j]), ay = Math.abs(GY[j]);
    sx += ax; sy += ay;
    if (Math.min(ax, ay) > 0.5 * Math.max(ax, ay)) diag++;
  }
  f.hv_balance = (sx - sy) / (sx + sy + EPS);
  f.diag_frac = esum > 0 ? diag / esum : 0;
  var b3 = boxMean(Y, H, W, 1);
  var resAll = new Float64Array(ni), resFlat = [];
  for (y = 1; y < H - 1; y++) {
    for (x = 1; x < W - 1; x++) {
      j = (y - 1) * Wi + (x - 1);
      var rr = Y[y * W + x] - b3[y * W + x];
      resAll[j] = rr;
      if (GM[j] < 0.02) resFlat.push(rr);
    }
  }
  f.noise = resFlat.length >= 100 ? meanStd(resFlat, resFlat.length)[1] : meanStd(resAll, ni)[1];
  // ---------------- local contrast
  var b8 = boxMean(Y, H, W, 8), lc = 0;
  for (i = 0; i < n; i++) lc += Math.abs(Y[i] - b8[i]);
  f.local_contrast = lc / n;
  f.lc_ratio = f.local_contrast / (f.y_std + EPS);
  // ---------------- spatial / composition (8x8 grid)
  var cnt = new Float64Array(64), gyS = new Float64Array(64), gsS = new Float64Array(64);
  var syw = 0, sxw = 0, sYw = 0, dxw = 0, dyw = 0, sD = 0;
  var inSum = 0, inN = 0, outSum = 0, outN = 0, topS = 0, botS = 0;
  var t3 = Math.floor(H / 3), b3r = H - Math.floor(H / 3);
  var iy0 = Math.floor(H / 4), iy1 = Math.floor((3 * H) / 4), ix0 = Math.floor(W / 4), ix1 = Math.floor((3 * W) / 4);
  for (y = 0; y < H; y++) {
    var cyc = Math.floor((y * 8) / H), ys = (y + 0.5) / H - 0.5;
    for (x = 0; x < W; x++) {
      i = y * W + x;
      var cell = cyc * 8 + Math.floor((x * 8) / W), xs = (x + 0.5) / W - 0.5, v = Y[i];
      cnt[cell] += 1; gyS[cell] += v; gsS[cell] += S[i];
      sYw += v; sxw += v * xs; syw += v * ys;
      var dv = 1 - v;
      sD += dv; dxw += dv * xs; dyw += dv * ys;
      if (y >= iy0 && y < iy1 && x >= ix0 && x < ix1) { inSum += v; inN++; } else { outSum += v; outN++; }
      if (y < t3) topS += v;
      if (y >= b3r) botS += v;
    }
  }
  var gyM = new Float64Array(64), darkC = 0, accC = 0;
  for (k = 0; k < 64; k++) {
    gyM[k] = gyS[k] / cnt[k];
    if (gyM[k] < 0.2) darkC++;
    if (gsS[k] / cnt[k] > 0.5) accC++;
  }
  f.grid_y_std = meanStd(gyM, 64)[1];
  var sorted = Array.prototype.slice.call(gsS).sort(function (p, q) { return p - q; });
  var top8 = 0, allS = 0;
  for (k = 0; k < 64; k++) { allS += gsS[k]; if (k >= 56) top8 += sorted[k]; }
  f.sat_concentration = top8 / (allS + EPS);
  f.dark_cells = darkC / 64; f.accent_cells = accC / 64;
  f.centroid_x = sxw / (sYw + EPS); f.centroid_y = syw / (sYw + EPS);
  f.dark_centroid_x = dxw / (sD + EPS); f.dark_centroid_y = dyw / (sD + EPS);
  f.center_ratio = (inSum / inN) / (outSum / outN + 0.02);
  f.top_bottom = topS / (t3 * W) - botS / ((H - b3r) * W);
  var lin = [], lout = [], ein = 0, eout = 0;
  for (y = 1; y < H - 1; y++) {
    for (x = 1; x < W - 1; x++) {
      j = (y - 1) * Wi + (x - 1);
      if (y >= iy0 && y < iy1 && x >= ix0 && x < ix1) { lin.push(LAP[j]); ein += EDG[j]; }
      else { lout.push(LAP[j]); eout += EDG[j]; }
    }
  }
  var vin = meanStd(lin, lin.length)[1], vout = meanStd(lout, lout.length)[1];
  f.sharp_center_ratio = Math.log((vin * vin + 1e-8) / (vout * vout + 1e-8));
  f.edge_center_ratio = (ein / lin.length + 0.01) / (eout / lout.length + 0.01);
  return f;
}

/* Exact 2x2 box average (integer rounding), bit-identical to features.py box2x. */
function box2x(image) {
  var W = Math.floor(image.width / 2), H = Math.floor(image.height / 2), sw = image.width, d = image.data;
  var out = new Uint8ClampedArray(W * H * 4);
  for (var y = 0; y < H; y++) {
    for (var x = 0; x < W; x++) {
      var i0 = ((2 * y) * sw + 2 * x) * 4, i1 = i0 + sw * 4, o = (y * W + x) * 4;
      for (var c = 0; c < 3; c++) out[o + c] = (d[i0 + c] + d[i0 + 4 + c] + d[i1 + c] + d[i1 + 4 + c] + 2) >> 2;
      out[o + 3] = 255;
    }
  }
  return { width: W, height: H, data: out };
}

function targetSize(w, h) {
  var s = Math.min(1, WORK_SIDE / Math.max(w, h));
  return [Math.max(1, Math.floor(w * s + 0.5)), Math.max(1, Math.floor(h * s + 0.5))];
}

function featureVector(f) {
  var v = new Float64Array(FEATURE_NAMES.length);
  for (var k = 0; k < FEATURE_NAMES.length; k++) v[k] = f[FEATURE_NAMES[k]];
  return v;
}

module.exports = { FEATURE_VERSION: FEATURE_VERSION, WORK_SIDE: WORK_SIDE, FEATURE_NAMES: FEATURE_NAMES,
  styleFeatures: styleFeatures, featureVector: featureVector, box2x: box2x, targetSize: targetSize };
