/* FilmLookStyle public API (ES2017). In the bundle it is exposed as the global `FilmLookStyle`.
 *
 *   FilmLookStyle.analyzeImageData({width, height, data}, 'zh')   // data = RGBA, long side <= WORK_SIDE
 *   FilmLookStyle.analyzeImage(imgElement, 'en')                   // draws to a canvas at WORK_SIDE, then analyses
 */
'use strict';

var F = require('./features.js');
var E = require('./engine.js');
var M = require('./model-data.js');

function analyzeImageData(imageData, lang) {
  var f = F.styleFeatures(imageData);
  return E.analyzeFeatures(f, M, lang || 'en', F.featureVector);
}

/* Draw an <img> (or ImageBitmap/canvas) and analyse it. Two-stage downscale, identical in spirit to the Python
 * reference: the browser draws at 2x the working size (progressive halving keeps each drawImage step <= 2x),
 * then box2x() - our own exact 2x2 average - produces the final pixels, so browser resampler differences are
 * largely averaged away. `canvas` is optional (re-use one to avoid allocations). Images from a file input
 * (data:/blob: URLs) are same-origin in the Mini Tool sandbox, so getImageData is allowed. */
function analyzeImage(img, lang, canvas) {
  var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  if (!w || !h) throw new Error('image not loaded');
  if (Math.max(w, h) <= F.WORK_SIDE) return analyzeImageData(drawTo(img, w, h, w, h, canvas), lang);
  var t = F.targetSize(w, h), tw = 2 * t[0], th = 2 * t[1];
  var src = img, sw = w, sh = h, tmp = null;
  while (sw / 2 >= tw && sh / 2 >= th) {
    var nw = Math.max(tw, Math.floor(sw / 2)), nh = Math.max(th, Math.floor(sh / 2));
    var next = document.createElement('canvas');
    next.width = nw; next.height = nh;
    next.getContext('2d').drawImage(src, 0, 0, sw, sh, 0, 0, nw, nh);
    if (tmp) { tmp.width = 0; tmp.height = 0; }  // release previous intermediate bitmap
    tmp = next; src = next; sw = nw; sh = nh;
  }
  var data = drawTo(src, sw, sh, tw, th, canvas);
  if (tmp) { tmp.width = 0; tmp.height = 0; }
  return analyzeImageData(F.box2x(data), lang);
}

function drawTo(src, sw, sh, tw, th, canvas) {
  var c = canvas || document.createElement('canvas');
  c.width = tw; c.height = th;
  var ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  if ('imageSmoothingQuality' in ctx) ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, sw, sh, 0, 0, tw, th);
  return ctx.getImageData(0, 0, tw, th);
}

module.exports = {
  version: M.model_version, schemaVersion: M.schema_version, featureVersion: F.FEATURE_VERSION,
  WORK_SIDE: F.WORK_SIDE, analyzeImageData: analyzeImageData, analyzeImage: analyzeImage,
  styleFeatures: F.styleFeatures, box2x: F.box2x, targetSize: F.targetSize, profiles: Object.keys(M.profiles)
};
