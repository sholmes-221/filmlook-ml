/* Optional Dedicated Worker (Mini Tool allows ONE classic worker per page, client >= 9.53).
 * The main thread draws the photo at 2x the working size (FilmLookStyle.targetSize-style math, see README),
 * calls getImageData, and posts {width, height, buffer: imageData.data.buffer, lang} with the buffer transferred.
 * Feature extraction + scoring then run off the UI thread.
 */
importScripts('../dist/filmlook-style.js');   // classic worker: in-package script only
self.onmessage = function (e) {
  var m = e.data;  // {width, height, buffer, lang} - a 2x intermediate from the main thread
  try {
    var img = { width: m.width, height: m.height, data: new Uint8ClampedArray(m.buffer) };
    self.postMessage({ ok: true, result: FilmLookStyle.analyzeImageData(FilmLookStyle.box2x(img), m.lang) });
  } catch (err) {
    self.postMessage({ ok: false, error: String(err && err.message || err) });
  }
};
