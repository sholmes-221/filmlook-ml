// Mini Tool sandbox: no inline scripts, no network. File input -> blob URL -> <img> -> analyzeImage.
document.getElementById('pick').addEventListener('change', function (e) {
  var file = e.target.files && e.target.files[0];
  if (!file) return;
  var url = URL.createObjectURL(file), img = new Image();
  img.onload = function () {
    var t0 = Date.now();
    var r = FilmLookStyle.analyzeImage(img, 'zh');
    URL.revokeObjectURL(url);
    document.getElementById('out').textContent = JSON.stringify(r, null, 2) + '\n' + (Date.now() - t0) + ' ms';
  };
  img.src = url;
});
