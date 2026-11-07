"""CommonCatalog CC-BY (Flickr/YFCC100M photos re-hosted on Hugging Face) pilot tools.

Licensing: every row carries its own ``licenseurl``. We accept ONLY
``creativecommons.org/licenses/by/<ver>/`` (CC-BY: commercial use allowed, attribution required).
The dataset-level HF tag (cc-by-4.0) is NOT relied on. The CC-BY-SA sibling dataset is not used.

Capture medium is inferred from EXIF / capture device / user tags. It is *metadata*, not a
human judgement of aesthetics, but it is independent of every teacher in this repo, which makes
it the only non-circular signal available without human labels.

Image bytes are fetched page-by-page with HTTP range reads (the files carry a parquet offset index
and one JPEG per data page), so we never download whole 130 MB row groups.
"""
from __future__ import annotations

import io
import json
import re
import struct
import urllib.parse
from dataclasses import dataclass

REPO = "common-canvas/commoncatalog-cc-by"
META_COLS = ["photoid", "uid", "unickname", "datetaken", "capturedevice", "title", "usertags",
             "licensename", "licenseurl", "pageurl", "width", "height", "exif", "sha256",
             "blip2_caption", "status"]
LICENSE_RE = re.compile(r"creativecommons\.org/licenses/by/(\d\.\d)/?$")

SCANNER_RE = re.compile(r"coolscan|super ?coolscan|ls-\d{3,4}|perfection|canoscan|scanjet|noritsu|frontier|"
                        r"pakon|plustek|opticfilm|reflecta|film ?scanner|\bscanner\b|v[5-8][05]0|4990|\bx-?5\b|"
                        r"imacon|flextight|hasselblad x", re.I)
FILM_TAG_RE = re.compile(r"(^|,)(film|35mm|35mmfilm|120|120film|mediumformat|filmphotography|analog|analogue|"
                         r"filmisnotdead|believeinfilm|istillshootfilm|portra\d*|kodakportra\d*|ektar\d*|ektachrome|"
                         r"kodachrome\d*|velvia\d*|provia\d*|superia\d*|fujisuperia\d*|trix|tri-x|trix400|hp5|hp5plus|"
                         r"fp4|ilford\w*|tmax\w*|kodakgold\w*|gold200|gold400|fujicolor\w*|reala|neopan\w*|"
                         r"acros\w*|xp2|kodak400|kodakmax\w*|ultramax\w*|c41|e6|crossprocessed|xpro|negative|slide|"
                         r"filmscan|scannedfilm|holga|lomo|lca|lomography|diana|yashica\w*|pentaxk1000|canonae1|"
                         r"olympusom\w*|nikonfm\w*|nikonf\d+|leicam\d)(,|$)", re.I)
PHONE_RE = re.compile(r"iphone|htc|nokia|motorola|droid|nexus|galaxy|samsung (gt|sgh|sch|sph)|sony ?ericsson|"
                      r"\blg[- ]|blackberry|palm|\bmi \d|huawei|t-mobile", re.I)
CAMERA_RE = re.compile(r"canon|nikon|sony|olympus|panasonic|fujifilm|pentax|leica|ricoh|samsung|kodak|"
                       r"casio|sigma|minolta|konica|hasselblad|phase one|epson r-d1|gopro", re.I)
FILTER_TAG_RE = re.compile(r"(^|,)(vsco\w*|instagram\w*|hipstamatic\w*|iphoneography|filter|vintage|"
                           r"crossprocess\w*|faux\w*|fakefilm|filmsimulation|filmlook|retro|lomoeffect|"
                           r"instagramapp|vscocam|snapseed|afterlight)(,|$)", re.I)
# BLIP captions that indicate a non-photograph or a reproduction we should not train on
EXCLUDE_CAPTION_RE = re.compile(r"screenshot|\blogo\b|poster|drawing|painting|illustration|cartoon|\bmap\b|"
                                r"diagram|\btext\b|magazine|newspaper|book cover|comic|sign that says|"
                                r"computer screen|website", re.I)
SUBJECTS = {
    "portrait": r"\b(man|woman|men|women|girl|boy|child|children|person|people|face|couple|baby|portrait)\b",
    "street": r"\b(street|crowd|sidewalk|market|traffic|bus|taxi|alley|shop|storefront|pedestrian)",
    "night": r"\b(night|nighttime|dark|neon|lit up|fireworks|lights at)\b",
    "landscape": r"\b(mountain|mountains|lake|beach|field|forest|ocean|sea|river|sunset|sunrise|desert|valley|"
                 r"waterfall|hills|snow|sky)\b",
    "architecture": r"\b(building|buildings|church|cathedral|bridge|tower|skyline|city|interior|room|house|temple|"
                    r"architecture|skyscraper)\b",
}


def _decode(s):
    return urllib.parse.unquote_plus(s or "")


def classify(row: dict) -> dict:
    """Return license / medium / subject / exclusion info for one metadata row."""
    out = {"license_ok": False, "license": None, "exclude": None}
    m = LICENSE_RE.search((row.get("licenseurl") or "").strip())
    if m:
        out["license_ok"], out["license"] = True, f"CC-BY-{m.group(1)}"
    try:
        exif = json.loads(row.get("exif") or "{}")
    except json.JSONDecodeError:
        exif = {}
    make = str(exif.get("Image Make", "")).strip()
    model = str(exif.get("Image Model", "")).strip()
    software = str(exif.get("Image Software", "")).strip()
    device = _decode(row.get("capturedevice"))
    devstr = f"{make} {model} {device}"
    tags = (row.get("usertags") or "").lower()
    cap = (row.get("blip2_caption") or "").lower()
    has_film_tag = bool(FILM_TAG_RE.search(tags))
    if SCANNER_RE.search(devstr) or SCANNER_RE.search(software):
        medium = "film_scan" if has_film_tag or not CAMERA_RE.search(devstr) else "scan_unknown"
    elif PHONE_RE.search(devstr):
        medium = "smartphone_filtered" if FILTER_TAG_RE.search(tags) else "smartphone"
    elif CAMERA_RE.search(devstr) and exif.get("EXIF DateTimeOriginal"):
        if has_film_tag:
            medium = "ambiguous"          # film tags on a digital-camera EXIF: DSLR scan or film-look edit
        elif FILTER_TAG_RE.search(tags):
            medium = "digital_filtered"
        else:
            medium = "digital_camera"
    elif has_film_tag and not devstr.strip():
        medium = "film_tagged_no_exif"   # weaker evidence; not used for the medium benchmark
    else:
        medium = "unknown"
    subjects = [k for k, rx in SUBJECTS.items() if re.search(rx, cap) or re.search(rx, tags.replace(",", " "))]
    if EXCLUDE_CAPTION_RE.search(cap):
        out["exclude"] = "non_photo_caption"
    if row.get("status") != "success":
        out["exclude"] = "status"
    mono = bool(re.search(r"(^|,)(bw|blackandwhite|blackwhite|monochrome|noiretblanc|b&w)(,|$)", tags)) \
        or "black and white" in cap
    out.update(medium=medium, subjects=subjects, monochrome_tag=mono, camera=f"{make} {model}".strip() or device,
               software=software)
    return out


# --------------------------------------------------------------------------------------------
# Minimal parquet page reader (thrift compact protocol) for single-row byte-range fetches.
# --------------------------------------------------------------------------------------------
class _Thrift:
    def __init__(self, buf: bytes):
        self.b, self.i = buf, 0

    def byte(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def varint(self):
        shift = res = 0
        while True:
            x = self.byte()
            res |= (x & 0x7F) << shift
            if not x & 0x80:
                return res
            shift += 7

    def zz(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, t):
        if t in (1, 2):
            return t == 1
        if t == 3:
            return self.byte()
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            self.i += 8
            return None
        if t == 8:
            n = self.varint()
            self.i += n
            return None
        if t in (9, 10):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            for _ in range(n):
                self.value(et if et not in (1, 2) else 3)
            return None
        if t == 11:
            n = self.varint()
            if n:
                kv = self.byte()
                for _ in range(n):
                    self.value(kv >> 4)
                    self.value(kv & 0x0F)
            return None
        if t == 12:
            return self.struct()
        raise ValueError(f"thrift type {t}")

    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0:
                return out
            d, t = h >> 4, h & 0x0F
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.value(t)


@dataclass
class Page:
    offset: int           # file offset of page *data* (after header)
    compressed: int
    uncompressed: int
    num_values: int
    v2: bool
    def_len: int = 0
    rep_len: int = 0
    is_compressed: bool = True


def page_index(fh, start: int, end: int, peek: int = 512) -> list[Page]:
    """Walk data-page headers in [start, end). One small range read per page."""
    pages, off = [], start
    while off < end:
        fh.seek(off)
        head = fh.read(min(peek, end - off))
        t = _Thrift(head)
        h = t.struct()
        data_off = off + t.i
        ptype, unc, comp = h.get(1), h.get(2), h.get(3)
        if ptype == 0:
            if h[5].get(2, 0) != 0:
                raise NotImplementedError("only PLAIN-encoded pages are supported")
            pages.append(Page(data_off, comp, unc, h[5][1], False))
        elif ptype == 3:
            d = h[8]
            if d.get(4, 0) != 0:
                raise NotImplementedError("only PLAIN-encoded pages are supported")
            pages.append(Page(data_off, comp, unc, d[1], True, d.get(5, 0), d.get(6, 0), d.get(7, True)))
        off = data_off + comp
    return pages


def _rle_bits(buf: bytes, n: int) -> list[int]:
    t, out = _Thrift(buf), []
    while len(out) < n and t.i < len(buf):
        h = t.varint()
        if h & 1:
            for _ in range(h >> 1):
                byte = t.byte()
                out.extend((byte >> k) & 1 for k in range(8))
        else:
            v = t.byte()
            out.extend([v] * (h >> 1))
    return out[:n]


def read_page_values(fh, page: Page, codec: str) -> list[bytes | None]:
    import pyarrow as pa

    fh.seek(page.offset)
    raw = fh.read(page.compressed)

    def dec(b, size):
        if codec == "UNCOMPRESSED":
            return b
        return pa.decompress(b, decompressed_size=size, codec=codec.lower()).to_pybytes()

    if page.v2:
        lv = page.def_len + page.rep_len
        defs = _rle_bits(raw[page.rep_len:lv], page.num_values)
        body = dec(raw[lv:], page.uncompressed - lv) if page.is_compressed else raw[lv:]
    else:
        body = dec(raw, page.uncompressed)
        ln = struct.unpack_from("<I", body)[0]
        defs = _rle_bits(body[4:4 + ln], page.num_values)
        body = body[4 + ln:]
    vals, p = [], 0
    for d in defs:
        if not d:
            vals.append(None)
            continue
        ln = struct.unpack_from("<I", body, p)[0]
        vals.append(body[p + 4:p + 4 + ln])
        p += 4 + ln
    return vals


def fetch_rows(fs, path: str, row_group: int, rows: set[int], column: str = "jpg") -> dict[int, bytes]:
    """Fetch selected rows (indices within the row group) of a BYTE_ARRAY column."""
    import pyarrow.parquet as pq

    with fs.open(path, block_size=1 << 16, cache_type="none") as fh:   # range reads, no block cache
        pf = pq.ParquetFile(fh)
        rg = pf.metadata.row_group(row_group)
        col = next(rg.column(i) for i in range(rg.num_columns) if rg.column(i).path_in_schema == column)
        start = col.dictionary_page_offset or col.data_page_offset
        pages = page_index(fh, start, start + col.total_compressed_size)
        out, r0 = {}, 0
        for pg in pages:
            want = {r - r0 for r in rows if r0 <= r < r0 + pg.num_values}
            if want:
                vals = read_page_values(fh, pg, col.compression)
                for k in want:
                    if vals[k] is not None:
                        out[r0 + k] = vals[k]
            r0 += pg.num_values
        del pf
    return out


def jpeg_ok(b: bytes) -> bool:
    return b[:3] == b"\xff\xd8\xff"


def as_image_bytes(b: bytes) -> io.BytesIO:
    return io.BytesIO(b)
