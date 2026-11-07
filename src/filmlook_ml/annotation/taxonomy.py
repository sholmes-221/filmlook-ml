"""Controlled taxonomy for categorization annotations (taxonomy-v2, 2026-11-07).

Used by the main DINOv2-small classifier project only. It is independent of the Mini Tool's style profiles
(``filmlook_ml.style.profiles`` / ``js/filmlook-style``), which are left unchanged.

Every label means "visually resembles". Nothing here identifies authorship or the film stock actually used.
- ``unknown``: the image gives too little evidence to judge (tiny, damaged, not a photograph, ...).
- ``none_of_the_above``: a judgeable photograph that resembles no listed style or stock.

Adding a category = edit this file, bump TAXONOMY_VERSION, and record it in docs/AI_DATA_POLICY.md.
Annotations made under an older version are kept and never silently remapped.
"""
TAXONOMY_VERSION = "taxonomy-v2"
UNKNOWN, NONE = "unknown", "none_of_the_above"
SPECIAL = [UNKNOWN, NONE]

# Short visual descriptions are given to the annotator so that labels refer to a shared description.
PHOTOGRAPHERS: dict[str, str] = {
    "Saul Leiter": "muted, painterly colour; layered views through glass, rain, snow; large dark masses; small accents of red",
    "William Eggleston": "saturated everyday colour; banal subjects treated formally; strong single colour fields; flash or flat light",
    "Alex Webb": "dense, complex street layers; intense saturated colour; hard tropical light and deep shadow fragments",
    "Daido Moriyama": "black-and-white, high contrast, coarse grain, blur and tilt; gritty urban snapshots",
    "Shomei Tomatsu": "black-and-white documentary; dense blacks, raw texture; postwar urban and social subjects",
    "Rinko Kawauchi": "soft, pale, high-key colour; gentle overexposure; quiet intimate details and light",
    "Stephen Shore": "calm, frontal, precise colour; neutral daylight; ordinary architecture and streets; large-format clarity",
    "Martin Parr": "garish saturated colour; harsh direct flash; close, satirical views of people, food and leisure",
    "Fan Ho": "black-and-white; dramatic shafts of light and long shadows; geometric street scenes with small figures",
    "Nobuyoshi Araki": "intimate diaristic snapshots; flash; saturated or black-and-white; close personal subjects",
}
PHOTOGRAPHER_LABELS = [*PHOTOGRAPHERS, NONE, UNKNOWN]

FILM_STOCKS: dict[str, str] = {
    "Kodak Portra 160": "low-saturation, fine-grain colour negative; neutral-warm; very soft contrast",
    "Kodak Portra 400": "warm, soft-contrast colour negative; natural skin tones; forgiving highlights",
    "Kodak Portra 800": "warm colour negative; more grain and saturation than Portra 400",
    "Kodak Gold 200": "warm, yellow-golden consumer colour negative; moderate saturation",
    "Kodak Ektar 100": "high-saturation, very fine-grain colour negative; punchy reds and blues",
    "Kodak ColorPlus 200": "budget warm colour negative; yellow-orange cast; moderate grain",
    "Kodak Ultramax 400": "consumer colour negative; warm, saturated, visible grain",
    "Fujifilm Pro 400H": "cool, pastel colour negative; green-cyan bias; lifted shadows",
    "Fujifilm Superia X-TRA 400": "consumer colour negative; green-leaning shadows; cooler tones; visible grain",
    "Fujifilm C200": "consumer colour negative; slightly cool-green; moderate saturation",
    "Fujifilm Velvia 50": "slide film; very high saturation and contrast; deep blacks",
    "Fujifilm Provia 100F": "slide film; neutral accurate colour; moderate-high contrast; fine grain",
    "Kodak Tri-X 400": "black-and-white; punchy midtone contrast; visible classic grain",
    "Ilford HP5 Plus 400": "black-and-white; moderate contrast; moderate grain; broad tonality",
    "Ilford FP4 Plus 125": "black-and-white; fine grain; smooth tonality; medium contrast",
    "Ilford Delta 3200": "black-and-white; very coarse grain; low-light; compressed tones",
    "Cinestill 800T": "tungsten-balanced colour; cool cast; red halation around bright lights; night scenes",
}
FILM_LABELS = [*FILM_STOCKS, NONE, UNKNOWN]
BW_STOCKS = {"Kodak Tri-X 400", "Ilford HP5 Plus 400", "Ilford FP4 Plus 125", "Ilford Delta 3200"}
BW_PHOTOGRAPHERS = {"Daido Moriyama", "Shomei Tomatsu", "Fan Ho"}

MAX_ALTERNATIVES = 2
MAX_EVIDENCE = 3
EVIDENCE_MAX_CHARS = 80

COLOR = {
    "temperature": ["warm", "neutral", "cool", "not_applicable"],          # not_applicable: monochrome
    "saturation": ["monochrome", "low", "medium", "high"],
    "color_cast": ["none", "yellow", "orange", "red", "magenta", "blue", "cyan", "green", "not_applicable"],
}
DOMINANT_COLORS = ["red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "brown", "beige",
                   "black", "white", "gray"]
LIGHTING = {
    "source": ["natural", "artificial", "mixed", "unknown"],
    "quality": ["soft", "hard", "mixed"],
    "key": ["low_key", "balanced", "high_key"],
    "direction": ["backlit", "front_lit", "side_lit", "diffuse", "unknown"],
}
TONE = {
    "contrast": ["low", "medium", "high"],
    "highlight_rolloff": ["soft", "moderate", "abrupt", "indeterminate"],
    "shadow_depth": ["lifted", "neutral", "deep"],
}
TEXTURE = {
    "visible_grain": ["none", "subtle", "moderate", "strong"],
    "grain_appearance": ["fine", "coarse", "indeterminate"],               # indeterminate when grain is none
    "sharpness": ["soft", "moderate", "crisp"],
}
GENRES = ["street", "portrait", "landscape", "architecture", "still_life", "documentary", "abstract", "other"]
AESTHETICS = ["muted", "vivid", "nostalgic", "cinematic", "minimal", "gritty", "dreamy", "graphic", "snapshot",
              "painterly", "clinical", "documentary", "moody", "airy"]
IMAGE_FLAGS = ["watermark_or_text", "border_or_frame", "low_resolution", "heavy_post_processing", "scan_of_print",
               "not_a_photograph", "artwork_reproduction", "multiple_exposure_or_collage"]


def describe() -> dict:
    return {"taxonomy_version": TAXONOMY_VERSION, "photographers": PHOTOGRAPHERS, "film_stocks": FILM_STOCKS,
            "special_labels": SPECIAL, "color": COLOR, "dominant_colors": DOMINANT_COLORS, "lighting": LIGHTING,
            "tone": TONE, "texture": TEXTURE, "genres": GENRES, "aesthetics": AESTHETICS, "image_flags": IMAGE_FLAGS}


# ----------------------------------------------------------------------------- taxonomy-v3 (2026-11-07, pilot 2)
# Photographers and film stocks are unchanged from v2. Added in v3:
# - broad film-aesthetic categories (visual looks, never claims about the actual process);
# - an apparent-capture impression, compared with EXIF metadata only after annotation;
# - an orientation flag.
TAXONOMY_V3_VERSION = "taxonomy-v3"
FILM_CATEGORIES = {
    "classic_color_negative": "colour-negative look: soft contrast, forgiving highlights, warm or pastel colour",
    "vivid_color_reversal": "slide look: high saturation and contrast, dense shadows",
    "bw_negative": "black-and-white film look with fine to moderate grain",
    "high_grain_monochrome": "black-and-white with coarse, dominant grain (push-processed / high-ISO look)",
    "archival_early_process": "historic print or early process: toned, faded, scratched, or plate-like",
    "cross_processed_redscale": "strong unnatural casts and shifted contrast typical of cross-processing or redscale",
    "digital_film_filter_emulation": "app or preset imitation of film: uniform toning, fake borders, overlay grain",
    "clean_digital": "no film aesthetic: clean digital rendering (added by Claude; not in the owner's list)",
    "unknown_indeterminate": "cannot be judged (reproduction of a print, too small, heavily damaged, ...)",
}
APPARENT_CAPTURE = ["film_look", "digital_look", "digital_filter_look", "print_or_reproduction", "indeterminate"]
ORIENTATION_FLAG = "misoriented"
IMAGE_FLAGS_V3 = [*IMAGE_FLAGS, ORIENTATION_FLAG]
TOP_K = 3


def describe_v3() -> dict:
    return {**describe(), "taxonomy_version": TAXONOMY_V3_VERSION, "film_categories": FILM_CATEGORIES,
            "apparent_capture": APPARENT_CAPTURE, "image_flags": IMAGE_FLAGS_V3, "top_k": TOP_K}
