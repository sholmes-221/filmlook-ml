"""Streamlit human-review app (rubric v1).  Run:  python scripts/review_labels.py --reviewer <name>

Saves to data/labels/human_labels.jsonl and human_pairs.jsonl with reviewer provenance.
Isolation: labels on TEST-split images form the human benchmark. Training only reads the train split and
model selection only the val split, so the human test set never influences training or selection.
"""
from __future__ import annotations

import random
import sys

import streamlit as st
from PIL import Image

from filmlook_ml import paths as P
from filmlook_ml.constants import ATTRIBUTE_NAMES, ATTRIBUTES, RUBRIC
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.labeling.schema import ImageLabel, PairLabel, TeacherOutput
from filmlook_ml.compliance.provenance import make_provenance
from filmlook_ml.labeling.store import append_label, append_pair, load_labels

reviewer = sys.argv[sys.argv.index("--reviewer") + 1] if "--reviewer" in sys.argv else "anonymous"
st.set_page_config(page_title="FilmLook label review", layout="wide")


@st.cache_data
def _manifest():
    return read_jsonl(P.MANIFESTS / "manifest.jsonl")


manifest = _manifest()
if not manifest:
    st.error("No manifest. Run scripts/prepare_dataset.py first.")
    st.stop()
labels = load_labels(P.LABELS)
st.sidebar.write(f"Reviewer: **{reviewer}**")
st.sidebar.write(f"Human labels so far: {len(labels['human'])}")
mode = st.sidebar.radio("Mode", ["Pairwise comparison", "Single image", "Style similarity"])
only_unreviewed = st.sidebar.checkbox("Only unreviewed", True)
split_filter = st.sidebar.multiselect("Splits", ["train", "val", "test"], ["test"])
st.sidebar.caption("Default: TEST split = isolated human benchmark.")
with st.sidebar.expander("Rubric (higher = more film-like / natural)"):
    for k, v in RUBRIC.items():
        st.markdown(f"**{k}**: {v}")

pool = [r for r in manifest if r.get("split") in split_filter
        and (not only_unreviewed or r["image_id"] not in labels["human"])]
if not pool:
    st.success("Nothing left to review with these filters.")
    st.stop()

if mode == "Style similarity":
    from filmlook_ml.labeling.schema import StyleJudgement
    from filmlook_ml.style.profiles import PROFILES
    done = {j["image_id"] for j in read_jsonl(P.LABELS / "style_judgements.jsonl") if j.get("reviewer") == reviewer}
    spool = [r for r in pool if r["edit_type"] == "original" and r["image_id"] not in done]
    if not spool:
        st.success("No unjudged originals left with these filters.")
        st.stop()
    r = spool[int(st.sidebar.number_input("Index", 0, len(spool) - 1, 0))]
    c1, c2 = st.columns([3, 2])
    c1.image(Image.open(r["path"]), caption=r["image_id"], width="stretch")
    with c2:
        st.caption("Which styles does the photo's RENDERING resemble (colour, tone, texture, light, composition)? "
                   "Ignore subject matter. Most photos resemble none - that is a valid answer.")
        opts = {pid: f"{p['name']['en']} - {p['basis']['en']}" for pid, p in PROFILES.items()}
        picks = st.multiselect("Up to 3 styles, best first", list(opts), format_func=lambda k: opts[k],
                               max_selections=3)
        none = st.checkbox("No reliable match")
        unc = st.radio("Your certainty", ["low", "medium", "high"], index=1, horizontal=True)
        if st.button("Save judgement", type="primary"):
            j = StyleJudgement(image_id=r["image_id"], sha256=r["sha256"], source="human", reviewer=reviewer,
                               matches=[] if none else picks, no_match=none or not picks,
                               uncertainty={"low": "high", "medium": "medium", "high": "low"}[unc],
                               provenance=make_provenance("human", r, reviewer=reviewer))
            with open(P.LABELS / "style_judgements.jsonl", "a") as fh:
                fh.write(j.model_dump_json() + "\n")
            st.success("Saved.")
elif mode == "Single image":
    idx = st.sidebar.number_input("Index", 0, len(pool) - 1, 0)
    r = pool[idx]
    c1, c2 = st.columns([3, 2])
    c1.image(Image.open(r["path"]), caption=f"{r['image_id']}", width="stretch")
    with c2:
        st.caption("Judge the rendering, not the subject. The edit type is hidden on purpose.")
        mono = st.checkbox("Monochrome image")
        score = st.slider("Film score (0-100)", 0, 100, 50)
        attrs = {}
        for a in ATTRIBUTES:
            if a == "color_response" and mono:
                attrs[a] = None
                continue
            attrs[a] = st.slider(ATTRIBUTE_NAMES[a]["en"], 0.0, 1.0, 0.5, 0.05, help=RUBRIC[a])
        notes = st.text_input("Notes (strengths / weaknesses)")
        unsure = st.checkbox("I am unsure about this image")
        if st.button("Save label", type="primary"):
            lab = ImageLabel(image_id=r["image_id"], sha256=r["sha256"], source="human", teacher_model="human",
                             prompt_version="review-v1-rubric1", review_status="accepted", reviewer=reviewer,
                             notes=notes, provenance=make_provenance("human", r, reviewer=reviewer),
                             label=TeacherOutput(film_score=score, attributes=attrs, monochrome=mono,
                                                              uncertainty="high" if unsure else "low",
                                                              visual_evidence=notes,
                                                              label_confidence=0.4 if unsure else 0.9))
            append_label(P.LABELS, "human", lab)
            st.success("Saved.")
else:
    fams: dict[str, list[dict]] = {}
    for r in pool:
        fams.setdefault(r["family_id"], []).append(r)
    same_photo = st.sidebar.checkbox("Two versions of the same photo", True)
    rng = random.Random(st.session_state.get("pair_seed", 0))
    multi = [f for f in fams.values() if len(f) > 1]
    if same_photo and multi:
        a, b = rng.sample(rng.choice(multi), 2)
    else:
        a, b = rng.sample(pool, 2)
    attr = st.selectbox("Judge on", ["film_score", *ATTRIBUTES],
                        format_func=lambda x: "Overall film look" if x == "film_score" else ATTRIBUTE_NAMES[x]["en"])
    c1, c2 = st.columns(2)
    c1.image(Image.open(a["path"]), caption="A", width="stretch")
    c2.image(Image.open(b["path"]), caption="B", width="stretch")
    st.write("Which looks more **convincingly film-like / natural** on this criterion? (Tie is fine.)")
    cols = st.columns(4)
    for col, (lbl, w) in zip(cols, [("A", "a"), ("B", "b"), ("Tie", "tie"), ("Skip", None)]):
        if col.button(lbl):
            if w:
                append_pair(P.LABELS, PairLabel(image_a=a["image_id"], image_b=b["image_id"], attribute=attr,
                                                winner=w, source="human", reviewer=reviewer,
                                                provenance=make_provenance("human", a, reviewer=reviewer)))
            st.session_state["pair_seed"] = rng.randint(0, 10**9)
            st.rerun()
