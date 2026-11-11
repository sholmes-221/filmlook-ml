"""Who the reference database looks for, and where.

``collect`` photographers have works that sources mark public domain or CC0 (US federal FSA/OWI and NCLC work,
authors dead long enough, pre-1930 US publication). ``metadata_only`` photographers are researched for the Visual DNA
knowledge base but none of their images enter the pipeline: their work is in copyright and no open licence was found
(Magnum photographers among them; Magnum images are never assumed to be public domain).

``website_id`` is the id in FilmAnalysis's ``photographer_dna/seed_data`` where the photographer exists there.
``match`` lists the tokens that must appear in a source's attribution text, so a portrait OF the photographer by
someone else, or a work by an assistant, is not taken as a work BY them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    status: str  # collect | metadata_only
    match: tuple[str, ...] = ()
    website_id: str | None = None
    commons: tuple[str, ...] = ()
    aic: tuple[str, ...] = ()
    note: str = ""
    cap: int = 300


ROSTER: tuple[Entry, ...] = (
    # Farm Security Administration / Office of War Information: same agency, era and subjects. Kept together on
    # purpose so that subject matter cannot separate them (the subject-bias control the brief asks for).
    Entry("dorothea_lange", "Dorothea Lange", "collect", ("lange",), "dorothea_lange", ("Photographs by Dorothea Lange",)),
    Entry("walker_evans", "Walker Evans", "collect", ("evans",), "walker_evans", ("Photographs by Walker Evans",)),
    Entry("jack_delano", "Jack Delano", "collect", ("delano",), None, ("Photographs by Jack Delano",)),
    Entry("russell_lee", "Russell Lee", "collect", ("russell lee", "lee, russell"), None, ("Photographs by Russell Lee",)),
    Entry("arthur_rothstein", "Arthur Rothstein", "collect", ("rothstein",), None, ("Photographs by Arthur Rothstein",)),
    Entry("john_vachon", "John Vachon", "collect", ("vachon",), None, ("Photographs by John Vachon",)),
    Entry("marion_post_wolcott", "Marion Post Wolcott", "collect", ("wolcott",), None, ("Photographs by Marion Post Wolcott",)),
    Entry("ben_shahn", "Ben Shahn", "collect", ("shahn",), None, ("Photographs by Ben Shahn",)),
    Entry("carl_mydans", "Carl Mydans", "collect", ("mydans",), None, ("Photographs by Carl Mydans",)),
    Entry("lewis_hine", "Lewis Hine", "collect", ("hine",), "lewis_hine", ("Photographs by Lewis Hine",)),
    Entry("eugene_atget", "Eugène Atget", "collect", ("atget",), None, ("Photographs by Eugène Atget",), ("Jean-Eugène-Auguste Atget",)),
    Entry("alfred_stieglitz", "Alfred Stieglitz", "collect", ("stieglitz",), None, ("Photographs by Alfred Stieglitz",), ("Alfred Stieglitz",)),
    Entry("gertrude_kasebier", "Gertrude Käsebier", "collect", ("käsebier", "kasebier"), None, ("Photographs by Gertrude Käsebier",)),
    Entry("clarence_h_white", "Clarence H. White", "collect", ("clarence",), None, (), ("Clarence H. White",)),
    # Batch 2 (2026-11-11): more bodies of work that sources mark public domain. US federal work (FSA/OWI, War
    # Relocation Authority, National Park Service), authors dead long enough, and Japanese photographs that Commons
    # reviews as PD-Japan-oldphoto. Same per-file rule as above: only a PD / CC0 template on the file itself counts.
    Entry("ansel_adams", "Ansel Adams", "collect", ("ansel adams", "adams, ansel"), "ansel_adams", ("Photographs by Ansel Adams",), note="federal work only (Manzanar, National Park Service murals project)"),
    Entry("gordon_parks", "Gordon Parks", "collect", ("gordon parks", "parks, gordon"), None, ("Photographs by Gordon Parks",), note="FSA/OWI federal work only"),
    Entry("esther_bubley", "Esther Bubley", "collect", ("bubley",), None, ("Photographs by Esther Bubley",)),
    Entry("alfred_palmer", "Alfred T. Palmer", "collect", ("palmer",), None, ("Photographs by Alfred T. Palmer",), note="OWI colour transparencies"),
    Entry("john_collier", "John Collier Jr.", "collect", ("collier",), None, ("Photographs by John Collier, Jr.", "Photographs by John Collier")),
    Entry("berenice_abbott", "Berenice Abbott", "collect", ("abbott",), None, ("Photographs by Berenice Abbott",)),
    Entry("carleton_watkins", "Carleton Watkins", "collect", ("watkins",), None, ("Photographs by Carleton Watkins",)),
    Entry("timothy_osullivan", "Timothy H. O'Sullivan", "collect", ("sullivan",), None, ("Photographs by Timothy H. O'Sullivan",)),
    Entry("edward_curtis", "Edward S. Curtis", "collect", ("curtis",), None, ("Photographs by Edward Sheriff Curtis", "Photographs by Edward S. Curtis")),
    Entry("prokudin_gorsky", "Sergey Prokudin-Gorsky", "collect", ("prokudin", "прокудин"), None, ("Photographs by Sergey Prokudin-Gorsky",), note="early three-colour photographs"),
    Entry("julia_margaret_cameron", "Julia Margaret Cameron", "collect", ("cameron",), None, ("Photographs by Julia Margaret Cameron",)),
    Entry("jacob_riis", "Jacob Riis", "collect", ("riis",), None, ("Photographs by Jacob Riis",)),
    Entry("roger_fenton", "Roger Fenton", "collect", ("fenton",), None, ("Photographs by Roger Fenton",)),
    Entry("nadar", "Nadar", "collect", ("nadar", "tournachon"), None, ("Photographs by Nadar",)),
    Entry("eadweard_muybridge", "Eadweard Muybridge", "collect", ("muybridge",), None, ("Photographs by Eadweard Muybridge",)),
    Entry("kusakabe_kimbei", "Kusakabe Kimbei", "collect", ("kimbei", "kusakabe"), None, ("Photographs by Kusakabe Kimbei",)),
    Entry("felice_beato", "Felice Beato", "collect", ("beato",), None, ("Photographs by Felice Beato",)),
    Entry("toni_frissell", "Toni Frissell", "collect", ("frissell",), None, ("Photographs by Toni Frissell",)),
    Entry("ihei_kimura", "Ihei Kimura", "collect", ("kimura", "木村"), "ihei_kimura", ("Photographs by Ihei Kimura", "Ihei Kimura")),
    Entry("ken_domon", "Ken Domon", "collect", ("domon", "土門"), "ken_domon", ("Photographs by Ken Domon", "Ken Domon")),
    Entry("tadahiko_hayashi", "Tadahiko Hayashi", "collect", ("hayashi", "林忠彦"), "tadahiko_hayashi", ("Photographs by Tadahiko Hayashi", "Tadahiko Hayashi")),
    Entry("shinzo_fukuhara", "Shinzo Fukuhara", "collect", ("fukuhara", "福原"), "shinzo_fukuhara", ("Photographs by Shinzō Fukuhara", "Shinzō Fukuhara")),
    Entry("nakaji_yasui", "Nakaji Yasui", "collect", ("yasui", "安井"), "nakaji_yasui", ("Photographs by Nakaji Yasui", "Nakaji Yasui")),
    Entry("hiroshi_hamaya", "Hiroshi Hamaya", "collect", ("hamaya", "濱谷", "浜谷"), "hiroshi_hamaya", ("Photographs by Hiroshi Hamaya", "Hiroshi Hamaya")),
    # Researched, never collected.
    Entry("henri_cartier_bresson", "Henri Cartier-Bresson", "metadata_only", website_id="henri_cartier_bresson", note="in copyright (d. 2004); Magnum / Fondation HCB"),
    Entry("robert_capa", "Robert Capa", "metadata_only", note="treated as in copyright (d. 1954); Magnum / ICP; no open licence found"),
    Entry("david_seymour", "David Seymour (Chim)", "metadata_only", note="in copyright (d. 1956); Magnum"),
    Entry("george_rodger", "George Rodger", "metadata_only", note="in copyright (d. 1995); Magnum"),
    Entry("masahisa_fukase", "Masahisa Fukase", "metadata_only", website_id="masahisa_fukase", note="in copyright (d. 2012); Masahisa Fukase Archives"),
    Entry("rinko_kawauchi", "Rinko Kawauchi", "metadata_only", website_id="rinko_kawauchi", note="living"),
    Entry("saul_leiter", "Saul Leiter", "metadata_only", website_id="saul_leiter", note="in copyright (d. 2013); Saul Leiter Foundation"),
    Entry("daido_moriyama", "Daido Moriyama", "metadata_only", website_id="daido_moriyama", note="living"),
)

BY_ID = {e.id: e for e in ROSTER}


def collectable() -> list[Entry]:
    return [e for e in ROSTER if e.status == "collect"]
