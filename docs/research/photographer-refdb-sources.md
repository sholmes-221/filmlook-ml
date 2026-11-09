# Photographer reference database: source and rights research (2026-11-09)

## Sources tried, in the brief's order of preference

| source | outcome | how it is used |
|---|---|---|
| Library of Congress (loc.gov JSON API, FSA/OWI, NCLC) | every scripted request answered with a Cloudflare challenge page ("Just a moment...") | **not used directly**; a challenge is an access control and is not worked around. Many LoC FSA/OWI scans are mirrored on Wikimedia Commons with the LoC record and a US-government licence template, and arrive that way. |
| Smithsonian Open Access | not tried in the pilot: it needs an api.data.gov key the project does not hold; none of the pilot photographers is known to be held there with CC0 images | candidate for a later batch, after a key is requested by the owner |
| Wikimedia Commons (API) | works: category crawl + `imageinfo`/`extmetadata`, one request at a time, `maxlag`; scaled thumbnails from the file servers | main image source; licence class from the file's own template (`License`, `LicenseShortName`, `Copyrighted`) |
| The Met Open Access | `MetObjects.csv` (already local) marks almost only 19th-century photographs as public domain; essay pages answer HTTP 429 to automated reads | not used for the pilot photographers (Walker Evans: 1 public-domain object of 7,638; Atget 0 of 90; Stieglitz 0 of 265) |
| Art Institute of Chicago (API) | works; `is_public_domain` per artwork | Stieglitz (188 public-domain records), Clarence H. White, Atget (artist name "Jean-Eugène-Auguste Atget") |
| Museum essays / foundations for the Visual DNA research | Fondation HCB, Saul Leiter Foundation, ICP, Tate, Aperture, Grolier Club readable; MoMA 403/404/429, Met 429, Smithsonian 403, LensCulture 403, Oakland Museum 403 | primary sources where readable; Wikipedia as a secondary source otherwise, marked as such |

## Rights rules applied (per image)

- **Public domain**: the Commons file's licence template says so and `Copyrighted` is false (e.g. PD-USGov-FSA for
  FSA/OWI employees' work, PD-old-70 for authors dead more than 70 years, PD-US-expired). Commons requires a work to
  be free in the US and in its source country; this pipeline relies on that review and records the evidence URL. It
  is not an independent legal opinion.
- **CC0**: AIC `is_public_domain = true` (AIC releases those images under CC0) or a Commons CC0 template.
- **Licensed** (CC BY, CC BY-SA): metadata only. CC BY-SA is on the project's licence denylist; storing an embedding
  is not assumed to resolve share-alike or attribution questions.
- **Unverified / restricted**: metadata only.
- Attribution must name the photographer in the source's own record; portraits *of* a photographer, signatures,
  book covers, exhibition views, retouched or colourised versions, crops and details are rejected by title and
  category rules (`refdb/sources.py`).

## Photographers and what is legally usable

| photographer | in FilmAnalysis | image status | why |
|---|---|---|---|
| Dorothea Lange | yes | public domain subset (RA/FSA, BAE, WRA federal work) | US government work |
| Walker Evans | yes | public domain subset (~1,000 RA/FSA negatives) | US government work; the Met holds copyright to the rest |
| Lewis Hine | yes | public domain | died 1940; NCLC work |
| Eugène Atget | no | public domain | died 1927 |
| Alfred Stieglitz | no | public domain subset | pre-1930 publications; AIC CC0 flags |
| Jack Delano, Arthur Rothstein, John Vachon, Marion Post Wolcott, Russell Lee, Ben Shahn, Carl Mydans | no | public domain subset | FSA/OWI federal work; added as subject-matched controls |
| Gertrude Käsebier, Clarence H. White | no | public domain subset | early-20th-century prints; per-file templates |
| Henri Cartier-Bresson, Robert Capa, David Seymour, George Rodger | HCB yes | **restricted** | in copyright; Magnum / foundations; no open licence; Magnum images are never assumed public domain |
| Masahisa Fukase, Rinko Kawauchi, Saul Leiter, Daido Moriyama | yes | **restricted** | in copyright or living; knowledge only |

Consequence: the photographers the brief names first for their motifs (Fukase, Kawauchi, Leiter, Moriyama,
Cartier-Bresson) can be matched only through the Visual DNA knowledge graph, never through image similarity, until
licensed images exist (artist or foundation opt-in). Image similarity is evaluated on the public-domain set.
