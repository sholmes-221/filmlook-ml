"""Map encyclopedia species (research/species taxa ids) onto the website's 30 badge species ids (filmlook/species.py).

The website's album badges name coarse groups ("sparrow", "rose", 白鹭, 牡丹), not species. A group's score is the summed
softmax probability of its member classes, thresholded and validated per group (evaluate_groups in evaluate_badges.py).
Membership follows the Chinese badge name, which is the stricter reading:

- peony 牡丹 = Paeonia suffruticosa only (P. lactiflora is 芍药);
- egret 白鹭 = Egretta garzetta only (a grey heron is not an egret);
- gull 海鸥 = gulls and the kittiwake, not terns (燕鸥);
- duck 鸭子 = ducks incl. the mandarin and merganser, not geese; mandarin_duck 鸳鸯 = Aix galericulata;
- kingfisher 翠鸟 = the four Alcedinidae in the encyclopedia; magpie 喜鹊 = Pica and the azure-winged magpie (灰喜鹊);
- water_lily 睡莲 = Nymphaea only (Victoria is 王莲, Nuphar 萍蓬草);
- cherry_blossom 樱花 = Prunus serrulata, × yedoensis, campanulata; plum_blossom 梅花 = P. mume; peach_blossom 桃花 = P. persica.

No encyclopedia species exist for peacock, swan, swallow, oriole or a true pigeon (the spotted dove is 珠颈斑鸠), so those
badges stay unsupported by local recognition.
"""
from __future__ import annotations

GROUPS: dict[str, tuple[str, ...]] = {
    # birds
    "crane": ("red_crowned_crane", "siberian_crane", "demoiselle_crane"),
    "duck": ("mallard", "mandarin_duck", "scaly_sided_merganser", "cotton_pygmy_goose", "king_eider", "ruddy_shelduck"),
    "mandarin_duck": ("mandarin_duck",),
    "egret": ("little_egret",),
    "gull": ("black_headed_gull", "relict_gull", "vega_gull", "lesser_black_backed_gull", "iceland_gull", "ivory_gull", "black_legged_kittiwake"),
    "kingfisher": ("common_kingfisher", "white_throated_kingfisher", "stork_billed_kingfisher", "collared_kingfisher"),
    "magpie": ("oriental_magpie", "azure_winged_magpie"),
    "owl": ("long_eared_owl", "little_owl", "eurasian_eagle_owl", "snowy_owl", "buffy_fish_owl", "tawny_fish_owl", "sunda_scops_owl",
            "collared_scops_owl", "oriental_scops_owl", "spotted_wood_owl", "himalayan_owl", "eastern_barn_owl", "eastern_grass_owl"),
    "sparrow": ("eurasian_tree_sparrow", "rock_sparrow"),
    # flowers
    "bougainvillea": ("bougainvillea", "bougainvillea_spectabilis"),
    "cherry_blossom": ("yoshino_cherry", "prunus_serrulata", "prunus_campanulata"),
    "chrysanthemum": ("florists_chrysanthemum", "chrysanthemum_indicum"),
    "hydrangea": ("hydrangea",),
    "lavender": ("lavandula_angustifolia",),
    "lilac": ("syringa_oblata",),
    "lotus": ("sacred_lotus",),
    "magnolia": ("magnolia_denudata", "magnolia_liliiflora"),
    "peach_blossom": ("prunus_persica",),
    "peony": ("paeonia_suffruticosa",),
    "plum_blossom": ("japanese_apricot",),
    "rapeseed": ("brassica_napus",),
    "rose": ("rosa_multiflora", "rosa_rugosa", "rosa_chinensis", "rosa_banksiae"),
    "sunflower": ("sunflower",),
    "tulip": ("tulipa_gesneriana",),
    "water_lily": ("nymphaea_tetragona", "nymphaea_nouchali"),
}
UNSUPPORTED = ("peacock", "pigeon", "swallow", "swan", "oriole")
