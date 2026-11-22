# Research notes: CR450AF and Chinese high-speed infrastructure

## Why the CR450AF
The CR450AF *Fuxing* (CRRC Qingdao Sifang, revealed 29 Dec 2024) is the most striking current Chinese
high-speed train: 400 km/h design operating speed (450 km/h design, 453 km/h record), a 15 m streamlined
nose, fully enclosed bogies, 10 % lighter and 22 % lower running resistance than the CR400AF. The prototype
CR450AF-0201 completed its 600 000 km operational assessment in 2026. The film runs it at its design
operating speed of **400 km/h = 111.11 m/s** on a line built for that speed.

## Train specification used
| Item | Value | Source |
|---|---|---|
| Formation | 8 cars, 4M4T (motor cars 2/4/5/7) | Wikipedia CR450AF (coach table) |
| Length | 211 m: end cars 28.55 m, middle cars 25.65 m | Wikipedia CR450AF |
| Width | 3.36 m | Wikipedia |
| Height | 3.85 m (200 mm lower than CR400AF's 4.05 m) | Baidu Baike / china-emu.cn |
| Nose | ~15 m streamlined head | CRRC press, news reports |
| Wheels | 920 mm new (Fuxing family standard) | zh.wikipedia Fuxing |
| Bogies | H-frame bolsterless, fully enclosed by skirts; wheelbase 2.5 m, truck centres 17.5 m | Wikipedia / CRRC |
| Pantographs | on trailer cars 3 and 6 (CX-GI030), trailing one raised | zh.wikipedia, Baidu |
| Axle load | 14.5 t | Wikipedia |
| Livery | metallic silver, black window band sweeping into a black canopy, thin red canopy line, red/orange stripe with white pinstripe, orange/red end blocks with italic "CR450", gold 复兴号 lettering, gold China Railway emblem on the nose | Commons reference photographs |

### Measured from photographs
A side photograph (Commons: *CR450AF-0201 at Beijingdong Railway Station*) was calibrated with a
projective fit (rail and car-height scale; residual < 0.02 px on the fit points) to read positions in
metres. Key results, X measured from the nose tip, z above top of rail:

- nose tip 1.00 m; bill lip 1.44 m; nose top profile 2.05 m at X = 2 m, 2.95 m at X = 4 m, 3.50 m at X = 6 m, 3.85 m from X ≈ 11 m
- windscreen X = 2.1–5.3 m; cab door centre X = 9.0 m; first passenger door X = 15.6 m
- windows z = 2.05–2.85 m (business 1.0 m wide at 1.4 m pitch); black band z = 2.0–3.0 m
- red/orange stripe z = 1.75–1.98 m; dark skirt band up to 0.83 m; skirt bottom 0.18 m above rail

The 3D nose is a loft of cross-sections driven by longitudinal rails fitted to these numbers. A solved camera
for the same photo (OpenCV, 5.8 px RMS) is used to render the model from the photographer's exact viewpoint.

## Infrastructure (TB 10621-2014 high-speed design code and common practice)
- Standard gauge 1435 mm, CHN60 rails at 1:40 inclination, CRTS III slab track (5.6 × 2.5 × 0.2 m slabs, 9 WJ-8 fastenings per rail per slab)
- Track centres 5.0 m, left-hand running (China)
- 32 m simply supported PC box girders (12.6 m deck, 3.05 m deep) on round-ended piers
- OCS: contact wire 5.30 m above rail, system height 1.6 m, stagger ±300 mm alternating at every support,
  H-beam masts every 48 m on viaducts (1.5 girders, "about every 50 m"), droppers ~7 per span, feeder (AF) and protection (PW) wires
- Horizontal curve for 400 km/h: R = 10 000 m, cant 175 mm, 670 m clothoids.
  Equilibrium cant 189 mm → cant deficiency 14 mm, uncompensated lateral acceleration 0.09 m/s²
- Noise barriers 2.15 m above rail near settlements; deep cuttings with 2 m benches every 8 m height and diamond-lattice slope protection; hat-brim oblique tunnel portal

## Physics checks
| Quantity | Value |
|---|---|
| speed | 400 km/h = 111.11 m/s |
| wheel rotation (920 mm) | 38.44 rev/s = 576.6° per frame at 24 fps |
| 180° shutter | 1/48 s, train moves 2.31 m during exposure |
| pass-by of the 211 m train | 1.90 s |
| masts passing in the cab view | every 0.43 s (48 m) |
| pantograph contact-wire uplift | 120 mm at the head, wave ahead (9 m) / behind (22 m) |
| drone (S3) | 26 m/s max (DJI Inspire 3) → overtaken at 85 m/s |
| helicopter (S4) | 70 m/s (AS350 cruise) → overtaken at 41 m/s |

## Assets
CC0 assets from Poly Haven (pinned, MD5-verified by `tools/fetch_assets.py`): sky `kloppenheim_06_puresky`
(sun 4.6° above horizon), textures `concrete_wall_008`, `precast_concrete_wall`, `rock_face_03`,
`marble_cliff_02`, `forest_ground_04`, `sparse_grass`, `brown_mud_02`, `gravel_floor_02`, models
`island_tree_01/02/03`, `tree_small_02`, `shrub_01/02/04`, `fern_02`, `grass_medium_01/02`, `weed_plant_02`.

## Sources
- https://en.wikipedia.org/wiki/China_Railway_CR450AF
- https://en.wikipedia.org/wiki/China_Railway_CR450BF
- https://commons.wikimedia.org/wiki/Category:China_Railway_CR450AF (reference photographs)
- https://baike.baidu.com/item/复兴号CR450AF型电力动车组 (height, pantograph type)
- https://www.china-emu.cn/Trains/Model/Detail-9601-101-S.html
- https://crrcgc.cc/en/2025-11/06/article_2025110609294916177.html (453 km/h runs)
- https://www.caixinglobal.com/2026-08-10/china-takes-worlds-fastest-train-one-step-closer-to-market-102472951.html
- CRTS III slab and WJ-8 fastening dimensions: PMC11990785, sciopen High-speed Railway 2 (2024) 133-142
