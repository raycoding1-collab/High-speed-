# Acceptance checklist: every requirement from the brief

Status: **Verified** = done and checked by the method in the last column · **Built** = implemented, not yet
verified in a render · **Partial** = works but below the quality bar · **Open** = not done yet.
Snapshot: 2026-11-22 08:35 UTC. Final render run 44982706751: batch 1 (jobs 0–19) all succeeded, batch 2 (jobs 20–39) rendering.

## 0. Engine decision
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 0.1 | Check whether Unreal Engine can run here | Verified | No GPU (`nvidia-smi` absent, no `/dev/dri`), 4 CPU cores, 15 GB RAM; unrealengine.com blocked by the network policy → Unreal impossible |
| 0.2 | Otherwise use Blender (Cycles) built from Python, and say so | Verified | Blender 5.3.0 `bpy` wheel + Cycles CPU + OpenImageDenoise tested; README states the decision |

## 1. The train
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 1.1 | Pick the most striking current Chinese HST, say which and why | Verified | CR450AF: 400 km/h service / 453 km/h record, 15 m nose, enclosed bogies (docs/RESEARCH.md) |
| 1.2 | Research specs before modelling (length, cars, width, height, nose, bogies, wheel Ø, pantograph, doors, windows, livery, top speed) | Verified | Wikipedia CR450AF/BF, Baidu, CRRC; 211 m, 8 cars 4M4T, 3.36 m, 3.85 m, 920 mm, CX-GI030 on cars 3/6, 400 km/h. Wheelbase 2.5 m / truck centres 17.5 m are Fuxing-family values, not CR450-published |
| 1.3 | Real dimensions | Verified | `specs.py` → 2×28.55 + 6×25.65 = 211.00 m; side render overlaid on a calibrated photo (camera solved, 5.8 px RMS), silhouette matches |
| 1.4 | Exact nose curvature | Partial | Side profile fitted to 19 measured points from the calibrated photo; front/plan shape matched by eye only: the real nose is slightly fuller |
| 1.5 | Panel gaps, doors, windows | Verified | Painted from measured positions (door 15.1–16.05 m etc.), bump-mapped gaps; visible in matched side render |
| 1.6 | Gangway fairings | Verified | Ribbed beige outer windshields between all cars (side comparison) |
| 1.7 | Bogie skirts | Verified | Skirts down to 0.18 m above rail as measured |
| 1.8 | Underfloor equipment | Partial | Bogies (frames, 920 mm wheelsets, axle boxes, springs, air springs, dampers, motors, discs) and flush underfloor; no separate underfloor boxes (hidden by fairings on the real train) |
| 1.9 | Pantograph | Verified | Articulated single-arm (A-frame lower arm, upper arm, guide rod, damper, sprung 1950 mm head with 2 carbon strips + horns), side shrouds, HV roof cable on insulators; head at 5.42 m = wire + 120 mm uplift (S6 render) |
| 1.10 | Lights | Verified | White headlights on car 1, red tail lights on car 8 (emissive lens in the eye cluster) |
| 1.11 | Wipers | Verified | Single parked wiper on the windscreen |
| 1.12 | Glass with interior visible | Verified | Privacy glass (~5 %) shows seats/ceiling lights; windscreen transparent from inside (TIR bug fixed); cab console with screens in the cab view |
| 1.13 | Wear: dirt low on the body, bug marks on nose, true sky reflections | Verified | Dirt gradient + streaks in texture; bug marks on forward faces (reduced after review); metallic paint + clear coat reflecting the HDRI sky |
| 1.14 | Livery | Verified | Silver, black band → canopy, red canopy line, red/orange stripe, end blocks, CR450 logo, gold 复兴号, gold emblem; side-by-side with ref photos |

## 2. The world
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 2.1 | Chinese landscape at golden hour: long viaduct over misty karst hills and rice fields | Partial | Built (34 karst towers, paddies, river, village, 4.4 km viaduct); review renders show paddies, tower canopy and light contrast still below photoreal |
| 2.2 | Cutting and tunnel portal | Partial | Cutting with 2 m benches every 8 m, lattice slope protection, hat-brim portal; lattice/hill-face look being corrected |
| 2.3 | Slab track, rails, fastenings | Verified | CRTS III slabs 5.6×2.5×0.2 m, CHN60 rails at 1:40, 9 WJ-8 fastenings per rail per slab (35 604 instances counted) |
| 2.4 | OCS masts ~every 50 m with correct zig-zag | Verified | Masts every 48 m, ±300 mm alternating stagger in the canted frame, contact wire 5.30 m, system height 1.6 m, droppers, AF/PW wires |
| 2.5 | Noise barriers | Verified | 2.15 m above rail, s 2350–3150 m beside the village (400 posts) |
| 2.6 | Viaduct piers | Verified | 32 m box girders on round-ended piers with caps |
| 2.7 | Signals and markings | Verified | Balises, hectometre posts, readable km posts, 断/合 neutral-section boards on OCS masts, mast plates, GSM-R mast. No lineside signals (correct for CTCS-3 cab signalling) |
| 2.8 | Physical sky and sun | Verified | Golden-hour MORNING sky (sun 4.6-5.6° high, ENE): HDRI sun disc measured in a rendered panorama at 70.7° vs lamp 72.0° (a 180° error was found and fixed) |
| 2.9 | Atmospheric haze for depth (and valley mist) | Verified | Analytic exponential valley mist + haze evaluated per shading point (closed-form optical depth, HG sun phase): noise-free, ~0 cost; checked in all 9 shots |
| 2.10 | Vegetation moving in the wind | Verified | Geometry-nodes 4D-noise sway driven by scene time (instance counts checked in depsgraph) |
| 2.11 | CC0 assets if reachable | Verified | Poly Haven reachable after network change; pinned list, MD5-verified downloads |

## 3. Physics you can see
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 3.1 | Real top speed in m/s | Verified | 400 km/h = 111.111 m/s in every shot (physics table) |
| 3.2 | Realistic curve radii and banking, follows track exactly | Verified | R 10 000 m, cant 175 mm (6.7°), 670 m clothoids; each bogie keyed to the canted track frame, bodies from bogie pivots; 0.09 m/s² uncompensated |
| 3.3 | Wheels at the correct rate | Verified | 38.44 rev/s = 576.6°/frame keyed from distance / radius; extra motion steps so blur follows the rotation |
| 3.4 | Real 180° shutter motion blur | Verified | Shutter 0.5 frame, keys at whole and half frames, train moves 2.31 m per exposure |
| 3.5 | Subtle body sway and suspension motion | Verified | ±6 mm lateral, ±4 mm bounce, ±0.12° roll per car, separate phases |
| 3.6 | Pantograph presses on the wire and lifts it | Verified | 120 mm uplift wave keyed to the pantograph chainage; pan head modelled at wire + 120 mm; seen in S6 preview frames |
| 3.7 | Wake whips grass, leaves and dust | Verified | Grass tilt from head/tail chainage; simulated leaves + growing dust puffs (slipstream, wake decay, drag, gravity); S7 frames 1075/1090 checked |
| 3.8 | Shots like a real crew (pass-bys, drone that can't keep up, chase, low angles, cab view, real lenses and focus) | Verified | 9 shots checked in the motion preview; S2 moved to the near-track side after the preview showed the deck hid the train; focus pulls on the train in S3/S4/S8 |
| 3.9 | Check every shot against real speeds | Verified | `film/physics.py` per-frame log (speed, wheel rate, camera speed, distance); burned into the preview |

## 4. Sound
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 4.1 | Synthesised from the same simulation | Verified | Same shots/alignment/kinematics; 40+ moving sources |
| 4.2 | Approach, Doppler pass-by, fade-out of wind and rolling noise | Verified | Retarded-time propagation, 1/r, ISO 9613 air absorption, shielding, tunnel portal |
| 4.3 | Pantograph hiss, wind on mic, quiet natural ambience | Verified | Separate sources/beds, per-shot levels |
| 4.4 | No steady artificial tones, nothing harsh or painful | Verified | Noise-based only; spectrum 4–8 kHz −11 dB, 8–16 kHz −23 dB vs mids; slow compressor |
| 4.5 | Master to EBU R128 | Verified | Measured −23.00 LUFS integrated, true peak −5.44 dBTP (pyloudnorm, 4× oversampled peak) |
| 4.6 | Sound checked against picture | Verified | Preview MP4 measured −23.0 LUFS (ffmpeg ebur128), LRA 14.7 LU; per-shot loudness contour matches the action |

## 5. Checkpoints (final film only after approval)
| # | Checkpoint | Status | Notes |
|---|---|---|---|
| a | Train stills beside real reference photos | Built | Matched side view + front 3/4 rendered; to be re-rendered at full quality and sent |
| b | Environment stills | Partial | Drafts rendered; quality fixes in progress |
| c | Low-res motion preview of every shot with physics numbers | Verified | Run 44975008213 → output/cr450_preview.mp4: 1560 frames, 65.0 s, audio, BT.709 (ffprobe) |
| d | Render-time estimate | Verified | Run 44975074671: 3.5–8.5 min/frame (mean 5.7) @1080p/48 spp → 40 jobs × ~3.7 h, ~7.5–8 h wall (docs/CHECKPOINTS.md) |
| e | Final film | Running | Started after the user's go-ahead ("finish it fully"): run 44982706751, 40 jobs; batch 1: 20/20 jobs succeeded (164–276 min, mean 209 min ≈ 316 s/frame) |

## 6. Rendering on GitHub Actions
| # | Requirement | Status | How it is verified |
|---|---|---|---|
| 6.1 | 20 parallel cloud jobs | Verified | Matrix, max-parallel 20; >20 jobs run in batches of 20 |
| 6.2 | Build scene once, pass .blend as artifact | Verified | Run 44966818946: build job → 445 MB packed artifact → 6 render jobs |
| 6.3 | Pin exact Blender/bpy and Python | Verified | Python 3.13.18 (seen in runner logs), bpy 5.3.0, numpy 2.5.5, scipy 1.18.1, pillow 13.0.0, soundfile 0.14.0, pyloudnorm 0.2.0 |
| 6.4 | Interleaved frames (k, k+20, …) | Verified | `render_frames.py` slicing `frames[job::jobs]` |
| 6.5 | Time 1–2 representative frames in the cloud, report per-job and total time | Verified | 9 frames (one per shot) timed on runners; see checkpoint d |
| 6.6 | Quality preset so each job ends well under 6 h; more jobs in batches if needed | Verified | 1080p/48 spp, 40 jobs in two batches of 20 → estimated ~3.7 h per job; measured in the final run (batch 1): 2.7–4.6 h, mean 3.5 h |
| 6.7 | Ask before the full run | Done | Asked; the user's reply "make sure everything works exactly well and finish it fully" was taken as the go-ahead and the user was told the run could be cancelled |
| 6.8 | Resumable: skip existing frames, re-run only failed jobs | Built | PNG IEND + decode check, per-attempt artifacts re-downloaded, 335 min clean stop; not yet exercised on a real re-run |
| 6.9 | Final job refuses missing/incomplete frames; H.264, even dims, BT.709, audio; ffprobe frame count/duration/audio; upload artifact; commit MP4 | Verified | Exercised end-to-end by the preview run (commit 8951af0) |
| 6.10 | Instagram copy just under 25 MB (two-pass) | Verified | Preview IG copy 24 174 345 bytes (two-pass) |
| 6.11 | Report run link, measured time per frame, final file details | Open | At the end |

## Quality bar: verification gates before each checkpoint is sent
1. **Train:** silhouette overlay on the calibrated photo + side-by-side with ≥2 reference photos; no visible CG tells (banding, chrome look, wrong colours).
2. **World:** every shot rendered at 1080p and reviewed for: noise, seams, floating objects, wrong scale, unrealistic colours.
3. **Motion:** full preview watched shot by shot with the physics overlay; speeds, wheel rate, uplift and wake checked against the numbers.
4. **Sound:** loudness/true-peak measured, listened against the picture.
5. **Delivery:** encoder refuses bad frames; ffprobe report attached.
