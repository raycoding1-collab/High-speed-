# v3 verification matrix (gate checks before the full render)

Every change was tested on Modal (5-10 frames, full scene build, 1920x1080, 48 spp, stills committed to
`output/tests/<run>/`). Latest complete gate run: [45117447575](https://github.com/raycoding1-collab/High-speed-/actions/runs/45117447575).

| Gate | Status | Method | Measured evidence |
|---|---|---|---|
| Every instanced asset renders (no NaN transforms) | PASS | `tools/verify_scene.py`: evaluated depsgraph at the middle frame of every shot | 405,141 instances per shot, 0 invalid (`verify_scene.json`) |
| All textures available on the render machines | PASS | same: every FILE image packed or on disk | 149 / 149 packed, 0 missing |
| Worldbuilding present | PASS | object counts at build + visual check in S1, S2, S3, S4, S6 frames | 251 houses in hamlets placed in the camera views, power lines, fish ponds, straw stacks, greenhouses, citrus orchards, tombs, ~17,000 grove trees, ~1,300 bamboo clumps, ~100,000 crowns + conifers on the hill massif, 102,000 grass tufts in the cutting |
| Exposure / clipping | PASS | `tools/verify_frames.py` on the full-res frames | clipped <= 0.91 % (S9 sunlit roof), crushed <= 9.2 % (S5 cab console, intended), mean luma 0.23-0.43 |
| Shadow colour neutrality | PASS | mean (G+B)/2 - R of the darkest 25 % of pixels | -7.9 ... +4.5 (limit +8) |
| Loudness | PASS | pyloudnorm BS.1770 integrated + 4x oversampled true peak | -23.00 LUFS, -4.71 dBTP, 71.67 s (`verify_audio.json`) |
| Film length / edit | PASS | shot timeline | 1720 frames = 71.67 s @ 24 fps, 10 shots (S1-S9 + SX finale) |
| Render cost | measured | Modal L40S per-frame times (`timing_modal.json`) | 85-97 s/frame on GPU -> ~7-8 min/frame on a 4-core runner -> 60 jobs in 3 batches of 20 |

## Issues found in the test frames and fixed (v3 rounds 1-7)
| Seen in | Problem | Root cause | Fix |
|---|---|---|---|
| S8 | train hidden behind trees / black frame | tree bases below the long-lens sightline grew into it | sightline mask samples the whole shot and the whole tree height (0-18 m) |
| S9 | slow-motion beat looked at grass | ramp timed before the nose arrived | ramp retimed to the nose approach, push-in camera |
| S7 | slow-motion beat too dark | bodyside in cutting shade | +1.3 EV exposure ride during the whip-pan |
| S6 | roof highlights clipped | sun on the white roof | exposure -1.6 EV |
| S3, S5 | pale "scar" on the hill over the portal | the 1:1 portal face was faded back to the natural hill over 20 m: a 120 m vertical wall lit by the low western sun | face runs on (benched 1:1, hydroseeded) until it meets the hill; the face can only cut, never fill; the corridor strip blends into the far terrain at its edge (was a 28 m step) |
| S3 | tunnel massif read as a grey gravel heap | Poly Haven pine twig cards are 93 % grey-brown | needles toned to Masson-pine green, fir slightly greener |
| SX | slope over the portal a smooth green plane | corridor-wide scatters capped, little landed on the face | dense shrubs, ferns and grass on the visible part of the slope |
