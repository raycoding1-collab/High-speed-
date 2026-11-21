"""CR450AF livery + surface-detail texture painter (numpy + PIL).

1. A metric side-elevation atlas (X along the car, z above rail) is drawn with PIL
   from the livery shapes traced on the calibrated reference photo ref08.
2. Every texel of the car's UV space is mapped back to its 3D surface point
   (same parametrisation as trainshape/train) and the atlas is sampled there.

Outputs per car type two RGBA PNGs:
  *_col.png  : sRGB albedo, A = glass mask (1 side window, 0.5 windscreen, 0.25 headlight lens)
  *_dat.png  : R metallic, G panel gap, B dirt, A roughness   (non-colour)
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import specs as S
from . import trainshape as TS

PX = 0.0025          # atlas resolution (m/px), drawn at this, sampled bilinearly
ZMAX = 4.10
UV_PX = 0.005        # UV texture resolution along X (m/px)
HALF_ROWS = 1120     # texture rows per half section

FONT_CJK = os.environ.get("FONT_CJK", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
FONT_LAT = os.environ.get("FONT_LAT", "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf")

# colours (sRGB 0-255)
SILVER = (178, 181, 186)
BLACK = (16, 17, 20)
RED = (200, 18, 40)
ORANGE = (240, 92, 36)
WHITE = (236, 236, 232)
SKIRT = (48, 50, 54)
GOLD = (206, 160, 62)
GASKET = (20, 20, 22)

# material channels
MET_SILVER, MET_PAINT = 255, 0
R_SILVER, R_BLACK, R_PAINT, R_SKIRT, R_GLASS = 70, 40, 55, 150, 10


class Atlas:
    """Layered side-elevation drawing in metres."""

    def __init__(self, length):
        self.L = length
        self.W = int(math.ceil(length / PX)); self.H = int(math.ceil(ZMAX / PX))
        self.col = Image.new("RGB", (self.W, self.H), SILVER)
        self.met = Image.new("L", (self.W, self.H), MET_SILVER)
        self.rgh = Image.new("L", (self.W, self.H), R_SILVER)
        self.gls = Image.new("L", (self.W, self.H), 0)
        self.gap = Image.new("L", (self.W, self.H), 0)
        self.d = {k: ImageDraw.Draw(getattr(self, k)) for k in ("col", "met", "rgh", "gls", "gap")}

    def p(self, X, z):
        return (X / PX, (ZMAX - z) / PX)

    def poly(self, pts, col=None, met=None, rgh=None, gls=None):
        q = [self.p(*a) for a in pts]
        if col is not None: self.d["col"].polygon(q, fill=col)
        if met is not None: self.d["met"].polygon(q, fill=met)
        if rgh is not None: self.d["rgh"].polygon(q, fill=rgh)
        if gls is not None: self.d["gls"].polygon(q, fill=gls)

    def paint(self, pts, colour, metallic=MET_PAINT, rough=R_PAINT):
        self.poly(pts, col=colour, met=metallic, rgh=rough)

    def rrect(self, x0, z0, x1, z1, r, **kw):
        return rounded_rect(x0, z0, x1, z1, r)

    def line(self, pts, width_m, layer="gap", value=255):
        self.d[layer].line([self.p(*a) for a in pts], fill=value, width=max(1, int(round(width_m / PX))), joint="curve")

    def outline(self, pts, width_m=0.006, value=255):
        self.line(list(pts) + [pts[0]], width_m, "gap", value)

    def text(self, X, z, s, height_m, colour, font, mirror=False, italic_shear=0.0, anchor="ls"):
        """Draw text with baseline-left at (X, z) reading towards +X (or mirrored)."""
        size = max(8, int(height_m / PX))
        f = ImageFont.truetype(font, size)
        bbox = f.getbbox(s)
        w = bbox[2] - bbox[0] + 8; h = bbox[3] - bbox[1] + 8
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), s, fill=255, font=f)
        if italic_shear:
            img = img.transform((w + int(h * abs(italic_shear)), h), Image.AFFINE,
                                (1, italic_shear, -h * italic_shear if italic_shear > 0 else 0, 0, 1, 0), Image.BICUBIC)
        if mirror:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        px, py = self.p(X, z)
        pos = (int(px), int(py - h))
        if mirror:
            pos = (int(px - img.width), int(py - h))
        self.col.paste(Image.new("RGB", img.size, colour), pos, img)
        self.met.paste(MET_PAINT, pos, img)
        self.rgh.paste(R_PAINT, pos, img)


def rounded_rect(x0, z0, x1, z1, r, n=6):
    pts = []
    for cx, cz, a0 in ((x1 - r, z1 - r, 0), (x0 + r, z1 - r, 90), (x0 + r, z0 + r, 180), (x1 - r, z0 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    return pts


def interp_curve(pts, X):
    xs, zs = zip(*pts)
    return float(np.interp(X, xs, zs))


# ------------------------------------------------------------------ livery definitions
R1 = [(1.10, 1.58), (2.0, 1.78), (3.0, 2.08), (4.0, 2.38), (5.0, 2.66), (6.0, 2.90), (7.0, 3.08),
      (8.0, 3.22), (9.0, 3.34), (10.0, 3.45), (11.0, 3.56), (11.8, 3.70)]          # upper red line (canopy edge)
LB = [(1.15, 1.56), (2.0, 1.66), (2.6, 1.80), (4.0, 2.05), (5.5, 2.27), (7.0, 2.42)]  # lower edge of black side wedge
WS_LOW = [(2.05, 1.98), (3.0, 2.30), (4.0, 2.62), (5.0, 2.94), (5.35, 3.05)]      # windscreen lower edge
BAND_TOP_FRONT = [(7.0, 2.42), (8.0, 2.62), (9.0, 2.86), (9.6, 2.97), (10.2, 3.00)]
CANOPY_END_X = 9.4


def draw_end_car(side):
    """side: +1 left (text reads front->rear), -1 right."""
    L = S.END_CAR_LENGTH
    A = Atlas(L)
    mirror = side < 0
    # --- skirt band (dark) behind the nose fin
    A.paint([(5.0, -0.1), (6.3, 0.83), (L, 0.83), (L, -0.1)], SKIRT, rough=R_SKIRT)
    # --- black canopy (above R1) and black side wedge (between LB and R1)
    top = ZMAX
    canopy = [(x, z) for x, z in R1 if x <= CANOPY_END_X] + [(CANOPY_END_X, interp_curve(R1, CANOPY_END_X)), (CANOPY_END_X, top), (0.9, top), (0.9, 1.56)]
    A.paint(canopy, BLACK, rough=R_BLACK)
    wedge = LB + [(7.0, 2.42)] + [(x, z) for x, z in reversed(R1) if 1.1 <= x <= 7.0]
    A.paint(wedge, BLACK, rough=R_BLACK)
    # --- window band, with rising front edge
    band = [(7.0, 2.0)] + BAND_TOP_FRONT + [(L - 3.3, 3.0), (L - 3.3, 2.0)]
    A.paint(band, BLACK, rough=R_BLACK)
    # --- red front wedge + stripe
    A.paint([(5.10, 1.55), (7.0, 2.40), (7.0, 1.75)], RED)
    A.paint([(5.10, 1.55), (7.0, 1.75), (L, 1.75), (L, 1.905), (6.0, 1.905)], RED)
    A.paint([(6.0, 1.905), (L, 1.905), (L, 1.965), (6.6, 1.965)], ORANGE)
    A.paint([(6.6, 1.968), (L, 1.968), (L, 1.992), (7.0, 1.992)], WHITE, metallic=MET_PAINT, rough=R_PAINT)
    # thin red line along the rising band top and the wedge's upper edge
    A.d["col"].line([A.p(x, z) for x, z in [(5.10, 1.55), (7.0, 2.42)] + BAND_TOP_FRONT[1:]], fill=RED, width=int(0.035 / PX))
    # upper red line along R1
    A.d["col"].line([A.p(x, z - 0.02) for x, z in R1], fill=RED, width=int(0.045 / PX), joint="curve")
    A.d["met"].line([A.p(x, z - 0.02) for x, z in R1], fill=MET_PAINT, width=int(0.045 / PX))
    # silver separation line between side wedge and the band region (from photos: thin light line)
    A.d["col"].line([A.p(x, z) for x, z in LB], fill=(200, 202, 206), width=int(0.012 / PX))
    # --- rear end block: red diagonal + orange block with CR450 logo
    xb = L - 3.3
    A.paint([(xb, 1.75), (xb + 0.55, 3.0), (xb + 1.15, 3.0), (xb + 0.60, 1.75)], RED)
    A.paint([(xb + 0.60, 1.75), (xb + 1.15, 3.0), (L, 3.0), (L, 1.75)], ORANGE)
    A.paint([(xb + 0.60, 1.75), (xb + 0.86, 2.33), (L, 2.33), (L, 1.75)], (228, 64, 34))
    # --- windscreen (glass) and headlight lens
    ws = [(2.05, 1.98)] + WS_LOW[1:] + [(5.35, 3.05), (5.30, top), (2.6, top), (2.1, 2.3)]
    A.poly(ws, col=(10, 12, 14), met=MET_PAINT, rgh=R_GLASS, gls=128)
    A.outline(ws, 0.010)
    eye = [(2.95, 1.93), (5.45, 2.33), (5.42, 2.47), (3.15, 2.07)]
    A.poly(eye, col=(25, 25, 28), met=MET_PAINT, rgh=R_GLASS, gls=64)
    A.outline(eye, 0.008)
    # cab side window (in black area ahead of cab door)
    cw = rounded_rect(7.25, 2.10, 8.40, 2.62, 0.12)
    A.poly(cw, col=(12, 13, 15), rgh=R_GLASS, gls=255); A.outline(cw, 0.008)
    # --- windows: front group (business) and rear group
    for x0 in (10.40, 11.80, 13.20):
        w = rounded_rect(x0, 2.05, x0 + 1.0, 2.85, 0.10)
        A.poly(w, col=(14, 15, 17), rgh=R_GLASS, gls=255); A.outline(w, 0.008)
    for x0 in (18.30, 20.12, 21.94, 23.76):
        w = rounded_rect(x0, 2.05, x0 + 1.40, 2.85, 0.10)
        A.poly(w, col=(14, 15, 17), rgh=R_GLASS, gls=255); A.outline(w, 0.008)
    # --- doors
    cd = rounded_rect(S.CAB_DOOR_X - S.CAB_DOOR_W / 2, 1.40, S.CAB_DOOR_X + S.CAB_DOOR_W / 2, 3.37, 0.09)
    A.outline(cd, 0.008)
    A.outline(rounded_rect(S.CAB_DOOR_X - 0.11, 2.55, S.CAB_DOOR_X + 0.11, 2.95, 0.10), 0.006)   # door window
    pdx = S.PAX_DOOR_X_END
    pd = rounded_rect(pdx - S.DOOR_W / 2, 1.18, pdx + S.DOOR_W / 2, 3.42, 0.09)
    A.outline(pd, 0.008)
    pw = rounded_rect(pdx - 0.12, 2.35, pdx + 0.12, 2.95, 0.12)
    A.poly(pw, col=(14, 15, 17), rgh=R_GLASS, gls=255); A.outline(pw, 0.006)
    # --- bill / coupler cover seam, skirt seams, hatches
    A.outline([(1.15, 1.47), (1.32, 1.40), (1.42, 1.10), (1.40, 0.80), (1.30, 0.62), (0.2, 0.62), (0.2, 1.47)], 0.006)
    A.line([(6.3, 0.83), (L, 0.83)], 0.006)
    for xh in np.arange(7.6, L - 1.0, 2.35):
        A.line([(xh, 0.20), (xh, 0.82)], 0.005)
        A.d["col"].rectangle([A.p(xh + 0.25, 0.70), A.p(xh + 0.31, 0.66)], fill=(220, 190, 40))   # yellow latch marks
    A.outline(rounded_rect(4.30, 1.00, 4.80, 1.32, 0.04), 0.005)     # access hatch on nose fin
    A.outline(rounded_rect(9.55, 0.95, 9.95, 1.15, 0.03), 0.005)
    A.line([(L - 0.32, 0.18), (L - 0.32, 3.86)], 0.006)              # car end seam
    # --- text
    lat_font = FONT_LAT
    if not mirror:
        A.text(9.75, 1.44, "CR450AF-0201", 0.115, (22, 22, 24), FONT_LAT.replace("BoldOblique", "Bold"))
        A.text(L - 2.45, 2.30, "CR450", 0.36, WHITE, lat_font, italic_shear=0.0)
        A.text(20.4, 3.18, "复 兴 号", 0.30, GOLD, FONT_CJK)
        A.text(L - 7.5, 1.45, "ZYS 010201", 0.08, (22, 22, 24), FONT_LAT.replace("BoldOblique", "Bold"))
    else:
        A.text(11.0, 1.44, "CR450AF-0201", 0.115, (22, 22, 24), FONT_LAT.replace("BoldOblique", "Bold"), mirror=True)
        A.text(L - 0.45, 2.30, "CR450", 0.36, WHITE, lat_font, mirror=True)
        A.text(22.3, 3.18, "复 兴 号", 0.30, GOLD, FONT_CJK, mirror=True)
        A.text(L - 6.2, 1.45, "ZYS 010201", 0.08, (22, 22, 24), FONT_LAT.replace("BoldOblique", "Bold"), mirror=True)
    return A


def draw_mid_car(side, dining=False, car_no=2):
    L = S.MID_CAR_LENGTH
    A = Atlas(L)
    mirror = side < 0
    A.paint([(0, -0.1), (0, 0.83), (L, 0.83), (L, -0.1)], SKIRT, rough=R_SKIRT)
    # band
    A.paint([(0.0, 2.0), (0.0, 3.0), (L, 3.0), (L, 2.0)], BLACK, rough=R_BLACK)
    A.paint([(0, 1.75), (L, 1.75), (L, 1.905), (0, 1.905)], RED)
    A.paint([(0, 1.905), (L, 1.905), (L, 1.965), (0, 1.965)], ORANGE)
    A.paint([(0, 1.968), (L, 1.968), (L, 1.992), (0, 1.992)], WHITE)
    doors = [1.20] if dining else [1.20, L - 1.20]
    ends = [0] if dining else [0, 1]
    # end blocks: orange around the door, then two red diagonals leaning back
    for e in (0, 1):
        if e == 0:
            A.paint([(0, 1.75), (0, 3.0), (2.55, 3.0), (2.0, 1.75)], ORANGE)
            A.paint([(0, 1.75), (0, 2.33), (2.25, 2.33), (2.0, 1.75)], (228, 64, 34))
            A.paint([(2.0, 1.75), (2.55, 3.0), (3.05, 3.0), (2.50, 1.75)], RED)
            A.paint([(2.75, 1.75), (3.30, 3.0), (3.62, 3.0), (3.07, 1.75)], RED)
        else:
            xb = L - 3.3
            A.paint([(xb, 1.75), (xb + 0.55, 3.0), (xb + 1.15, 3.0), (xb + 0.60, 1.75)], RED)
            A.paint([(xb + 0.60, 1.75), (xb + 1.15, 3.0), (L, 3.0), (L, 1.75)], ORANGE)
            A.paint([(xb + 0.60, 1.75), (xb + 0.86, 2.33), (L, 2.33), (L, 1.75)], (228, 64, 34))
    for dx in doors:
        pd = rounded_rect(dx - S.DOOR_W / 2, 1.18, dx + S.DOOR_W / 2, 3.42, 0.09)
        A.outline(pd, 0.008)
        pw = rounded_rect(dx - 0.10, 2.30, dx + 0.10, 2.95, 0.10)
        A.poly(pw, col=(14, 15, 17), rgh=R_GLASS, gls=255); A.outline(pw, 0.006)
    # windows: second class, 1.15 m wide at 1.78 m pitch between the end blocks
    xs = np.arange(4.30, L - 4.0, 1.78)
    if dining:
        xs = np.arange(4.30, L - 2.0, 2.4)
    for x0 in xs:
        w = rounded_rect(x0, 2.05, x0 + (1.15 if not dining else 1.7), 2.85, 0.10)
        A.poly(w, col=(14, 15, 17), rgh=R_GLASS, gls=255); A.outline(w, 0.008)
    A.line([(0, 0.83), (L, 0.83)], 0.006)
    for xh in np.arange(1.1, L - 0.8, 2.35):
        A.line([(xh, 0.20), (xh, 0.82)], 0.005)
        A.d["col"].rectangle([A.p(xh + 0.25, 0.70), A.p(xh + 0.31, 0.66)], fill=(220, 190, 40))
    A.line([(0.32, 0.18), (0.32, 3.86)], 0.006); A.line([(L - 0.32, 0.18), (L - 0.32, 3.86)], 0.006)
    font_b = FONT_LAT.replace("BoldOblique", "Bold")
    tag = f"ZE 0{car_no}0201"
    if not mirror:
        A.text(L - 2.45, 2.30, "CR450", 0.36, WHITE, FONT_LAT)
        A.text(3.6, 1.45, tag, 0.08, (22, 22, 24), font_b)
        A.text(4.2, 3.18, "复 兴 号", 0.30, GOLD, FONT_CJK) if car_no in (4, 5) else None
    else:
        A.text(L - 0.45, 2.30, "CR450", 0.36, WHITE, FONT_LAT, mirror=True)
        A.text(L - 3.6, 1.45, tag, 0.08, (22, 22, 24), font_b, mirror=True)
        A.text(L - 4.2, 3.18, "复 兴 号", 0.30, GOLD, FONT_CJK, mirror=True) if car_no in (4, 5) else None
    return A


# ------------------------------------------------------------------ UV remap
def _sections_for(kind):
    if kind == "end":
        st = TS.stations_end_car()
        secs = np.array([TS.half_section(x) for x in st])
        return st, secs, S.END_CAR_LENGTH
    L = S.MID_CAR_LENGTH
    hs = TS.body_section()
    st = np.array([0.0, L])
    return st, np.array([hs, hs]), L


def _bilinear(img, x, y):
    """Sample HxW(xC) array at float pixel coords (x right, y down)."""
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None] if img.ndim == 3 else (x - x0)
    fy = (y - y0)[..., None] if img.ndim == 3 else (y - y0)
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def surface_grid(kind):
    """Return arrays X, Y, Z (H x W) of the surface point for every texel (row 0 = v 0)."""
    st, secs, L = _sections_for(kind)
    W = int(math.ceil(L / UV_PX)); H = 2 * HALF_ROWS
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    Xc = u * L
    nsec = TS.NSEC
    # per column section by linear interpolation along X
    colsec = np.empty((W, nsec, 2))
    for j in range(nsec):
        for k in range(2):
            colsec[:, j, k] = np.interp(Xc, st, secs[:, j, k])
    # row parameter t in [0, nsec-1]
    t = np.where(v < 0.5, v / 0.5, (1 - v) / 0.5) * (nsec - 1)
    side = np.where(v < 0.5, -1.0, 1.0)            # right half (y<0) then left
    j0 = np.clip(np.floor(t).astype(int), 0, nsec - 2); ft = t - j0
    Y = (colsec[:, j0, 0] * (1 - ft) + colsec[:, j0 + 1, 0] * ft).T * side[:, None]
    Z = (colsec[:, j0, 1] * (1 - ft) + colsec[:, j0 + 1, 1] * ft).T
    X = np.broadcast_to(Xc[None, :], Y.shape)
    return X, Y, Z, side


def bake(kind, outdir, name, dining=False, car_no=2, seed=1):
    os.makedirs(outdir, exist_ok=True)
    X, Y, Z, side = surface_grid(kind)
    H, W = X.shape
    rng = np.random.default_rng(seed)
    col = np.zeros((H, W, 3), np.float32); dat = np.zeros((H, W, 4), np.float32); gls = np.zeros((H, W), np.float32)
    for sgn, rows in ((-1, slice(0, H // 2)), (1, slice(H // 2, H))):
        A = draw_end_car(sgn) if kind == "end" else draw_mid_car(sgn, dining, car_no)
        g = A.gap.filter(ImageFilter.GaussianBlur(0.8))
        layers = np.dstack([np.asarray(A.col, np.float32) / 255,
                            np.asarray(A.met, np.float32)[..., None] / 255,
                            np.asarray(g, np.float32)[..., None] / 255,
                            np.asarray(A.rgh, np.float32)[..., None] / 255,
                            np.asarray(A.gls, np.float32)[..., None] / 255])
        px = X[rows] / PX; py = (ZMAX - Z[rows]) / PX
        smp = _bilinear(layers, px, py)
        col[rows] = smp[..., :3]; dat[rows, ..., 0] = smp[..., 3]; dat[rows, ..., 1] = smp[..., 4]
        dat[rows, ..., 3] = smp[..., 5]; gls[rows] = smp[..., 6]
    # underside (below skirt edge, facing down) -> dark
    under = (Z < 0.32) & (np.abs(Y) < 1.5)
    col[under] = np.array(SKIRT) / 255 * 0.6; dat[under, 0] = 0; dat[under, 3] = 0.8
    # ---- wear: road dirt gradient (heavier low and towards the rear of each bogie), noise streaks
    zz = np.clip((1.35 - Z) / 1.15, 0, 1) ** 1.6
    nx = _value_noise(W, H, 0.8, rng) * 0.6 + _value_noise(W, H, 0.15, rng) * 0.4
    streak = _value_noise(W, H, 0.05, rng, aniso=6.0)
    dirt = np.clip(zz * (0.35 + 0.65 * nx) + 0.12 * streak * zz, 0, 1)
    if kind == "end":
        dirt += np.clip((1.2 - X) / 1.2, 0, 1) * 0.15 * (Z < 1.6)
    dat[..., 2] = np.clip(dirt * 0.85, 0, 1) * (gls < 0.4)
    # ---- insect marks on forward-facing nose surfaces (end car only)
    if kind == "end":
        fwd = _forward_facing(X, Y, Z)
        bugs = _bug_field(X, Y, Z, fwd, rng)
        col = col * (1 - bugs[..., None] * 0.85) + bugs[..., None] * np.array([0.20, 0.17, 0.10]) * 0.85
        dat[..., 3] = np.maximum(dat[..., 3], bugs * 0.6)
    a_col = np.dstack([col, gls[..., None]])
    _save(a_col, os.path.join(outdir, f"{name}_col.png"))
    _save(dat, os.path.join(outdir, f"{name}_dat.png"))
    return os.path.join(outdir, f"{name}_col.png"), os.path.join(outdir, f"{name}_dat.png")


def _forward_facing(X, Y, Z):
    dX = np.gradient(X, axis=1); dY1 = np.gradient(Y, axis=1); dZ1 = np.gradient(Z, axis=1)
    dY0 = np.gradient(Y, axis=0); dZ0 = np.gradient(Z, axis=0); dX0 = np.zeros_like(X)
    a = np.stack([dX, dY1, dZ1], -1); b = np.stack([dX0, dY0, dZ0], -1)
    n = np.cross(a, b); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # front is -X (towards the tip); facing = |n.x| where surface normal points to -X
    return np.clip(np.abs(n[..., 0]) - 0.15, 0, 1) * (X < 7.0)


def _bug_field(X, Y, Z, fwd, rng):
    H, W = X.shape
    m = np.zeros((H, W), np.float32)
    ys, xs = np.nonzero(fwd > 0.25)
    if len(xs) == 0:
        return m
    n = min(9000, len(xs) // 12)
    pick = rng.choice(len(xs), n, replace=False)
    for k in pick:
        y, x = ys[k], xs[k]
        if rng.random() > fwd[y, x]:
            continue
        r = rng.choice([1, 1, 1, 2, 2, 3])
        ln = rng.integers(0, 6)
        y0, y1 = max(0, y - r), min(H, y + r + 1); x0, x1 = max(0, x - r), min(W, x + r + 1 + ln)
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], rng.uniform(0.35, 0.95))
    return m


def _value_noise(W, H, scale_m, rng, aniso=1.0):
    """Smooth value noise with feature size ~scale_m (in texture metres along X)."""
    gx = max(2, int(W * UV_PX / scale_m)); gy = max(2, int(H / (HALF_ROWS / 5.5) / (scale_m * aniso)))
    g = rng.random((gy + 1, gx + 1)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def _save(arr, path):
    a = np.clip(arr, 0, 1)
    a = (a[::-1] * 255 + 0.5).astype(np.uint8)       # row 0 of the PNG = top = v 1
    mode = "RGBA" if a.shape[-1] == 4 else "RGB"
    Image.fromarray(a, mode).save(path, optimize=False, compress_level=3)


if __name__ == "__main__":
    import sys, time
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/livery"
    t = time.time()
    bake("end", out, "end")
    bake("mid", out, "mid", car_no=2)
    print("baked in", round(time.time() - t, 1), "s")
