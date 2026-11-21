"""Published / measured specifications used throughout the film.

Every number here is either a published figure (source noted) or a measurement
taken from reference photographs with a calibrated perspective fit
(see docs/RESEARCH.md).  Units: metres, seconds, kilograms unless noted.
"""
import math

# ---------------------------------------------------------------- train: CR450AF
# Wikipedia "China Railway CR450AF" (infobox + coach table), CRRC press release 2024-12-29
TRAIN_NAME = "CR450AF Fuxing"
N_CARS = 8
FORMATION = ["T", "M", "T", "M", "M", "T", "M", "T"]   # 4M4T, cars 2/4/5/7 powered
PANTO_CARS = [3, 6]                                   # pantographs on trailer cars 3 and 6
END_CAR_LENGTH = 28.55                                # ZYS cars 1 & 8
MID_CAR_LENGTH = 25.65                                # cars 2..7
TRAIN_LENGTH = 2 * END_CAR_LENGTH + 6 * MID_CAR_LENGTH  # = 211.0 m (published 211 m)
BODY_WIDTH = 3.36                                     # published
BODY_HEIGHT = 3.85                                    # 200 mm lower than CR400AF (4.05 m)
NOSE_LENGTH = 15.0                                    # published "15 m streamlined head"
WHEEL_DIAMETER = 0.920                                # Fuxing family standard (new wheel)
BOGIE_WHEELBASE = 2.500
TRUCK_CENTRES = 17.500
AXLE_LOAD_T = 14.5
SERVICE_SPEED_KMH = 400.0                             # design operating speed
DESIGN_SPEED_KMH = 450.0
RECORD_SPEED_KMH = 453.0
FLOOR_HEIGHT = 1.20                                   # measured (door sill) from photos

# measured from calibrated photo (ref08), X from nose tip, z above top of rail
NOSE_TIP_Z = 1.00
WINDOW_Z = (2.05, 2.85)
WINDOW_W = 1.00
WINDOW_PITCH = 1.40
BAND_Z = (2.00, 3.00)            # black window band
STRIPE_Z = (1.75, 1.98)          # red/orange stripe
SKIRT_BAND_TOP = 0.83            # dark grey skirt band
SKIRT_BOTTOM = 0.18
DOOR_Z = (1.20, 3.42)
DOOR_W = 0.95
CAB_DOOR_X = 9.02                # centre, from tip
CAB_DOOR_W = 0.66
PAX_DOOR_X_END = 15.58           # centre, from tip (end car)
END_FRONT_BOGIE_X = 7.20         # first bogie centre from nose tip

# ---------------------------------------------------------------- motion
SPEED = SERVICE_SPEED_KMH / 3.6                       # 111.11 m/s
WHEEL_REV_PER_S = SPEED / (math.pi * WHEEL_DIAMETER)  # 38.44 rev/s

# ---------------------------------------------------------------- infrastructure
# TB 10621-2014 Code for Design of High Speed Railway / TB 10009 traction power
GAUGE = 1.435
RAIL_HEAD_W = 0.073               # CHN60 rail head width (73 mm)
RAIL_H = 0.176                    # CHN60 height
RAIL_FOOT_W = 0.150
RAIL_INCLINATION = 1 / 40
TRACK_CENTRES = 5.0               # 350 km/h class; 400 km/h lines use >= 5.0 m
SLAB_L, SLAB_W, SLAB_T = 5.60, 2.50, 0.20   # CRTS III track slab
SLAB_GAP = 0.07
FASTENERS_PER_SLAB = 9            # WJ-8B, ~630 mm spacing
CONTACT_WIRE_HEIGHT = 5.30        # above top of rail
SYSTEM_HEIGHT = 1.60              # messenger above contact wire at supports
STAGGER = 0.30                    # +/- 300 mm zig-zag
MAST_SPACING_VIADUCT = 48.0       # 1.5 x 32 m girders ("about every 50 m")
GIRDER_SPAN = 32.0                # standard simply supported PC box girder
GIRDER_TOP_W = 12.6
GIRDER_DEPTH = 3.05
NOISE_BARRIER_H = 2.15            # above rail
CURVE_RADIUS = 10000.0            # m, for 400 km/h
CANT = 0.175                      # m (max 175 mm)
TRANSITION_L = 670.0              # clothoid length

if __name__ == "__main__":
    print(f"train length {TRAIN_LENGTH:.2f} m, speed {SPEED:.3f} m/s, wheel {WHEEL_REV_PER_S:.2f} rev/s")
    eq = 11.8 * SERVICE_SPEED_KMH ** 2 / CURVE_RADIUS
    print(f"equilibrium cant at R={CURVE_RADIUS:.0f}: {eq:.1f} mm, applied {CANT*1000:.0f} mm, deficiency {eq-CANT*1000:.1f} mm")
