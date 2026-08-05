# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Geophysical reasoning templates: multi-step physics/math computations
grounded in Wikidata entity properties (P2044, P625, P2048).

Each template has 3-5 reasoning steps. Questions describe a practical scenario
without naming the formula — the solver must determine the approach.

Template categories:
  - Solar/Astronomical (4): daylight, shadow, pendulum drift, solar noon
  - Atmospheric/Thermal (4): boiling time, terminal velocity, geothermal, wind chill
  - Geometric/Geodetic (4): visible horizon area, gravity train, line of sight, Coriolis
  - Crustal/Geophysical (3): isostatic root, seismic P-wave, free-air anomaly
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Constants
# ===========================================================================

# Physical constants used across templates
R_EARTH = 6371000       # Earth's mean radius in metres
R_EARTH_KM = 6371       # Earth's mean radius in km
G_SEA = 9.81            # Gravitational acceleration at sea level (m/s²)
OMEGA_EARTH = 7.2921e-5 # Earth's angular velocity (rad/s)
LAPSE_RATE = 6.5        # Standard tropospheric lapse rate (°C/km)
SEA_LEVEL_TEMP = 15.0   # ISA standard sea-level temperature (°C)

# Declination of the sun on June 21 (summer solstice) in degrees
SOLSTICE_DECLINATION = 23.44

# Reference cities for 2-entity templates
REFERENCE_CITIES_GEO = {
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
    "Cairo": (30.0444, 31.2357),
    "São Paulo": (-23.5505, -46.6333),
}


# ===========================================================================
# Geophysical Templates
# ===========================================================================

GEOPHYSICAL_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # Solar / Astronomical (4 templates)
    # -----------------------------------------------------------------------

    "daylight_hours_solstice": {
        "category": "solar",
        "label": "Daylight Hours on Summer Solstice",
        "type": "single",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "lat = {latitude}\n"
            "dec = math.radians(23.44)\n"
            "lat_r = math.radians(lat)\n"
            "cos_ha = -math.tan(lat_r) * math.tan(dec)\n"
            "cos_ha = max(-1.0, min(1.0, cos_ha))\n"
            "ha = math.acos(cos_ha)\n"
            "daylight = 2 * math.degrees(ha) / 15\n"
            "print(round(daylight, 2))"
        ),
        "question_hint": "How many hours of daylight does this city experience on June 21?",
        "answer_unit": "hours",
        "description": (
            "Steps: (1) compute solar declination on June 21 (23.44°), "
            "(2) compute hour angle cos(ha) = -tan(lat)*tan(dec), "
            "(3) daylight hours = 2*ha/15."
        ),
    },

    "shadow_length_solstice": {
        "category": "solar",
        "label": "Shadow Length at Noon on Summer Solstice",
        "type": "single",
        "entity_type": "structure",
        "required_properties": ["P2048", "P625"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "h = {height}\n"
            "lat = {latitude}\n"
            "dec = 23.44\n"
            "elev_angle = 90 - abs(lat - dec)\n"
            "shadow = h / math.tan(math.radians(elev_angle))\n"
            "print(round(shadow, 2))"
        ),
        "question_hint": (
            "How long (in metres) is this structure's shadow at solar noon "
            "on the summer solstice (June 21)?"
        ),
        "answer_unit": "metres",
        "description": (
            "Steps: (1) solar declination = 23.44°, "
            "(2) solar elevation = 90 - |lat - dec|, "
            "(3) shadow = height / tan(elevation)."
        ),
    },

    "pendulum_clock_drift": {
        "category": "solar",
        "label": "Pendulum Clock Drift at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 4,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "R = 6371000\n"
            "g0 = 9.81\n"
            "g_h = g0 * (R / (R + h)) ** 2\n"
            "# Period ratio: T_h/T_0 = sqrt(g0/g_h)\n"
            "ratio = math.sqrt(g0 / g_h)\n"
            "# Seconds lost per day = 86400 * (ratio - 1)\n"
            "drift = 86400 * (ratio - 1)\n"
            "print(round(drift, 2))"
        ),
        "question_hint": (
            "A pendulum clock calibrated at sea level is taken to this mountain's "
            "summit. How many seconds does it lose per day?"
        ),
        "answer_unit": "seconds",
        "description": (
            "Steps: (1) g at altitude = g0*(R/(R+h))², "
            "(2) period ratio = sqrt(g0/g_h), "
            "(3) drift = 86400*(ratio - 1)."
        ),
    },

    "solar_noon_time": {
        "category": "solar",
        "label": "Solar Noon Clock Time from Longitude",
        "type": "single",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "code_template": (
            "lon = {longitude}\n"
            "std_meridian = {standard_meridian}\n"
            "offset_deg = lon - std_meridian\n"
            "offset_min = offset_deg * 4\n"
            "solar_noon_min = 720 + offset_min\n"
            "print(round(solar_noon_min, 1))"
        ),
        "question_hint": (
            "If standard time is set to the {standard_meridian}° meridian, "
            "at what time (in minutes after midnight) does solar noon occur "
            "at this city's longitude?"
        ),
        "answer_unit": "minutes after midnight",
        "description": (
            "Steps: (1) offset = longitude - standard meridian, "
            "(2) offset in minutes = offset_deg * 4, "
            "(3) solar noon = 720 + offset (minutes after midnight)."
        ),
    },

    # -----------------------------------------------------------------------
    # Atmospheric / Thermal (4 templates)
    # -----------------------------------------------------------------------

    "boiling_time_at_altitude": {
        "category": "atmospheric",
        "label": "Time to Boil Water at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 5,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "P0 = 101325; M_air = 0.029; g = 9.81; R_gas = 8.314; T_avg = 288.15\n"
            "P = P0 * math.exp(-M_air * g * h / (R_gas * T_avg))\n"
            "T_b = 373.15; L = 2260000; R_w = 461.5\n"
            "dT = (R_w * T_b**2 / L) * math.log(P / P0)\n"
            "boil_C = 100 + dT\n"
            "E = {water_mass} * 4186 * (boil_C - {start_temp})\n"
            "print(round(E / {power}, 1))"
        ),
        "question_hint": (
            "Using a {power} W heater, how many seconds does it take to bring "
            "{water_mass} kg of water from {start_temp}°C to a boil at this "
            "mountain's summit?"
        ),
        "answer_unit": "seconds",
        "description": (
            "Steps: (1) barometric pressure P, (2) Clausius-Clapeyron boiling "
            "point, (3) boil_C = 100 + dT, (4) energy = m*c*(Tb-T0), "
            "(5) time = E/power."
        ),
    },

    "terminal_velocity_at_altitude": {
        "category": "atmospheric",
        "label": "Terminal Velocity at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 4,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "P0 = 101325; M_air = 0.029; g = 9.81; R_gas = 8.314; T_avg = 288.15\n"
            "P = P0 * math.exp(-M_air * g * h / (R_gas * T_avg))\n"
            "rho = P * M_air / (R_gas * T_avg)\n"
            "m = {skydiver_mass}; Cd = {drag_coefficient}; A = {cross_section_area}\n"
            "v_t = math.sqrt(2 * m * g / (rho * Cd * A))\n"
            "print(round(v_t, 2))"
        ),
        "question_hint": (
            "A {skydiver_mass} kg skydiver (drag coefficient {drag_coefficient}, "
            "cross-sectional area {cross_section_area} m²) jumps at this mountain's "
            "summit. What is their terminal velocity (m/s)?"
        ),
        "answer_unit": "m/s",
        "description": (
            "Steps: (1) barometric pressure, (2) air density ρ = PM/(RT), "
            "(3) terminal v = sqrt(2mg/(ρCdA))."
        ),
    },

    "geothermal_boiling_depth": {
        "category": "atmospheric",
        "label": "Depth to Underground Boiling",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "code_template": (
            "h = {elevation}\n"
            "T_surface = 15 - 6.5 * h / 1000\n"
            "geothermal_gradient = {geothermal_gradient}\n"
            "depth_km = (100 - T_surface) / geothermal_gradient\n"
            "print(round(depth_km, 2))"
        ),
        "question_hint": (
            "Assuming a geothermal gradient of {geothermal_gradient}°C/km, how deep "
            "(in km) below this mountain's base would you need to drill before "
            "underground water begins to boil?"
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) surface temp from lapse rate T_s = 15 - 6.5*h/1000, "
            "(2) depth = (100 - T_s) / gradient."
        ),
    },

    "wind_chill_at_summit": {
        "category": "atmospheric",
        "label": "Wind Chill Temperature at Summit",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "code_template": (
            "h = {elevation}\n"
            "T_a = 15 - 6.5 * h / 1000\n"
            "v = {wind_speed_kmh}\n"
            "# NWS wind chill formula (metric: T in °C, v in km/h)\n"
            "WC = 13.12 + 0.6215 * T_a - 11.37 * v**0.16 + 0.3965 * T_a * v**0.16\n"
            "print(round(WC, 2))"
        ),
        "question_hint": (
            "With a {wind_speed_kmh} km/h wind at this mountain's summit, what "
            "is the wind-chill temperature (°C)?"
        ),
        "answer_unit": "°C",
        "description": (
            "Steps: (1) summit temp from lapse rate, "
            "(2) NWS wind chill formula."
        ),
    },

    # -----------------------------------------------------------------------
    # Geometric / Geodetic (4 templates)
    # -----------------------------------------------------------------------

    "visible_horizon_area": {
        "category": "geometric",
        "label": "Visible Surface Area from Summit",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "R = 6371000\n"
            "d = math.sqrt(2 * R * h + h**2)\n"
            "h_cap = d**2 / (2 * R)\n"
            "area_km2 = 2 * math.pi * R * h_cap / 1e6\n"
            "print(round(area_km2, 1))"
        ),
        "question_hint": (
            "What area of Earth's surface (km²) is visible from the summit "
            "of this mountain?"
        ),
        "answer_unit": "km²",
        "description": (
            "Steps: (1) horizon distance d = sqrt(2Rh + h²), "
            "(2) spherical cap height h_cap = d²/(2R), "
            "(3) visible area = 2πR·h_cap."
        ),
    },

    "gravity_train_period": {
        "category": "geometric",
        "label": "Gravity Train One-Way Travel Time",
        "type": "comparative",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "# Famous result: one-way time = pi*sqrt(R/g) regardless of distance\n"
            "R = 6371000\n"
            "g = 9.81\n"
            "t_seconds = math.pi * math.sqrt(R / g)\n"
            "t_minutes = t_seconds / 60\n"
            "print(round(t_minutes, 2))"
        ),
        "question_hint": (
            "If a straight tunnel were dug through the Earth between these two "
            "cities, how many minutes would a gravity-powered train take for "
            "the one-way trip?"
        ),
        "answer_unit": "minutes",
        "description": (
            "Steps: (1) chord length (not needed for answer), "
            "(2) the famous result: period = π√(R/g), independent of distance, "
            "(3) one-way time = π√(R/g) / 60."
        ),
    },

    "line_of_sight_range": {
        "category": "geometric",
        "label": "Maximum Line-of-Sight Distance",
        "type": "comparative",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "h_a = {elevation_a}\n"
            "h_b = {elevation_b}\n"
            "R = 6371000\n"
            "d_a = math.sqrt(2 * R * h_a + h_a**2)\n"
            "d_b = math.sqrt(2 * R * h_b + h_b**2)\n"
            "los_km = (d_a + d_b) / 1000\n"
            "print(round(los_km, 2))"
        ),
        "question_hint": (
            "What is the maximum line-of-sight distance (km) between the "
            "summits of these two mountains?"
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) horizon_A = sqrt(2R·h_A + h_A²), "
            "(2) horizon_B = sqrt(2R·h_B + h_B²), "
            "(3) max LOS = d_A + d_B."
        ),
    },

    "coriolis_deflection": {
        "category": "geometric",
        "label": "Coriolis Deflection of a Projectile",
        "type": "single",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 4,
        "code_template": (
            "import math\n"
            "lat = {latitude}\n"
            "v = {projectile_speed}\n"
            "distance = {range_m}\n"
            "omega = 7.2921e-5\n"
            "f = 2 * omega * math.sin(math.radians(lat))\n"
            "t = distance / v\n"
            "deflection = 0.5 * f * v * t**2\n"
            "print(round(abs(deflection), 2))"
        ),
        "question_hint": (
            "A projectile is fired due north from this city at {projectile_speed} m/s "
            "over a {range_km} km range. How many metres does the Coriolis effect "
            "deflect it?"
        ),
        "answer_unit": "metres",
        "description": (
            "Steps: (1) Coriolis parameter f = 2Ω·sin(lat), "
            "(2) flight time t = d/v, "
            "(3) deflection = 0.5·f·v·t²."
        ),
    },

    # -----------------------------------------------------------------------
    # Crustal / Geophysical (3 templates)
    # -----------------------------------------------------------------------

    "isostatic_root_depth": {
        "category": "crustal",
        "label": "Airy Isostatic Root Depth",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "code_template": (
            "h = {elevation}\n"
            "rho_crust = {rho_crust}\n"
            "rho_mantle = {rho_mantle}\n"
            "root = h * rho_crust / (rho_mantle - rho_crust)\n"
            "root_km = root / 1000\n"
            "print(round(root_km, 2))"
        ),
        "question_hint": (
            "According to the Airy isostasy model (crustal density {rho_crust} kg/m³, "
            "mantle density {rho_mantle} kg/m³), how deep (km) does the crustal root "
            "extend beneath this mountain?"
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) Airy isostasy: root = h·ρ_c/(ρ_m - ρ_c), "
            "(2) convert to km."
        ),
    },

    "seismic_p_wave_time": {
        "category": "crustal",
        "label": "P-Wave Travel Time Between Cities",
        "type": "comparative",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "code_template": (
            "import math\n"
            "lat1, lon1 = math.radians({latitude_a}), math.radians({longitude_a})\n"
            "lat2, lon2 = math.radians({latitude_b}), math.radians({longitude_b})\n"
            "dlat = lat2 - lat1\n"
            "dlon = lon2 - lon1\n"
            "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
            "c = 2 * math.asin(math.sqrt(a))\n"
            "R = 6371\n"
            "dist_km = R * c\n"
            "p_wave_speed = {p_wave_speed}\n"
            "time_s = dist_km / p_wave_speed\n"
            "print(round(time_s, 2))"
        ),
        "question_hint": (
            "How many seconds would a P-wave (speed {p_wave_speed} km/s) take to "
            "travel through the upper crust between these two cities?"
        ),
        "answer_unit": "seconds",
        "description": (
            "Steps: (1) haversine distance, "
            "(2) divide by P-wave speed."
        ),
    },

    "free_air_gravity_anomaly": {
        "category": "crustal",
        "label": "Free-Air Gravity Anomaly",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 2,
        "code_template": (
            "h = {elevation}\n"
            "gradient = -0.3086\n"
            "anomaly = gradient * h\n"
            "print(round(anomaly, 2))"
        ),
        "question_hint": (
            "What is the free-air gravity anomaly (mGal) at this mountain's "
            "summit relative to sea level?"
        ),
        "answer_unit": "mGal",
        "description": (
            "Steps: (1) free-air gradient = -0.3086 mGal/m, "
            "(2) anomaly = gradient × elevation."
        ),
    },

    # -----------------------------------------------------------------------
    # Additional Solar (1 template)
    # -----------------------------------------------------------------------

    "solar_declination_day": {
        "category": "solar",
        "label": "Solar Declination on a Given Day",
        "type": "single",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 2,
        "reasoning_depth": 2,
        "code_template": (
            "import math\n"
            "day = {day_of_year}\n"
            "declination = -23.44 * math.cos(math.radians(360 / 365 * (day + 10)))\n"
            "print(round(declination, 2))"
        ),
        "question_hint": (
            "What is the solar declination angle (degrees) on day {day_of_year} "
            "of the year?"
        ),
        "answer_unit": "degrees",
        "description": (
            "Steps: (1) compute angle = 360/365*(day+10), "
            "(2) declination = -23.44*cos(angle)."
        ),
    },

    # -----------------------------------------------------------------------
    # Additional Atmospheric (4 templates)
    # -----------------------------------------------------------------------

    "lapse_rate_temperature": {
        "category": "atmospheric",
        "label": "Temperature at Altitude via Lapse Rate",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 2,
        "reasoning_depth": 2,
        "code_template": (
            "h = {elevation}\n"
            "T_sea = {base_temp}\n"
            "T = T_sea - 6.5 * h / 1000\n"
            "print(round(T, 2))"
        ),
        "question_hint": (
            "Given a sea-level temperature of {base_temp}\u00b0C, what is the "
            "temperature at this mountain's summit using the standard lapse rate?"
        ),
        "answer_unit": "\u00b0C",
        "description": (
            "Steps: (1) lapse rate = 6.5\u00b0C/km, "
            "(2) T = T_sea - 6.5*h/1000."
        ),
    },
    "pressure_altitude": {
        "category": "atmospheric",
        "label": "Barometric Pressure at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "P0 = 101325\n"
            "M = 0.029; g = 9.81; R = 8.314; T = 288.15\n"
            "P = P0 * math.exp(-M * g * h / (R * T))\n"
            "P_hPa = P / 100\n"
            "print(round(P_hPa, 2))"
        ),
        "question_hint": (
            "What is the atmospheric pressure (hPa) at this mountain's summit?"
        ),
        "answer_unit": "hPa",
        "description": (
            "Steps: (1) barometric formula P = P0*exp(-Mgh/RT), "
            "(2) convert Pa to hPa."
        ),
    },
    "dew_point_at_altitude": {
        "category": "atmospheric",
        "label": "Dew Point at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "h = {elevation}\n"
            "T_sea = {base_temp}\n"
            "RH = {relative_humidity}\n"
            "T = T_sea - 6.5 * h / 1000\n"
            "# Magnus formula approx\n"
            "import math\n"
            "a = 17.27; b = 237.7\n"
            "alpha = a * T / (b + T) + math.log(RH / 100)\n"
            "Td = b * alpha / (a - alpha)\n"
            "print(round(Td, 2))"
        ),
        "question_hint": (
            "At this mountain's summit with sea-level temp {base_temp}\u00b0C and "
            "relative humidity {relative_humidity}%, what is the dew point (\u00b0C)?"
        ),
        "answer_unit": "\u00b0C",
        "description": (
            "Steps: (1) summit temp via lapse rate, "
            "(2) Magnus formula for dew point, "
            "(3) Td = b*alpha/(a-alpha)."
        ),
    },
    "sound_speed_at_altitude": {
        "category": "atmospheric",
        "label": "Speed of Sound at Altitude",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "T_C = 15 - 6.5 * h / 1000\n"
            "T_K = T_C + 273.15\n"
            "gamma = 1.4; R = 287.05\n"
            "v = math.sqrt(gamma * R * T_K)\n"
            "print(round(v, 2))"
        ),
        "question_hint": (
            "What is the speed of sound (m/s) at this mountain's summit?"
        ),
        "answer_unit": "m/s",
        "description": (
            "Steps: (1) temperature via lapse rate, "
            "(2) convert to Kelvin, "
            "(3) speed = sqrt(gamma*R*T)."
        ),
    },

    # -----------------------------------------------------------------------
    # Additional Geometric (3 templates)
    # -----------------------------------------------------------------------

    "great_circle_distance": {
        "category": "geometric",
        "label": "Great Circle Distance Between Cities",
        "type": "comparative",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "import math\n"
            "lat1, lon1 = math.radians({latitude_a}), math.radians({longitude_a})\n"
            "lat2, lon2 = math.radians({latitude_b}), math.radians({longitude_b})\n"
            "dlat = lat2 - lat1\n"
            "dlon = lon2 - lon1\n"
            "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
            "c = 2 * math.asin(math.sqrt(a))\n"
            "R = 6371\n"
            "print(round(R * c, 2))"
        ),
        "question_hint": (
            "What is the great-circle distance (km) between these two cities?"
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) convert to radians, "
            "(2) haversine formula, "
            "(3) distance = R*c."
        ),
    },
    "angular_separation": {
        "category": "geometric",
        "label": "Angular Separation on Globe",
        "type": "comparative",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "import math\n"
            "lat1, lon1 = math.radians({latitude_a}), math.radians({longitude_a})\n"
            "lat2, lon2 = math.radians({latitude_b}), math.radians({longitude_b})\n"
            "dlat = lat2 - lat1\n"
            "dlon = lon2 - lon1\n"
            "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
            "c = 2 * math.asin(math.sqrt(a))\n"
            "print(round(math.degrees(c), 2))"
        ),
        "question_hint": (
            "What is the angular separation (degrees) between these two "
            "cities as seen from Earth's center?"
        ),
        "answer_unit": "degrees",
        "description": (
            "Steps: (1) convert to radians, "
            "(2) haversine formula, "
            "(3) convert central angle to degrees."
        ),
    },
    "antipodal_distance": {
        "category": "geometric",
        "label": "Distance to Antipodal Point",
        "type": "single",
        "entity_type": "city",
        "required_properties": ["P625"],
        "steps": 2,
        "reasoning_depth": 2,
        "code_template": (
            "import math\n"
            "R = 6371\n"
            "half_circumference = math.pi * R\n"
            "print(round(half_circumference, 2))"
        ),
        "question_hint": (
            "What is the great-circle distance (km) from this city to its "
            "antipodal point?"
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) antipodal distance = pi*R, "
            "(2) independent of location."
        ),
    },

    # -----------------------------------------------------------------------
    # Additional Crustal (2 templates)
    # -----------------------------------------------------------------------

    "bouguer_gravity_anomaly": {
        "category": "crustal",
        "label": "Simple Bouguer Gravity Anomaly",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "import math\n"
            "h = {elevation}\n"
            "rho = {crustal_density}\n"
            "G = 6.674e-11\n"
            "free_air = -0.3086 * h\n"
            "bouguer_correction = 2 * math.pi * G * rho * h * 1e5\n"
            "anomaly = free_air - bouguer_correction\n"
            "print(round(anomaly, 2))"
        ),
        "question_hint": (
            "Using a crustal density of {crustal_density} kg/m\u00b3, what is the "
            "simple Bouguer gravity anomaly (mGal) at this mountain's summit?"
        ),
        "answer_unit": "mGal",
        "description": (
            "Steps: (1) free-air correction = -0.3086*h, "
            "(2) Bouguer plate correction = 2\u03c0G\u03c1h, "
            "(3) anomaly = free-air - Bouguer."
        ),
    },
    "moho_depth_estimate": {
        "category": "crustal",
        "label": "Moho Depth Estimate",
        "type": "single",
        "entity_type": "mountain",
        "required_properties": ["P2044"],
        "steps": 3,
        "reasoning_depth": 3,
        "code_template": (
            "h = {elevation}\n"
            "avg_crust = {avg_crustal_thickness}\n"
            "rho_c = {rho_crust}\n"
            "rho_m = {rho_mantle}\n"
            "root = h * rho_c / (rho_m - rho_c)\n"
            "moho = avg_crust + root / 1000\n"
            "print(round(moho, 2))"
        ),
        "question_hint": (
            "With average crustal thickness {avg_crustal_thickness} km, crustal density "
            "{rho_crust} kg/m\u00b3, and mantle density {rho_mantle} kg/m\u00b3, estimate the "
            "Moho depth (km) beneath this mountain."
        ),
        "answer_unit": "km",
        "description": (
            "Steps: (1) Airy root = h*\u03c1_c/(\u03c1_m-\u03c1_c), "
            "(2) Moho depth = avg_crust + root/1000."
        ),
    },
}


# ===========================================================================
# Category → semantic entity type label
# ===========================================================================

# The ``entity_type`` field inside each theme entry below is a *property-group
# selector* used for template routing ("mountain" = has elevation P2044,
# "city" = has coordinates P625, "structure" = has height P2048 + P625).
# It does NOT describe the real-world type of the entity.
#
# ``category_to_entity_type()`` converts the plural category key (e.g.
# "lakes") into the correct singular label for output metadata (e.g. "lake").

_CATEGORY_SINGULAR: Dict[str, str] = {
    "archaeological_sites": "archaeological_site",
    "bridges": "bridge",
    "capes": "cape",
    "caves": "cave",
    "cities": "city",
    "countries": "country",
    "gorges": "gorge",
    "lakes": "lake",
    "lighthouses": "lighthouse",
    "national_parks": "national_park",
    "oases": "oasis",
    "observatories": "observatory",
    "ridges": "ridge",
    "valleys": "valley",
    "wind_turbines": "wind_turbine",
}


def category_to_entity_type(category: str) -> str:
    """Return the singular, human-readable entity type for *category*.

    Uses an explicit override table for irregular plurals, and falls back to
    stripping a trailing ``"s"`` / ``"es"`` for regular ones.
    """
    if category in _CATEGORY_SINGULAR:
        return _CATEGORY_SINGULAR[category]
    # Regular: "lakes" → "lake", "volcanoes" → "volcano", "passes" → "pass"
    if category.endswith("es"):
        return category[:-2]
    if category.endswith("s"):
        return category[:-1]
    return category


# ===========================================================================
# Entity type → category mapping
# ===========================================================================

# Maps theme name to entity_type + SPARQL discovery filters
GEOPHYSICAL_THEMES = {
    # --- entity_type "mountain" (P2044) ---
    "hills": {
        "wikidata_types": ["Q54050"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
            "FILTER(?elevation > 200)",
        ],
    },
    "ridges": {
        "wikidata_types": ["Q1437459"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
            "FILTER(?elevation > 500)",
        ],
    },
    "valleys": {
        "wikidata_types": ["Q39816"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
        ],
    },
    "craters": {
        "wikidata_types": ["Q3240715"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
        ],
    },
    "gorges": {
        "wikidata_types": ["Q150784"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
        ],
    },
    "canyons": {
        "wikidata_types": ["Q150784"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
            "FILTER(?elevation > 100)",
        ],
    },
    "fjords": {
        "wikidata_types": ["Q45776"],
        "entity_type": "mountain",
        "required_wikidata_props": ["P2044"],
        "sparql_filters": [
            "?item wdt:P2044 ?elevation .",
        ],
    },
    # --- entity_type "city" (P625 coordinates) ---
    "harbors": {
        "wikidata_types": ["Q283202"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    "observatories": {
        "wikidata_types": ["Q62832"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    "national_parks": {
        "wikidata_types": ["Q46169"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    "archaeological_sites": {
        "wikidata_types": ["Q839954"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    "oases": {
        "wikidata_types": ["Q43742"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    "capes": {
        "wikidata_types": ["Q185113"],
        "entity_type": "city",
        "required_wikidata_props": ["P625"],
        "sparql_filters": [
            "?item wdt:P625 ?coords .",
        ],
    },
    # --- entity_type "structure" (P2048 height + P625 coordinates) ---
    "lighthouses": {
        "wikidata_types": ["Q39715"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
        ],
    },
    "monuments": {
        "wikidata_types": ["Q4989906"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
            "FILTER(?height > 10)",
        ],
    },
    "stadiums": {
        "wikidata_types": ["Q483110"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
        ],
    },
    "chimneys": {
        "wikidata_types": ["Q245117"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
            "FILTER(?height > 50)",
        ],
    },
    "minarets": {
        "wikidata_types": ["Q105507"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
        ],
    },
    "wind_turbines": {
        "wikidata_types": ["Q49833"],
        "entity_type": "structure",
        "required_wikidata_props": ["P2048", "P625"],
        "sparql_filters": [
            "?item wdt:P2048 ?height .",
            "?item wdt:P625 ?coords .",
        ],
    },
}


# ===========================================================================
# Template Selection
# ===========================================================================

def select_geophysical_template(
    entity_type: str,
    quant_props: Dict[str, Any],
    used_templates: Set[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Select a template compatible with the entity type and available properties.

    Args:
        entity_type: "mountain", "city", or "structure"
        quant_props: Dict of property_id -> {amount, unit, label, year}
        used_templates: Template IDs already used

    Returns:
        (template_id, template_dict) or None
    """
    candidates = []
    for tmpl_id, tmpl in GEOPHYSICAL_TEMPLATES.items():
        if tmpl_id in used_templates:
            continue
        if tmpl["type"] == "comparative":
            continue  # handled separately
        if tmpl["entity_type"] != entity_type:
            continue
        # Check required properties
        if not all(p in quant_props for p in tmpl["required_properties"]):
            continue
        candidates.append((tmpl_id, tmpl))

    if not candidates:
        return None

    random.shuffle(candidates)
    return candidates[0]


def select_comparative_geophysical_template(
    entity_type: str,
    quant_props_a: Dict[str, Any],
    quant_props_b: Dict[str, Any],
    used_templates: Set[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Select a comparative template for two entities.

    Returns:
        (template_id, template_dict) or None
    """
    candidates = []
    for tmpl_id, tmpl in GEOPHYSICAL_TEMPLATES.items():
        if tmpl_id in used_templates:
            continue
        if tmpl["type"] != "comparative":
            continue
        if tmpl["entity_type"] != entity_type:
            continue
        # Both entities must have the required properties
        if not all(p in quant_props_a for p in tmpl["required_properties"]):
            continue
        if not all(p in quant_props_b for p in tmpl["required_properties"]):
            continue
        candidates.append((tmpl_id, tmpl))

    if not candidates:
        return None

    random.shuffle(candidates)
    return candidates[0]


# ===========================================================================
# Parameter Generation
# ===========================================================================

def generate_template_params(
    template_id: str,
    tmpl: Dict[str, Any],
    quant_props: Dict[str, Any],
    quant_props_b: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate parameters for a template from entity properties.

    For single-entity templates, uses quant_props only.
    For comparative templates, uses both quant_props (entity A) and quant_props_b.

    Returns:
        Parameter dict, or None if unable to fill.
    """
    params = {}

    # Extract coordinates
    if "P625" in quant_props:
        coords = quant_props["P625"]["amount"]
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            params["latitude"] = coords[0]
            params["longitude"] = coords[1]

    # Extract elevation
    if "P2044" in quant_props:
        params["elevation"] = quant_props["P2044"]["amount"]

    # Extract height
    if "P2048" in quant_props:
        params["height"] = quant_props["P2048"]["amount"]

    # For comparative: extract from entity B
    if quant_props_b is not None:
        if "P625" in quant_props_b:
            coords_b = quant_props_b["P625"]["amount"]
            if isinstance(coords_b, (list, tuple)) and len(coords_b) == 2:
                params["latitude_b"] = coords_b[0]
                params["longitude_b"] = coords_b[1]
        if "P2044" in quant_props_b:
            params["elevation_b"] = quant_props_b["P2044"]["amount"]
        if "P2048" in quant_props_b:
            params["height_b"] = quant_props_b["P2048"]["amount"]
        # Rename A-side for comparative templates that use _a/_b suffixes
        if "elevation" in params:
            params["elevation_a"] = params["elevation"]
        if "latitude" in params:
            params["latitude_a"] = params["latitude"]
            params["longitude_a"] = params["longitude"]

    # Add template-specific random parameters
    if template_id == "solar_noon_time":
        # Standard meridian: nearest multiple of 15 to the city's longitude
        lon = params.get("longitude", 0)
        params["standard_meridian"] = round(lon / 15) * 15

    elif template_id == "boiling_time_at_altitude":
        params["water_mass"] = random.choice([0.5, 1.0, 1.5, 2.0])
        params["start_temp"] = random.choice([10, 15, 20, 25])
        params["power"] = random.choice([1000, 1500, 2000, 2500])

    elif template_id == "terminal_velocity_at_altitude":
        params["skydiver_mass"] = random.choice([70, 75, 80, 85, 90])
        params["drag_coefficient"] = random.choice([0.8, 1.0, 1.2])
        params["cross_section_area"] = random.choice([0.5, 0.7, 0.9])

    elif template_id == "geothermal_boiling_depth":
        params["geothermal_gradient"] = random.choice([25, 30, 35])

    elif template_id == "wind_chill_at_summit":
        params["wind_speed_kmh"] = random.choice([20, 30, 40, 50, 60])

    elif template_id == "coriolis_deflection":
        params["projectile_speed"] = random.choice([500, 800, 1000, 1200])
        range_km = random.choice([5, 10, 15, 20])
        params["range_km"] = range_km
        params["range_m"] = range_km * 1000

    elif template_id == "isostatic_root_depth":
        params["rho_crust"] = random.choice([2700, 2750, 2800])
        params["rho_mantle"] = random.choice([3200, 3300, 3400])

    elif template_id == "seismic_p_wave_time":
        params["p_wave_speed"] = random.choice([6.0, 6.5, 7.0])

    elif template_id == "solar_declination_day":
        params["day_of_year"] = random.choice([1, 80, 172, 266, 355])

    elif template_id == "lapse_rate_temperature":
        params["base_temp"] = random.choice([15, 20, 25, 30])

    elif template_id == "pressure_altitude":
        pass  # No extra params needed

    elif template_id == "dew_point_at_altitude":
        params["base_temp"] = random.choice([15, 20, 25, 30])
        params["relative_humidity"] = random.choice([40, 50, 60, 70, 80])

    elif template_id == "sound_speed_at_altitude":
        pass  # No extra params needed

    elif template_id == "great_circle_distance":
        pass  # Uses coordinates from both entities

    elif template_id == "angular_separation":
        pass  # Uses coordinates from both entities

    elif template_id == "antipodal_distance":
        pass  # No extra params needed

    elif template_id == "bouguer_gravity_anomaly":
        params["crustal_density"] = random.choice([2670, 2700, 2750, 2800])

    elif template_id == "moho_depth_estimate":
        params["avg_crustal_thickness"] = random.choice([30, 35, 40])
        params["rho_crust"] = random.choice([2700, 2750, 2800])
        params["rho_mantle"] = random.choice([3200, 3300, 3400])

    return params


# ===========================================================================
# Code Execution
# ===========================================================================

def _execute_computation_code(code: str, timeout: int = 10) -> Optional[str]:
    """Execute computation code in a subprocess sandbox."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def compute_gold_answer(
    template_id: str,
    tmpl: Dict[str, Any],
    params: Dict[str, Any],
) -> Optional[str]:
    """Fill template code, execute, and return gold answer string."""
    try:
        code = tmpl["code_template"].format(**params)
    except KeyError:
        return None

    answer = _execute_computation_code(code)
    if answer is None:
        return None

    # Validate
    try:
        val = float(answer)
        if math.isnan(val) or math.isinf(val):
            return None
    except ValueError:
        return None

    return answer
