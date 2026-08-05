# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Multiskill reasoning templates: domain definitions, reference constants,
template selection, parameter filling, and gold-chain construction.

Extracted from ``multiskill_drbencher.py`` to keep the main module focused on
QA generation, verification, and CLI logic.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Unit Normalization
# ===========================================================================

# Expected units for computation templates and conversion factors from common alternatives.
# Format: property_id -> (expected_unit, {source_unit: divisor})
_UNIT_NORMALIZATION = {
    "P2046": ("square kilometre", {"square metre": 1_000_000}),
    "P2043": ("kilometre", {"metre": 1000}),
}


# ===========================================================================
# Reasoning Domains
# ===========================================================================

REASONING_DOMAINS = {
    "quantitative_modeling": {
        "label": "Quantitative Modeling",
        "description": "Apply mathematical formulas to KG-sourced numerical data.",
        "templates": {
            "exponential_growth": {
                "label": "Exponential Growth Projection",
                "required_properties": ["P1082"],
                "parameters": ["population", "growth_rate", "years"],
                "defaults": {"growth_rate": 0.01, "years": 20},
                "code_template": "P = {population}; r = {growth_rate}; t = {years}; print(round(P * (1 + r) ** t))",
                "question_hint": "Using its {data_year} population, if it grows at {growth_rate:.2%} per year, what will it be in {years} years?",
                "answer_unit": "people",
                "reasoning_depth": 2,
            },
            "solve_for_time": {
                "label": "Solve Exponential for Time",
                "required_properties": ["P1082"],
                "parameters": ["population", "target_population", "growth_rate"],
                "defaults": {"growth_rate": 0.005},
                "code_template": "import math; P = {population}; Q = {target_population}; r = {growth_rate}; print(round(math.log(Q / P) / math.log(1 + r)))",
                "question_hint": "Starting from its {data_year} population, at {growth_rate:.2%} annual growth, how many years until the population reaches {target_population}?",
                "answer_unit": "years",
                "reasoning_depth": 2,
            },
            "population_density": {
                "label": "Population Density",
                "required_properties": ["P1082", "P2046"],
                "parameters": ["population", "area"],
                "defaults": {},
                "code_template": "print(round({population} / {area}, 2))",
                "question_hint": "Based on its {data_year} population and area, what is the population density (people per km²)?",
                "answer_unit": "people/km²",
                "reasoning_depth": 2,
            },
            "ratio_two_properties": {
                "label": "GDP per Capita from Totals",
                "required_properties": ["P2131", "P1082"],
                "parameters": ["gdp", "population"],
                "defaults": {},
                "code_template": "print(round({gdp} / {population}, 2))",
                "question_hint": "Using its {data_year} GDP and population figures, what is the GDP per capita?",
                "answer_unit": "USD",
                "reasoning_depth": 2,
            },
            "compound_growth": {
                "label": "Compound Growth (n-times/year)",
                "required_properties": ["P1082"],
                "parameters": ["population", "growth_rate", "compounds_per_year", "years"],
                "defaults": {"growth_rate": 0.02, "compounds_per_year": 4, "years": 10},
                "code_template": "P = {population}; r = {growth_rate}; n = {compounds_per_year}; t = {years}; print(round(P * (1 + r / n) ** (n * t)))",
                "question_hint": "Using its {data_year} population, with {growth_rate:.2%} annual growth compounded {compounds_per_year} times per year, what will the population be in {years} years?",
                "answer_unit": "people",
                "reasoning_depth": 2,
            },
            "area_comparison": {
                "label": "Area Comparison Ratio",
                "required_properties": ["P2046"],
                "compatible_themes": ["countries", "cities", "islands", "lakes"],
                "parameters": ["area", "reference_area"],
                "defaults": {},
                "code_template": "print(round({area} / {reference_area}, 6))",
                "question_hint": "How many times larger is this entity's area than {reference_name}'s area?",
                "answer_unit": "times",
                "reasoning_depth": 2,
            },
            "travel_time": {
                "label": "Travel Time Across Entity",
                "required_properties": ["P2046"],
                "compatible_themes": ["countries", "cities", "islands", "lakes"],
                "parameters": ["area", "speed_kmh"],
                "defaults": {"speed_kmh": 100},
                "code_template": "import math; d = math.sqrt({area}); print(round(d / {speed_kmh}, 6))",
                "question_hint": "Approximating this entity as a square, if you drive at {speed_kmh} km/h across one side, how many hours would it take?",
                "answer_unit": "hours",
                "reasoning_depth": 3,
            },
            "length_comparison": {
                "label": "Length Comparison Ratio",
                "required_properties": ["P2043"],
                "compatible_themes": ["rivers"],
                "parameters": ["length", "reference_length"],
                "defaults": {},
                "code_template": "print(round({length} / {reference_length}, 6))",
                "question_hint": "How many times longer is this river than the {reference_river}?",
                "answer_unit": "times",
                "reasoning_depth": 2,
            },
            "height_comparison": {
                "label": "Height Comparison Ratio",
                "required_properties": ["P2048"],
                "compatible_themes": ["buildings", "dams", "waterfalls", "bridges", "towers", "skyscrapers"],
                "parameters": ["height", "reference_height"],
                "defaults": {},
                "code_template": "print(round({height} / {reference_height}, 6))",
                "question_hint": "How many times taller is this structure than {reference_structure}?",
                "answer_unit": "times",
                "reasoning_depth": 2,
            },
            "elevation_comparison": {
                "label": "Elevation Comparison Ratio",
                "required_properties": ["P2044"],
                "compatible_themes": ["mountains", "volcanoes", "cities", "countries"],
                "parameters": ["elevation", "reference_elevation"],
                "defaults": {},
                "code_template": "print(round({elevation} / {reference_elevation}, 6))",
                "question_hint": "How many times higher is this location's elevation than {reference_location}'s elevation?",
                "answer_unit": "times",
                "reasoning_depth": 2,
            },
            "mass_comparison": {
                "label": "Mass Comparison Ratio",
                "required_properties": ["P2067"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg", "reference_mass"],
                "defaults": {},
                "code_template": "print(round({mass_kg} / {reference_mass}, 6))",
                "question_hint": "How many times more massive is this body than {reference_body}?",
                "answer_unit": "times",
                "reasoning_depth": 2,
            },
            "river_travel_time": {
                "label": "River Travel Time",
                "required_properties": ["P2043"],
                "compatible_themes": ["rivers"],
                "parameters": ["length", "speed_kmh"],
                "defaults": {"speed_kmh": 80},
                "code_template": "print(round({length} / {speed_kmh}, 6))",
                "question_hint": "If you travel along the full length of this river at {speed_kmh} km/h, how many hours would the trip take?",
                "answer_unit": "hours",
                "reasoning_depth": 2,
            },
            "perimeter_from_area": {
                "label": "Circumference from Area (Circle Approximation)",
                "required_properties": ["P2046"],
                "compatible_themes": ["countries", "cities", "islands", "lakes", "deserts", "glaciers"],
                "parameters": ["area"],
                "defaults": {},
                "code_template": "import math; print(round(2 * math.sqrt(math.pi * {area}), 2))",
                "question_hint": "Approximating this entity as a circle, what is its circumference in km?",
                "answer_unit": "km",
                "reasoning_depth": 3,
            },
            "prominence_ratio": {
                "label": "Prominence-to-Elevation Ratio",
                "required_properties": ["P2660", "P2044"],
                "compatible_themes": ["mountains"],
                "parameters": ["prominence", "elevation"],
                "defaults": {},
                "code_template": "print(round({prominence} / {elevation}, 6))",
                "question_hint": "What fraction of this mountain's total elevation is its topographic prominence?",
                "answer_unit": "ratio",
                "reasoning_depth": 2,
            },
            "depth_pressure": {
                "label": "Water Pressure at Depth",
                "required_properties": ["P4511"],
                "compatible_themes": ["lakes"],
                "parameters": ["depth"],
                "defaults": {},
                "code_template": "print(round(101.325 + 9.81 * {depth}, 2))",
                "question_hint": "What is the water pressure (kPa) at the deepest point of this lake?",
                "answer_unit": "kPa",
                "reasoning_depth": 2,
            },
            "water_volume_lake": {
                "label": "Lake Volume Estimate",
                "required_properties": ["P2046"],
                "parameters": ["area", "avg_depth"],
                "defaults": {"avg_depth": None},
                "code_template": "area_km2 = {area}\navg_depth_m = {avg_depth}\nvolume_km3 = area_km2 * avg_depth_m / 1000\nprint(round(volume_km3, 4))",
                "question_hint": "Assuming an average depth of {avg_depth} m, estimate the volume (km\u00b3) of this lake.",
                "answer_unit": "km\u00b3",
                "reasoning_depth": 2,
                "compatible_themes": ["lakes"],
            },
            "dam_capacity_ratio": {
                "label": "Dam Height-to-Width Ratio",
                "required_properties": ["P2048", "P2787"],
                "parameters": ["height", "crest_length"],
                "defaults": {},
                "code_template": "h = {height}\nL = {crest_length}\nratio = h / L\nprint(round(ratio, 4))",
                "question_hint": "What is the height-to-crest-length ratio of this dam?",
                "answer_unit": "",
                "reasoning_depth": 2,
                "compatible_themes": ["dams"],
            },
            "bridge_span_ratio": {
                "label": "Bridge Span-to-Height Ratio",
                "required_properties": ["P2787", "P2048"],
                "parameters": ["total_length", "height"],
                "defaults": {},
                "code_template": "L = {total_length}\nh = {height}\nratio = L / h\nprint(round(ratio, 2))",
                "question_hint": "What is the total-length-to-height ratio of this bridge?",
                "answer_unit": "",
                "reasoning_depth": 2,
                "compatible_themes": ["bridges"],
            },
            "island_coastline_density": {
                "label": "Island Coastline Density",
                "required_properties": ["P2046", "P2660"],
                "parameters": ["area", "coastline"],
                "defaults": {},
                "code_template": "import math\nA = {area}\nC = {coastline}\ndensity = C / math.sqrt(A)\nprint(round(density, 4))",
                "question_hint": "What is the coastline density (coastline / sqrt(area)) of this island?",
                "answer_unit": "",
                "reasoning_depth": 2,
                "compatible_themes": ["islands"],
            },
        },
    },
    "scientific_inference": {
        "label": "Scientific Inference",
        "description": "Apply scientific principles to KG entity data.",
        "templates": {
            "boiling_point_altitude": {
                "label": "Boiling Point at Altitude",
                "required_properties": ["P2044"],
                "parameters": ["elevation"],
                "defaults": {},
                "code_template": (
                    "import math\n"
                    "h = {elevation}\n"
                    "# Barometric formula: P = P0 * exp(-Mgh / RT)\n"
                    "P0 = 101325  # sea level Pa\n"
                    "M = 0.029  # kg/mol air\n"
                    "g = 9.81\n"
                    "R = 8.314\n"
                    "T = 288.15  # avg temp K\n"
                    "P = P0 * math.exp(-M * g * h / (R * T))\n"
                    "# Clausius-Clapeyron: dT = R_water * T_b^2 / L * ln(P/P0)\n"
                    "T_b = 373.15  # boiling at 1 atm\n"
                    "L = 2260000  # J/kg\n"
                    "R_w = 461.5  # J/(kg·K) water vapor\n"
                    "dT = (R_w * T_b**2 / L) * math.log(P / P0)\n"
                    "print(round(100 + dT, 2))"
                ),
                "question_hint": "At this elevation, what is the boiling point of water (°C)?",
                "answer_unit": "°C",
                "reasoning_depth": 4,
            },
            "atmospheric_pressure": {
                "label": "Atmospheric Pressure at Elevation",
                "required_properties": ["P2044"],
                "parameters": ["elevation"],
                "defaults": {},
                "code_template": (
                    "import math\n"
                    "h = {elevation}\n"
                    "P0 = 101325\n"
                    "M = 0.029\n"
                    "g = 9.81\n"
                    "R = 8.314\n"
                    "T = 288.15\n"
                    "P = P0 * math.exp(-M * g * h / (R * T))\n"
                    "print(round(P / 1000, 2))"
                ),
                "question_hint": "What is the atmospheric pressure at this elevation (kPa)?",
                "answer_unit": "kPa",
                "reasoning_depth": 3,
            },
            "haversine_distance": {
                "label": "Great-Circle Distance",
                "required_properties": ["P625"],
                "hide_properties": ["P625"],  # Don't include coordinates in quant_data_text
                "parameters": ["lat1", "lon1", "lat2", "lon2"],
                "defaults": {},
                "code_template": (
                    "import math\n"
                    "lat1, lon1 = math.radians({lat1}), math.radians({lon1})\n"
                    "lat2, lon2 = math.radians({lat2}), math.radians({lon2})\n"
                    "dlat = lat2 - lat1\n"
                    "dlon = lon2 - lon1\n"
                    "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
                    "c = 2 * math.asin(math.sqrt(a))\n"
                    "R = 6371\n"
                    "print(round(R * c, 1))"
                ),
                "question_hint": "Using the haversine formula, what is the great-circle distance in km from this city to {reference_city}?",
                "answer_unit": "km",
                "reasoning_depth": 4,
            },
            "gravitational_weight": {
                "label": "Gravitational Weight on Another Body",
                "required_properties": ["P2067"],
                "parameters": ["mass_kg", "surface_gravity"],
                "defaults": {"surface_gravity": 3.72},  # Mars by default
                "code_template": "m = {mass_kg}; g = {surface_gravity}; print(round(m * g, 2))",
                "question_hint": "What would this object weigh (N) on a body with surface gravity {surface_gravity} m/s²?",
                "answer_unit": "N",
                "reasoning_depth": 2,
            },
            "surface_gravity": {
                "label": "Surface Gravity Calculation",
                "required_properties": ["P2067", "P2120"],
                "parameters": ["mass_kg", "radius_m"],
                "defaults": {},
                "code_template": "G = 6.674e-11; M = {mass_kg}; r = {radius_m}; print(round(G * M / r**2, 4))",
                "question_hint": "What is the surface gravity (m/s²) of this body?",
                "answer_unit": "m/s²",
                "reasoning_depth": 2,
            },
            "temperature_at_altitude": {
                "label": "Temperature at Altitude (Lapse Rate)",
                "required_properties": ["P2044"],
                "parameters": ["elevation", "sea_level_temp"],
                "defaults": {"sea_level_temp": 25},
                "code_template": "print(round({sea_level_temp} - 6.5 * {elevation} / 1000, 2))",
                "question_hint": "If the sea-level temperature is {sea_level_temp}°C, what is the expected temperature at this elevation using the standard lapse rate (−6.5°C per 1000 m)?",
                "answer_unit": "°C",
                "reasoning_depth": 2,
            },
            "free_fall_time": {
                "label": "Free-Fall Time from Height",
                "required_properties": ["P2048"],
                "parameters": ["height"],
                "defaults": {},
                "code_template": "import math; print(round(math.sqrt(2 * {height} / 9.81), 2))",
                "question_hint": "How many seconds would it take an object to free-fall from the top of this structure (ignoring air resistance)?",
                "answer_unit": "seconds",
                "reasoning_depth": 2,
            },
            "floor_height": {
                "label": "Average Floor Height",
                "required_properties": ["P2048", "P1101"],
                "parameters": ["height", "floors"],
                "defaults": {},
                "code_template": "print(round({height} / {floors}, 2))",
                "question_hint": "What is the average height per floor of this building?",
                "answer_unit": "metres",
                "reasoning_depth": 2,
            },
            "escape_velocity": {
                "label": "Escape Velocity",
                "required_properties": ["P2067", "P2120"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg", "radius_m"],
                "defaults": {},
                "code_template": "import math; G=6.674e-11; print(round(math.sqrt(2*G*{mass_kg}/{radius_m}), 2))",
                "question_hint": "What is the escape velocity (m/s) from the surface of this body?",
                "answer_unit": "m/s",
                "reasoning_depth": 2,
            },
            "orbital_velocity": {
                "label": "Orbital Velocity at Surface",
                "required_properties": ["P2067", "P2120"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg", "radius_m"],
                "defaults": {},
                "code_template": "import math; G=6.674e-11; print(round(math.sqrt(G*{mass_kg}/{radius_m}), 2))",
                "question_hint": "What is the orbital velocity (m/s) at the surface of this body?",
                "answer_unit": "m/s",
                "reasoning_depth": 2,
            },
            "density_from_mass_radius": {
                "label": "Mean Density from Mass and Radius",
                "required_properties": ["P2067", "P2120"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg", "radius_m"],
                "defaults": {},
                "code_template": "import math; print(round(3*{mass_kg}/(4*math.pi*{radius_m}**3), 2))",
                "question_hint": "What is the mean density (kg/m³) of this body?",
                "answer_unit": "kg/m³",
                "reasoning_depth": 3,
            },
            "energy_to_climb": {
                "label": "Gravitational Potential Energy (Height)",
                "required_properties": ["P2048"],
                "compatible_themes": ["buildings", "dams", "waterfalls", "bridges", "towers", "skyscrapers"],
                "parameters": ["height", "climber_mass"],
                "defaults": {"climber_mass": 70},
                "code_template": "print(round({climber_mass} * 9.81 * {height}, 2))",
                "question_hint": "How many joules of gravitational potential energy would a {climber_mass} kg person gain by climbing to the top of this structure?",
                "answer_unit": "joules",
                "reasoning_depth": 2,
            },
            "energy_to_climb_elevation": {
                "label": "Gravitational Potential Energy (Elevation)",
                "required_properties": ["P2044"],
                "compatible_themes": ["mountains", "volcanoes"],
                "parameters": ["elevation", "climber_mass"],
                "defaults": {"climber_mass": 70},
                "code_template": "print(round({climber_mass} * 9.81 * {elevation}, 2))",
                "question_hint": "How many joules of gravitational potential energy would a {climber_mass} kg person gain by climbing from sea level to the summit of this peak?",
                "answer_unit": "joules",
                "reasoning_depth": 2,
            },
            "pendulum_period": {
                "label": "Pendulum Period from Height",
                "required_properties": ["P2048"],
                "compatible_themes": ["buildings", "dams", "towers", "skyscrapers"],
                "parameters": ["height"],
                "defaults": {},
                "code_template": "import math; print(round(2 * math.pi * math.sqrt({height} / 9.81), 2))",
                "question_hint": "If a pendulum were as long as this structure is tall, what would its period be (seconds)?",
                "answer_unit": "seconds",
                "reasoning_depth": 2,
            },
            "horizon_distance": {
                "label": "Horizon Distance from Height",
                "required_properties": ["P2048"],
                "compatible_themes": ["buildings", "dams", "waterfalls", "bridges", "towers", "skyscrapers"],
                "parameters": ["height"],
                "defaults": {},
                "code_template": "import math; print(round(math.sqrt(2 * 6371000 * {height}) / 1000, 2))",
                "question_hint": "Standing at the top of this structure, how far is the horizon (km), assuming a spherical Earth?",
                "answer_unit": "km",
                "reasoning_depth": 3,
            },
            "horizon_distance_elevation": {
                "label": "Horizon Distance from Elevation",
                "required_properties": ["P2044"],
                "compatible_themes": ["mountains", "volcanoes"],
                "parameters": ["elevation"],
                "defaults": {},
                "code_template": "import math; print(round(math.sqrt(2 * 6371000 * {elevation}) / 1000, 2))",
                "question_hint": "Standing at the summit, how far is the horizon (km), assuming a spherical Earth?",
                "answer_unit": "km",
                "reasoning_depth": 3,
            },
            "kepler_orbital_period": {
                "label": "Low-Orbit Period (Kepler)",
                "required_properties": ["P2067", "P2120"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg", "radius_m", "orbit_altitude"],
                "defaults": {"orbit_altitude": 200000},
                "code_template": "import math; G=6.674e-11; r={radius_m}+{orbit_altitude}; print(round(2*math.pi*math.sqrt(r**3/(G*{mass_kg}))/3600, 2))",
                "question_hint": "What is the orbital period (hours) of a satellite at {orbit_altitude_km:.0f} km altitude above this body?",
                "answer_unit": "hours",
                "reasoning_depth": 3,
            },
            "schwarzschild_radius": {
                "label": "Schwarzschild Radius",
                "required_properties": ["P2067"],
                "compatible_themes": ["planets"],
                "parameters": ["mass_kg"],
                "defaults": {},
                "code_template": "G=6.674e-11; c=299792458; print(round(2*G*{mass_kg}/c**2, 6))",
                "question_hint": "What is the Schwarzschild radius (metres) for a black hole with the same mass as this body?",
                "answer_unit": "metres",
                "reasoning_depth": 2,
            },
            "oxygen_partial_pressure": {
                "label": "Oxygen Partial Pressure at Elevation",
                "required_properties": ["P2044"],
                "compatible_themes": ["mountains", "volcanoes", "cities"],
                "parameters": ["elevation"],
                "defaults": {},
                "code_template": (
                    "import math; P0=101325; M=0.029; g=9.81; R=8.314; T=288.15; "
                    "P=P0*math.exp(-M*g*{elevation}/(R*T)); print(round(0.2095*P/1000, 2))"
                ),
                "question_hint": "What is the partial pressure of oxygen (kPa) at this elevation?",
                "answer_unit": "kPa",
                "reasoning_depth": 3,
            },
            "sound_travel_time": {
                "label": "Sound Travel Time (Great-Circle)",
                "required_properties": ["P625"],
                "hide_properties": ["P625"],
                "compatible_themes": ["cities", "countries", "islands"],
                "parameters": ["lat1", "lon1", "lat2", "lon2"],
                "defaults": {},
                "code_template": (
                    "import math\n"
                    "lat1, lon1 = math.radians({lat1}), math.radians({lon1})\n"
                    "lat2, lon2 = math.radians({lat2}), math.radians({lon2})\n"
                    "dlat = lat2 - lat1\n"
                    "dlon = lon2 - lon1\n"
                    "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
                    "c = 2 * math.asin(math.sqrt(a))\n"
                    "R = 6371\n"
                    "dist_km = R * c\n"
                    "print(round(dist_km * 1000 / 343 / 3600, 2))"
                ),
                "question_hint": "If sound could travel in a straight line from this city to {reference_city}, how many hours would it take at 343 m/s?",
                "answer_unit": "hours",
                "reasoning_depth": 4,
            },
            "light_travel_time": {
                "label": "Light Travel Time (Great-Circle)",
                "required_properties": ["P625"],
                "hide_properties": ["P625"],
                "compatible_themes": ["cities", "countries", "islands"],
                "parameters": ["lat1", "lon1", "lat2", "lon2"],
                "defaults": {},
                "code_template": (
                    "import math\n"
                    "lat1, lon1 = math.radians({lat1}), math.radians({lon1})\n"
                    "lat2, lon2 = math.radians({lat2}), math.radians({lon2})\n"
                    "dlat = lat2 - lat1\n"
                    "dlon = lon2 - lon1\n"
                    "a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
                    "c = 2 * math.asin(math.sqrt(a))\n"
                    "R = 6371\n"
                    "dist_km = R * c\n"
                    "print(round(dist_km * 1000 / 299792458 * 1000, 4))"
                ),
                "question_hint": "How many milliseconds would it take light to travel the great-circle distance from this city to {reference_city}?",
                "answer_unit": "milliseconds",
                "reasoning_depth": 4,
            },
            "time_zone_difference": {
                "label": "Natural UTC Offset from Longitude",
                "required_properties": ["P625"],
                "hide_properties": ["P625"],
                "compatible_themes": ["cities", "countries", "islands"],
                "parameters": ["lon1"],
                "defaults": {},
                "code_template": "print(round({lon1} / 15, 2))",
                "question_hint": "Based on this location's longitude, what is its natural UTC offset (hours) using the solar time convention (longitude ÷ 15)?",
                "answer_unit": "hours",
                "reasoning_depth": 2,
            },
            "volcanic_eruption_energy": {
                "label": "Volcanic Eruption Energy (VEI)",
                "required_properties": ["P2044"],
                "parameters": ["elevation", "vei_index"],
                "defaults": {"vei_index": None},
                "code_template": (
                    "import math\n"
                    "vei = {vei_index}\n"
                    "energy_J = 10 ** (4.4 * vei + 9)\n"
                    "print(round(energy_J, 2))"
                ),
                "question_hint": "Given a Volcanic Explosivity Index (VEI) of {vei_index}, estimate the eruption energy in joules using E = 10^(4.4*VEI + 9).",
                "answer_unit": "J",
                "reasoning_depth": 2,
                "compatible_themes": ["volcanoes"],
            },
            "river_flow_rate": {
                "label": "River Cross-Section Flow Rate",
                "required_properties": ["P2787"],
                "parameters": ["length_km", "width_m", "depth_m", "velocity_ms"],
                "defaults": {"width_m": None, "depth_m": None, "velocity_ms": None},
                "code_template": (
                    "w = {width_m}\n"
                    "d = {depth_m}\n"
                    "v = {velocity_ms}\n"
                    "Q = w * d * v\n"
                    "print(round(Q, 2))"
                ),
                "question_hint": "If this river has a cross-section width of {width_m} m, depth of {depth_m} m, and flow velocity of {velocity_ms} m/s, what is the volumetric flow rate (m\u00b3/s)?",
                "answer_unit": "m\u00b3/s",
                "reasoning_depth": 2,
                "compatible_themes": ["rivers"],
            },
            "canal_transit_time": {
                "label": "Canal Transit Time",
                "required_properties": ["P2787"],
                "parameters": ["length_km", "speed_knots"],
                "defaults": {"speed_knots": None},
                "code_template": (
                    "L_km = {length_km}\n"
                    "speed_kts = {speed_knots}\n"
                    "speed_kmh = speed_kts * 1.852\n"
                    "time_h = L_km / speed_kmh\n"
                    "print(round(time_h, 2))"
                ),
                "question_hint": "At a vessel speed of {speed_knots} knots, how many hours does it take to transit this canal?",
                "answer_unit": "hours",
                "reasoning_depth": 3,
                "compatible_themes": ["canals"],
            },
            "stadium_capacity_density": {
                "label": "Stadium Capacity per Area",
                "required_properties": ["P1083"],
                "parameters": ["capacity", "footprint_m2"],
                "defaults": {"footprint_m2": None},
                "code_template": (
                    "cap = {capacity}\n"
                    "A = {footprint_m2}\n"
                    "density = cap / A\n"
                    "print(round(density, 4))"
                ),
                "question_hint": "If this stadium has a footprint area of {footprint_m2} m\u00b2, what is the capacity density (seats/m\u00b2)?",
                "answer_unit": "seats/m\u00b2",
                "reasoning_depth": 2,
                "compatible_themes": ["stadiums"],
            },
        },
    },
}

# Which properties are compatible with which domains
PROPERTY_DOMAIN_COMPATIBILITY = {
    "P1082": ["quantitative_modeling"],
    "P2046": ["quantitative_modeling"],
    "P2131": ["quantitative_modeling"],
    "P2132": ["quantitative_modeling"],
    "P2044": ["scientific_inference", "quantitative_modeling"],
    "P625":  ["scientific_inference"],
    "P2067": ["scientific_inference", "quantitative_modeling"],
    "P2120": ["scientific_inference"],
    "P2054": ["scientific_inference"],
    "P2043": ["quantitative_modeling"],
    "P2048": ["scientific_inference", "quantitative_modeling"],
    "P1101": ["scientific_inference"],
    "P2660": ["scientific_inference", "quantitative_modeling"],
    "P4511": ["quantitative_modeling"],
    "P2787": ["quantitative_modeling", "scientific_inference"],
    "P1083": ["scientific_inference"],
}

# Reference cities for haversine template
REFERENCE_CITIES = {
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
    "Cairo": (30.0444, 31.2357),
    "São Paulo": (-23.5505, -46.6333),
}

# Reference areas for area_comparison template (km²)
REFERENCE_AREAS = {
    "Manhattan": 59.1,
    "Luxembourg": 2586,
    "Singapore": 733,
}

# Reference rivers for length_comparison template (km)
REFERENCE_RIVERS = {
    "Thames": 346,
    "Danube": 2850,
    "Rhine": 1230,
}

# Reference heights for height_comparison template (m)
REFERENCE_HEIGHTS = {
    "Eiffel Tower": 330,
    "Statue of Liberty": 93,
    "Big Ben": 96,
    "Leaning Tower of Pisa": 56,
    "Great Pyramid of Giza": 146,
}

# Reference elevations for elevation_comparison template (m)
REFERENCE_ELEVATIONS = {
    "Denver": 1609,
    "Mexico City": 2240,
    "La Paz": 3640,
    "Dead Sea": -430,
    "Mount Everest": 8849,
}

# Reference masses for mass_comparison template (kg)
REFERENCE_MASSES = {
    "Earth": 5.972e24,
    "Moon": 7.342e22,
    "Mars": 6.417e23,
}

# Themes usable for multi-skill bench (entities with quantitative properties)
MULTISKILL_THEMES = {
    "countries": ["Q6256"],          # instance of: country
    "cities": ["Q515"],              # instance of: city
    "mountains": ["Q8502"],          # instance of: mountain
    "lakes": ["Q23397"],             # instance of: lake
    "planets": ["Q634"],             # instance of: planet
    "islands": ["Q23442"],           # instance of: island
    "rivers": ["Q4022"],             # instance of: river
    "buildings": ["Q11303"],         # instance of: skyscraper
    "volcanoes": ["Q8072"],          # instance of: volcano
    "dams": ["Q12323"],              # instance of: dam
    "waterfalls": ["Q34038"],        # instance of: waterfall
    "bridges": ["Q12280"],           # instance of: bridge
    "deserts": ["Q8514"],            # instance of: desert
    "glaciers": ["Q35666"],          # instance of: glacier
    "towers": ["Q12518"],            # instance of: tower
    "peninsulas": ["Q34763"],        # instance of: peninsula
    "caves": ["Q35509"],             # instance of: cave
    "canals": ["Q12284"],            # instance of: canal
    "stadiums": ["Q483110"],         # instance of: stadium
    "lighthouses": ["Q39715"],       # instance of: lighthouse
}


# ===========================================================================
# Code Execution (Sandboxed)
# ===========================================================================

def _execute_computation_code(code: str, timeout: int = 10) -> Optional[str]:
    """Execute computation code in a subprocess sandbox.

    Args:
        code: Python code string (must print the answer)
        timeout: Maximum execution time in seconds

    Returns:
        stdout string (the answer) or None on error/timeout.
    """
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


# ===========================================================================
# Phase 0: Domain + Answer Selection (Programmatic)
# ===========================================================================

def select_reasoning_context(entity_id, entity_label, quant_props, chains,
                             used_templates=None, theme=None):
    """Select a reasoning domain and template, fill parameters, compute gold answer.

    1. Find compatible domains via PROPERTY_DOMAIN_COMPATIBILITY
    2. Pick domain + template randomly (excluding used_templates)
    3. Fill template parameters from quant_props values
    4. Execute computation code -> gold_answer

    Args:
        entity_id: Wikidata entity ID
        entity_label: Human-readable name
        quant_props: Dict from fetch_quantitative_properties()
        chains: List of KG chain dicts (for metadata)
        used_templates: Set of (domain, template_id) already used
        theme: Theme key (e.g. 'rivers', 'dams') for template compatibility filtering

    Returns:
        Dict with domain, template_id, computation_code, gold_answer,
        reasoning_chain, required_facts, quant_data_text, question_hint, answer_unit
        or None if no valid template found.
    """
    if used_templates is None:
        used_templates = set()

    # Find which domains are compatible with available properties
    available_domains = set()
    for prop_id in quant_props:
        for domain in PROPERTY_DOMAIN_COMPATIBILITY.get(prop_id, []):
            available_domains.add(domain)

    if not available_domains:
        return None

    # Try each domain+template combo (randomized)
    TIME_VARYING_PROPERTIES = {"P1082", "P2131", "P2132"}
    candidates = []
    for domain_id in available_domains:
        domain = REASONING_DOMAINS.get(domain_id)
        if not domain:
            continue
        for tmpl_id, tmpl in domain["templates"].items():
            if (domain_id, tmpl_id) in used_templates:
                continue
            # Check required properties are available
            required = tmpl["required_properties"]
            if not all(p in quant_props for p in required):
                continue
            # Skip templates incompatible with this entity's theme
            compatible = tmpl.get("compatible_themes")
            if compatible and theme and theme not in compatible:
                continue
            # Skip templates using time-varying properties that lack a year qualifier
            has_undated_tv = any(
                p in TIME_VARYING_PROPERTIES and not quant_props[p].get("year")
                for p in required
            )
            if has_undated_tv:
                print(f"  Template {domain_id}/{tmpl_id}: skipping — "
                      f"time-varying property lacks year qualifier", flush=True)
                continue
            candidates.append((domain_id, tmpl_id, tmpl))

    if not candidates:
        return None

    random.shuffle(candidates)

    for domain_id, tmpl_id, tmpl in candidates:
        try:
            params = _fill_template_parameters(tmpl, quant_props, entity_label)
            if params is None:
                continue

            code = tmpl["code_template"].format(**params)
            gold_answer = _execute_computation_code(code)
            if gold_answer is None:
                continue

            # Validate the answer is reasonable
            try:
                float(gold_answer)
            except ValueError:
                continue

            # Build quant_data_text: entity's properties (for reference, NOT to be revealed)
            # Include year qualifier when available so the prompt references a specific year
            hide_props = set(tmpl.get("hide_properties", []))
            quant_lines = []
            data_year = None  # Track the year for the question hint
            for prop_id in tmpl["required_properties"]:
                qp = quant_props[prop_id]
                year = qp.get("year")
                year_str = f" (as of {year})" if year else ""
                if year:
                    data_year = year
                if prop_id not in hide_props:
                    quant_lines.append(f"- {qp['label']}: {qp['amount']} {qp['unit']}{year_str}")
            quant_data_text = "\n".join(quant_lines) if quant_lines else "(none)"

            # Build question hint with filled parameters (including data_year)
            # Note: undated time-varying templates are filtered out upstream,
            # so data_year is always a real year or None (for non-time-varying props)
            hint_params = {k: v for k, v in params.items()}
            hint_params["data_year"] = data_year if data_year else "Wikidata"
            question_hint = tmpl["question_hint"].format(**hint_params)

            # Separate external/hypothetical parameters from entity properties
            # Entity properties come from Wikidata and must be looked up by the solver
            entity_param_names = set()
            param_to_prop = {
                "population": "P1082", "area": "P2046", "gdp": "P2131",
                "gdp_per_capita": "P2132", "elevation": "P2044",
                "mass_kg": "P2067", "radius_m": "P2120", "density": "P2054",
                "lat1": "P625", "lon1": "P625",
                "length": "P2043", "height": "P2048", "floors": "P1101",
                "prominence": "P2660", "depth": "P4511",
                "crest_length": "P2787", "total_length": "P2787",
                "coastline": "P2660", "capacity": "P1083",
                "length_km": "P2787",
            }
            for pname in params:
                if pname in param_to_prop:
                    entity_param_names.add(pname)

            # External params: hypothetical values the question CAN include
            # Exclude reference city/area/river/height/elevation/mass coordinates and values — solver must look those up too
            external_params = {}
            ref_params = {"lat2", "lon2", "reference_city",
                          "reference_area", "reference_name",
                          "reference_length", "reference_river",
                          "reference_height", "reference_structure",
                          "reference_elevation", "reference_location",
                          "reference_mass", "reference_body",
                          "orbit_altitude_km"}
            for pname, pval in params.items():
                if pname not in entity_param_names and pname not in ref_params:
                    external_params[pname] = pval
            # Reference names only (no numeric values — solver must look those up)
            if "reference_city" in params:
                external_params["reference_city"] = params["reference_city"]
            if "reference_name" in params:
                external_params["reference_name"] = params["reference_name"]
            if "reference_river" in params:
                external_params["reference_river"] = params["reference_river"]
            if "reference_structure" in params:
                external_params["reference_structure"] = params["reference_structure"]
            if "reference_location" in params:
                external_params["reference_location"] = params["reference_location"]
            if "reference_body" in params:
                external_params["reference_body"] = params["reference_body"]

            # Collect entity property values + reference coordinates for leak detection
            entity_values = []
            for pname in entity_param_names:
                val = params[pname]
                if isinstance(val, float):
                    entity_values.append(val)
                elif isinstance(val, int):
                    entity_values.append(float(val))
            # Also prevent leaking reference coordinates and reference numeric values
            for ref_key in ("lat2", "lon2", "reference_area", "reference_length",
                            "reference_height", "reference_elevation", "reference_mass"):
                if ref_key in params:
                    try:
                        entity_values.append(float(params[ref_key]))
                    except (ValueError, TypeError):
                        pass

            reasoning_chain = (
                f"Step 1: Identify {entity_label} from clue facts. "
                f"Step 2: Look up quantitative properties from Wikidata/Wikipedia. "
                f"Step 3: Apply {REASONING_DOMAINS[domain_id]['label']} "
                f"template '{tmpl['label']}'. "
                f"Step 4: Compute → {gold_answer}"
            )

            return {
                "domain": domain_id,
                "domain_label": REASONING_DOMAINS[domain_id]["label"],
                "template_id": tmpl_id,
                "template_label": tmpl["label"],
                "computation_code": code,
                "gold_answer": gold_answer,
                "reasoning_chain": reasoning_chain,
                "quant_data_text": quant_data_text,
                "question_hint": question_hint,
                "data_year": data_year,
                "answer_unit": tmpl["answer_unit"],
                "parameters": params,
                "external_params": external_params,
                "entity_values": entity_values,
            }

        except Exception as e:
            print(f"  Template {domain_id}/{tmpl_id} failed: {e}", flush=True)
            continue

    return None


def _fill_none_defaults(params: Dict[str, Any]) -> None:
    """Replace ``None`` default values with randomly generated ones.

    Each parameter that a template declares with ``None`` as the default
    gets a plausible random value here so the computation code receives a
    concrete number.
    """
    _RANDOM_RANGES: Dict[str, tuple] = {
        # water_volume_lake
        "avg_depth": (5, 80),           # average lake depth in metres
        # volcanic_eruption_energy
        "vei_index": (2, 7),            # VEI integer 2-7
        # river_flow_rate
        "width_m": (50, 500),           # river cross-section width (m)
        "depth_m": (2, 20),             # river cross-section depth (m)
        "velocity_ms": (0.5, 3.0),      # flow velocity (m/s)
        # canal_transit_time
        "speed_knots": (4, 12),         # vessel speed in knots
        # stadium_capacity_density
        "footprint_m2": (15000, 80000), # stadium footprint (m²)
    }
    for pname, pval in list(params.items()):
        if pval is not None:
            continue
        rng = _RANDOM_RANGES.get(pname)
        if rng is None:
            continue
        lo, hi = rng
        if pname == "vei_index":
            # VEI should be an integer
            params[pname] = random.randint(int(lo), int(hi))
        elif isinstance(lo, int) and isinstance(hi, int):
            params[pname] = random.randint(lo, hi)
        else:
            params[pname] = round(random.uniform(lo, hi), 2)


def _fill_template_parameters(tmpl, quant_props, entity_label):
    """Fill template parameters from quantitative properties and defaults.

    Returns:
        Dict of parameter name -> value, or None if unable to fill.
    """
    params = dict(tmpl.get("defaults", {}))

    # Generate random values for defaults that are None
    _fill_none_defaults(params)

    for param_name in tmpl["parameters"]:
        if param_name in params:
            continue  # Already has a default

        # Map parameter names to property IDs
        param_to_prop = {
            "population": "P1082",
            "area": "P2046",
            "gdp": "P2131",
            "gdp_per_capita": "P2132",
            "elevation": "P2044",
            "mass_kg": "P2067",
            "radius_m": "P2120",
            "density": "P2054",
            "length": "P2043",
            "height": "P2048",
            "floors": "P1101",
            "prominence": "P2660",
            "depth": "P4511",
            "crest_length": "P2787",
            "total_length": "P2787",
            "coastline": "P2660",
            "capacity": "P1083",
            "length_km": "P2787",
        }

        prop_id = param_to_prop.get(param_name)
        if prop_id and prop_id in quant_props:
            qp = quant_props[prop_id]
            amount = qp["amount"]
            # Normalize to expected unit for computation templates
            if prop_id in _UNIT_NORMALIZATION:
                _, conversions = _UNIT_NORMALIZATION[prop_id]
                actual_unit = qp.get("unit", "")
                if actual_unit in conversions:
                    amount = amount / conversions[actual_unit]
            params[param_name] = amount
            continue

        # Reference area: pick a random reference for area_comparison
        if param_name == "reference_area":
            if "reference_name" not in params:
                ref_name = random.choice(list(REFERENCE_AREAS.keys()))
                params["reference_area"] = REFERENCE_AREAS[ref_name]
                params["reference_name"] = ref_name
            continue

        # Reference river: pick a random reference for length_comparison
        if param_name == "reference_length":
            if "reference_river" not in params:
                ref_name = random.choice(list(REFERENCE_RIVERS.keys()))
                params["reference_length"] = REFERENCE_RIVERS[ref_name]
                params["reference_river"] = ref_name
            continue

        # Reference height: pick a random reference for height_comparison
        if param_name == "reference_height":
            if "reference_structure" not in params:
                ref_name = random.choice(list(REFERENCE_HEIGHTS.keys()))
                params["reference_height"] = REFERENCE_HEIGHTS[ref_name]
                params["reference_structure"] = ref_name
            continue

        # Reference elevation: pick a random reference for elevation_comparison
        if param_name == "reference_elevation":
            if "reference_location" not in params:
                ref_name = random.choice(list(REFERENCE_ELEVATIONS.keys()))
                params["reference_elevation"] = REFERENCE_ELEVATIONS[ref_name]
                params["reference_location"] = ref_name
            continue

        # Reference mass: pick a random reference for mass_comparison
        if param_name == "reference_mass":
            if "reference_body" not in params:
                ref_name = random.choice(list(REFERENCE_MASSES.keys()))
                params["reference_mass"] = REFERENCE_MASSES[ref_name]
                params["reference_body"] = ref_name
            continue

        # Special handling for derived parameters
        if param_name == "target_population" and "population" in params:
            # Set target to ~107% of current population for interesting solve_for_time
            pop = params.get("population") or quant_props.get("P1082", {}).get("amount")
            if pop:
                # Round target to a nice number
                target = round(pop * random.uniform(1.05, 1.15), -int(math.log10(max(pop, 1))) + 2)
                params[param_name] = target
                continue

        if param_name == "target_population" and "P1082" in quant_props:
            pop = quant_props["P1082"]["amount"]
            target = round(pop * random.uniform(1.05, 1.15), -int(math.log10(max(pop, 1))) + 2)
            params[param_name] = target
            params.setdefault("population", pop)
            continue

        # Haversine: pick a reference city
        if param_name in ("lat1", "lon1") and "P625" in quant_props:
            coords = quant_props["P625"]["amount"]
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                params["lat1"] = coords[0]
                params["lon1"] = coords[1]
                continue

        if param_name in ("lat2", "lon2"):
            if "reference_city" not in params:
                city_name = random.choice(list(REFERENCE_CITIES.keys()))
                city_coords = REFERENCE_CITIES[city_name]
                params["lat2"] = city_coords[0]
                params["lon2"] = city_coords[1]
                params["reference_city"] = city_name
            continue

        if param_name == "surface_gravity":
            # Default is already in template defaults
            continue

        if param_name == "orbit_altitude":
            # Default is already in template defaults
            continue

        return None  # Cannot fill required parameter

    # Post-processing: compute derived display values for question hints
    if "orbit_altitude" in params:
        params["orbit_altitude_km"] = params["orbit_altitude"] / 1000

    return params


# ===========================================================================
# Gold Chain Builder
# ===========================================================================

def _build_gold_chain(entity_id, entity_label, quant_props, reasoning_ctx):
    """Build a gold chain listing all intermediate entities and values needed
    to reach the final answer.

    Returns:
        List of dicts, each representing one step in the reasoning chain.
        E.g. [
            {"step": "entity_identification", "entity": "Eswatini", "wikidata_id": "Q1050"},
            {"step": "property_lookup", "property": "population", "value": 1172000, "year": 2023, "source": "Wikidata P1082"},
            {"step": "external_parameter", "parameter": "growth_rate", "value": 0.02},
            {"step": "computation", "formula": "compound_growth", "result": "1430247"},
        ]
    """
    chain = []

    # Step 1: Entity identification
    chain.append({
        "step": "entity_identification",
        "entity": entity_label,
        "wikidata_id": entity_id,
    })

    # Step 2: Property lookups (values the solver must retrieve)
    params = reasoning_ctx.get("parameters", {})
    param_to_prop = {
        "population": "P1082", "area": "P2046", "gdp": "P2131",
        "gdp_per_capita": "P2132", "elevation": "P2044",
        "mass_kg": "P2067", "radius_m": "P2120", "density": "P2054",
        "lat1": "P625", "lon1": "P625",
        "length": "P2043", "height": "P2048", "floors": "P1101",
        "prominence": "P2660", "depth": "P4511",
        "crest_length": "P2787", "total_length": "P2787",
        "coastline": "P2660", "capacity": "P1083",
        "length_km": "P2787",
    }

    seen_props = set()
    for pname, prop_id in param_to_prop.items():
        if pname in params and prop_id not in seen_props and prop_id in quant_props:
            seen_props.add(prop_id)
            qp = quant_props[prop_id]
            step = {
                "step": "property_lookup",
                "property": qp["label"],
                "value": qp["amount"],
                "unit": qp.get("unit", ""),
                "source": f"Wikidata {prop_id}",
            }
            if qp.get("year"):
                step["year"] = qp["year"]
            chain.append(step)

    # Step 3: Reference lookups (haversine, area_comparison, length_comparison)
    if "reference_city" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_city"],
            "property": "coordinates",
            "value": {"lat": params.get("lat2"), "lon": params.get("lon2")},
            "unit": "degrees",
        })
    if "reference_name" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_name"],
            "property": "area",
            "value": params.get("reference_area"),
            "unit": "km²",
        })
    if "reference_river" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_river"],
            "property": "length",
            "value": params.get("reference_length"),
            "unit": "km",
        })
    if "reference_structure" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_structure"],
            "property": "height",
            "value": params.get("reference_height"),
            "unit": "m",
        })
    if "reference_location" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_location"],
            "property": "elevation",
            "value": params.get("reference_elevation"),
            "unit": "m",
        })
    if "reference_body" in params:
        chain.append({
            "step": "reference_lookup",
            "entity": params["reference_body"],
            "property": "mass",
            "value": params.get("reference_mass"),
            "unit": "kg",
        })

    # Step 4: External/hypothetical parameters
    external = reasoning_ctx.get("external_params", {})
    for pname, pval in external.items():
        if pname in ("reference_city", "reference_name", "reference_area",
                      "reference_river", "reference_length",
                      "reference_structure", "reference_height",
                      "reference_location", "reference_elevation",
                      "reference_body", "reference_mass"):
            continue  # Already handled above
        chain.append({
            "step": "external_parameter",
            "parameter": pname,
            "value": pval,
        })

    # Step 5: Computation
    chain.append({
        "step": "computation",
        "formula": reasoning_ctx.get("template_label", ""),
        "domain": reasoning_ctx.get("domain_label", ""),
        "result": reasoning_ctx.get("gold_answer", ""),
        "unit": reasoning_ctx.get("answer_unit", ""),
    })

    return chain
