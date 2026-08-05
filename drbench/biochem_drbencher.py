"""Biochemistry Benchmark: PubChem + UniProt data as ground truth.

Creates questions in a 2-level difficulty space:
  Level 1 (Single-Entity): identify entity from clues -> fetch data -> compute
  Level 2 (Multi-Step):    multi-step scientific calculations with known parameters

Data sources:
  Proteins:  UniProt REST API (MW, length, sequence, function)
  Compounds: PubChem PUG REST (MW, formula, XLogP, TPSA, HBA, HBD, etc.)
  Structures: RCSB PDB (resolution, atom count, cell dimensions)

Run via:  python -m drbench.biochem_drbencher --exp_mode biochem_bench ...
"""

import argparse
import copy
import datetime
import json
import math
import os
import random
import re
import sys
import time

import requests
import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

# OpenAI Harmony imports
from openai_harmony import (
    Message, SystemContent, Role, ReasoningEffort,
)

from .util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from .tool_util import (
    _generate_lm_answers, _generate_lm_answers_harmony, extract_json_v2,
    fetch_multihop_triples, fetch_wikipedia_for_entities,
)
from .harmony_vllm import HarmonyVLLMGenerator

# Reuse verification infrastructure (V1/V2) from bench_verify
from .bench_verify import (
    verify_bench_v1,
    _sample_concurrency,
    run_samples_concurrently,
)
from .harmony_base import (
    gen_from_prompt_harmony,
    get_harmony_generator,
    set_harmony_generator,
    shutdown_and_exit,
)
from .multiskill_utils import (
    run_phase3_validation,
    check_kg_uniqueness_comparative,
    strip_side_prefix,
    extract_chain_clue_facts,
    verify_facts_grounding,
    filter_poisoned_facts,
    poison_used_facts,
    filter_cross_entity_facts,
    compute_cci_fields,
)
from .wikidata_harmony import format_extracted_facts, parse_used_facts
from .multiskill_drbencher import (
    build_multiskill_grounding_articles,
)

# Bio utilities
from .bio_util import (
    get_protein_data,
    get_compound_data,
    get_organism_data,
    get_pdb_data,
    search_protein,
    search_compound,
    fetch_entity_clues,
    get_category_entities,
    resolve_biochem_wikidata_id,
    ENTITY_UNIVERSE,
    BIOCHEM_THEMES,
)

# Browser tool for V2 verification
try:
    from tools.kg_browser import MultiSourceKnowledgeBackend, MultiSourceKnowledgeBrowserTool
    _HAS_BROWSER = True
except ImportError:
    _HAS_BROWSER = False

# Python tool for V2 verification
try:
    from tools.hybrid_exec_qid import HybridPythonTool
    _HAS_PYTHON_TOOL = True
except ImportError:
    _HAS_PYTHON_TOOL = False

# Bio tool for V2 verification
try:
    from tools.bio_tool import BiochemTool
    _HAS_BIO_TOOL = True
except ImportError:
    _HAS_BIO_TOOL = False

# Diversity filter (optional)
try:
    from .diversity import diversity_filter, diversity_report
    _HAS_DIVERSITY = True
except ImportError:
    _HAS_DIVERSITY = False


# ===========================================================================
# Biochemistry Templates
# ===========================================================================

BIOCHEM_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --- Level 1: Single-Entity Computation (Protein) ---
    "protein_mw_to_kda": {
        "level": 1,
        "type": "single",
        "entity_type": "protein",
        "label": "Protein MW in kDa",
        "required_data": ["molecular_weight"],
        "code_template": "print(round({molecular_weight} / 1000, 2))",
        "question_hint": "What is the molecular weight of this protein in kilodaltons (kDa)?",
        "answer_unit": "kDa",
        "reasoning_depth": 2,
    },
    "protein_aa_percentage": {
        "level": 1,
        "type": "single",
        "entity_type": "protein",
        "label": "Amino Acid Percentage",
        "required_data": ["amino_acid_counts", "length"],
        "code_template": "print(round({target_aa_count} / {length} * 100, 2))",
        "question_hint": "What percentage of residues in this protein are {target_aa}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "protein_extinction_coefficient": {
        "level": 1,
        "type": "single",
        "entity_type": "protein",
        "label": "Extinction Coefficient Estimate",
        "required_data": ["amino_acid_counts"],
        "code_template": (
            "# Pace method: Ext = nTrp*5500 + nTyr*1490 + nCys*125\n"
            "ext = {trp_count} * 5500 + {tyr_count} * 1490 + {cys_count} * 125\n"
            "print(round(ext, 0))"
        ),
        "question_hint": "Estimate the extinction coefficient (M^-1 cm^-1) of this protein at 280nm using the Pace method.",
        "answer_unit": "M^-1 cm^-1",
        "reasoning_depth": 3,
    },

    # --- Level 1: Single-Entity Computation (Compound) ---
    "molar_concentration": {
        "level": 1,
        "type": "single",
        "entity_type": "compound",
        "label": "Molar Concentration",
        "required_data": ["molecular_weight"],
        "code_template": (
            "mass_g = {mass_g}\n"
            "volume_L = {volume_L}\n"
            "molarity = (mass_g / {molecular_weight}) / volume_L\n"
            "print(round(molarity, 4))"
        ),
        "question_hint": "If {mass_g} g of this compound is dissolved in {volume_L} L of solution, what is the molar concentration (mol/L)?",
        "answer_unit": "mol/L",
        "reasoning_depth": 2,
    },
    "dilution_factor": {
        "level": 1,
        "type": "single",
        "entity_type": "compound",
        "label": "Dilution Calculation (C1V1=C2V2)",
        "required_data": ["molecular_weight"],
        "code_template": (
            "C1 = {C1}\n"
            "V1 = {V1}\n"
            "V2 = {V2}\n"
            "C2 = C1 * V1 / V2\n"
            "print(round(C2, 4))"
        ),
        "question_hint": "A {C1} mol/L stock solution of this compound ({V1} mL) is diluted to {V2} mL. What is the final concentration (mol/L)?",
        "answer_unit": "mol/L",
        "reasoning_depth": 2,
    },
    "mass_from_moles": {
        "level": 1,
        "type": "single",
        "entity_type": "compound",
        "label": "Mass from Moles",
        "required_data": ["molecular_weight"],
        "code_template": (
            "moles = {moles}\n"
            "mass = moles * {molecular_weight}\n"
            "print(round(mass, 2))"
        ),
        "question_hint": "What is the mass (in grams) of {moles} moles of this compound?",
        "answer_unit": "g",
        "reasoning_depth": 2,
    },

    # --- Level 1: Cross-Entity ---
    "mw_ratio": {
        "level": 1,
        "type": "comparative",
        "entity_type": "compound",
        "label": "Molecular Weight Ratio",
        "required_data": ["molecular_weight"],
        "code_template": "print(round({molecular_weight_b} / {molecular_weight_a}, 4))",
        "question_hint": "How many times heavier is the second compound compared to the first (by molecular weight)?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "logP_interpretation": {
        "level": 1,
        "type": "single",
        "entity_type": "compound",
        "label": "LogP Hydrophilicity Classification",
        "required_data": ["xlogp"],
        "code_template": "print(round({xlogp}, 2))",
        "question_hint": "What is the XLogP value of this compound?",
        "answer_unit": "",
        "reasoning_depth": 1,
    },
    "rule_of_five_check": {
        "level": 1,
        "type": "single",
        "entity_type": "compound",
        "label": "Lipinski Rule of Five Violations",
        "required_data": ["molecular_weight", "xlogp", "hbond_donor_count", "hbond_acceptor_count"],
        "code_template": (
            "violations = 0\n"
            "if {molecular_weight} > 500: violations += 1\n"
            "if {xlogp} > 5: violations += 1\n"
            "if {hbond_donor_count} > 5: violations += 1\n"
            "if {hbond_acceptor_count} > 10: violations += 1\n"
            "print(violations)"
        ),
        "question_hint": "How many Lipinski Rule of Five violations does this compound have?",
        "answer_unit": "violations",
        "reasoning_depth": 2,
    },

    # --- Level 1: Comparative ---
    "mw_difference": {
        "level": 1,
        "type": "comparative",
        "entity_type": "compound",
        "label": "Molecular Weight Difference",
        "required_data": ["molecular_weight"],
        "code_template": "print(round(abs({molecular_weight_b} - {molecular_weight_a}), 2))",
        "question_hint": "What is the difference in molecular weight (g/mol) between these two compounds?",
        "answer_unit": "g/mol",
        "reasoning_depth": 2,
    },
    "polarity_comparison": {
        "level": 1,
        "type": "comparative",
        "entity_type": "compound",
        "label": "TPSA Comparison",
        "required_data": ["tpsa"],
        "code_template": "print(round(abs({tpsa_b} - {tpsa_a}), 2))",
        "question_hint": "What is the difference in topological polar surface area (TPSA, in A^2) between these two compounds?",
        "answer_unit": "A^2",
        "reasoning_depth": 2,
    },
    "protein_length_ratio": {
        "level": 1,
        "type": "comparative",
        "entity_type": "protein",
        "label": "Protein Sequence Length Ratio",
        "required_data": ["length"],
        "code_template": "print(round({length_b} / {length_a}, 4))",
        "question_hint": "How many times longer is the second protein's sequence compared to the first?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "hbond_capacity_comparison": {
        "level": 1,
        "type": "comparative",
        "entity_type": "compound",
        "label": "H-Bond Capacity Comparison",
        "required_data": ["hbond_donor_count", "hbond_acceptor_count"],
        "code_template": (
            "cap_a = {hbond_donor_count_a} + {hbond_acceptor_count_a}\n"
            "cap_b = {hbond_donor_count_b} + {hbond_acceptor_count_b}\n"
            "print(round(abs(cap_b - cap_a), 0))"
        ),
        "question_hint": "What is the difference in total H-bond capacity (donors + acceptors) between these two compounds?",
        "answer_unit": "bonds",
        "reasoning_depth": 3,
    },

    # --- Level 2: Multi-Step Scientific ---
    "henderson_hasselbalch": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Henderson-Hasselbalch pH",
        "required_data": ["molecular_weight"],
        "code_template": (
            "import math\n"
            "pKa = {pKa}\n"
            "ratio = {acid_base_ratio}  # [A-]/[HA]\n"
            "pH = pKa + math.log10(ratio)\n"
            "print(round(pH, 2))"
        ),
        "question_hint": "Using the Henderson-Hasselbalch equation with pKa={pKa} and [A-]/[HA]={acid_base_ratio}, what is the pH of a buffer solution of this compound?",
        "answer_unit": "",
        "reasoning_depth": 3,
    },
    "beer_lambert": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Beer-Lambert Concentration",
        "required_data": ["molecular_weight"],
        "code_template": (
            "absorbance = {absorbance}\n"
            "epsilon = {epsilon}  # molar absorptivity\n"
            "path_length = {path_length}  # cm\n"
            "concentration = absorbance / (epsilon * path_length)\n"
            "print(round(concentration, 6))"
        ),
        "question_hint": "Using Beer-Lambert law (A=epsilon*c*l), if a solution of this compound has absorbance {absorbance} with epsilon={epsilon} M^-1 cm^-1 and path length {path_length} cm, what is the concentration (mol/L)?",
        "answer_unit": "mol/L",
        "reasoning_depth": 3,
    },
    "michaelis_menten": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Michaelis-Menten Rate",
        "required_data": ["molecular_weight"],
        "code_template": (
            "Vmax = {Vmax}  # umol/min\n"
            "Km = {Km}  # mM\n"
            "S = {substrate_conc}  # mM\n"
            "v = Vmax * S / (Km + S)\n"
            "print(round(v, 4))"
        ),
        "question_hint": "Using Michaelis-Menten kinetics with Vmax={Vmax} umol/min and Km={Km} mM, what is the reaction rate (umol/min) at [S]={substrate_conc} mM?",
        "answer_unit": "umol/min",
        "reasoning_depth": 3,
    },
    "gibbs_free_energy": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Gibbs Free Energy",
        "required_data": ["molecular_weight"],
        "code_template": (
            "delta_H = {delta_H}  # kJ/mol\n"
            "delta_S = {delta_S}  # J/(mol*K)\n"
            "T = {temperature}  # K\n"
            "delta_G = delta_H - T * delta_S / 1000  # convert S to kJ\n"
            "print(round(delta_G, 2))"
        ),
        "question_hint": "Given DeltaH={delta_H} kJ/mol and DeltaS={delta_S} J/(mol*K), what is the Gibbs free energy change (kJ/mol) at {temperature} K?",
        "answer_unit": "kJ/mol",
        "reasoning_depth": 3,
    },
    "nernst_equation": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Nernst Equation Potential",
        "required_data": ["molecular_weight"],
        "code_template": (
            "import math\n"
            "E_standard = {E_standard}  # V\n"
            "n = {n_electrons}  # electrons transferred\n"
            "T = {temperature}  # K\n"
            "Q = {reaction_quotient}\n"
            "R = 8.314  # J/(mol*K)\n"
            "F = 96485  # C/mol\n"
            "E = E_standard - (R * T) / (n * F) * math.log(Q)\n"
            "print(round(E, 4))"
        ),
        "question_hint": "Using the Nernst equation with E_standard={E_standard} V, n={n_electrons} electrons, T={temperature} K, and Q={reaction_quotient}, what is the cell potential (V)?",
        "answer_unit": "V",
        "reasoning_depth": 3,
    },
    "drug_dosage": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Drug Dosage Calculation",
        "required_data": ["molecular_weight"],
        "code_template": (
            "MW = {molecular_weight}  # g/mol\n"
            "patient_weight = {patient_weight}  # kg\n"
            "target_concentration = {target_concentration}  # umol/L\n"
            "volume_of_distribution = {vd}  # L/kg\n"
            "total_volume = patient_weight * volume_of_distribution  # L\n"
            "moles_needed = target_concentration * 1e-6 * total_volume  # mol\n"
            "dose_mg = moles_needed * MW * 1000  # mg\n"
            "print(round(dose_mg, 2))"
        ),
        "question_hint": "For a {patient_weight} kg patient with Vd={vd} L/kg, what dose (mg) of this compound is needed to achieve a plasma concentration of {target_concentration} umol/L?",
        "answer_unit": "mg",
        "reasoning_depth": 3,
    },

    # --- Level 2: Composite ---
    "therapeutic_index": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Therapeutic Index",
        "required_data": ["molecular_weight"],
        "code_template": (
            "LD50 = {LD50}\n"
            "ED50 = {ED50}\n"
            "TI = LD50 / ED50\n"
            "print(round(TI, 2))"
        ),
        "question_hint": "Given LD50={LD50} mg/kg and ED50={ED50} mg/kg, what is the therapeutic index of this drug?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "binding_energy_from_kd": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Binding Energy from Kd",
        "required_data": ["molecular_weight"],
        "code_template": (
            "import math\n"
            "Kd = {Kd}  # M (molar)\n"
            "T = {temperature}  # K\n"
            "R = 8.314  # J/(mol*K)\n"
            "delta_G = R * T * math.log(Kd) / 1000  # kJ/mol\n"
            "print(round(delta_G, 2))"
        ),
        "question_hint": "Given Kd={Kd} M at {temperature} K, what is the binding free energy (kJ/mol)?",
        "answer_unit": "kJ/mol",
        "reasoning_depth": 3,
    },
    # --- Additional Level 2: Scientific ---
    "enzyme_turnover_number": {
        "level": 2,
        "type": "single",
        "entity_type": "protein",
        "label": "Enzyme Turnover Number (kcat)",
        "required_data": ["molecular_weight"],
        "code_template": (
            "Vmax = {Vmax}  # umol/min\n"
            "E_total = {enzyme_concentration}  # umol\n"
            "kcat = Vmax / E_total\n"
            "print(round(kcat, 2))"
        ),
        "question_hint": "Given Vmax={Vmax} umol/min and total enzyme concentration {enzyme_concentration} umol, what is the turnover number (kcat, in min^-1)?",
        "answer_unit": "min^-1",
        "reasoning_depth": 2,
    },
    "protein_isoelectric_point": {
        "level": 2,
        "type": "single",
        "entity_type": "protein",
        "label": "Isoelectric Point Estimate",
        "required_data": ["amino_acid_counts"],
        "code_template": (
            "# Simple pI estimate: average of pKa values of ionizable groups\n"
            "asp_count = {asp_count}\n"
            "glu_count = {glu_count}\n"
            "lys_count = {lys_count}\n"
            "arg_count = {arg_count}\n"
            "his_count = {his_count}\n"
            "acidic = asp_count + glu_count\n"
            "basic = lys_count + arg_count + his_count\n"
            "if acidic > basic:\n"
            "    pI = 4.0 + (basic / acidic) * 2\n"
            "elif basic > acidic:\n"
            "    pI = 8.0 + (1 - acidic / basic) * 2\n"
            "else:\n"
            "    pI = 7.0\n"
            "print(round(pI, 2))"
        ),
        "question_hint": "Estimate the isoelectric point (pI) of this protein based on its amino acid composition.",
        "answer_unit": "",
        "reasoning_depth": 3,
    },
    "osmolarity_calculation": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Osmolarity of Solution",
        "required_data": ["molecular_weight"],
        "code_template": (
            "mass_g = {mass_g}\n"
            "MW = {molecular_weight}\n"
            "volume_L = {volume_L}\n"
            "i = {van_hoff_factor}\n"
            "molarity = mass_g / MW / volume_L\n"
            "osmolarity = molarity * i\n"
            "print(round(osmolarity, 4))"
        ),
        "question_hint": "If {mass_g} g of this compound (van't Hoff factor i={van_hoff_factor}) is dissolved in {volume_L} L, what is the osmolarity (Osm/L)?",
        "answer_unit": "Osm/L",
        "reasoning_depth": 3,
    },
    "buffer_capacity": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Buffer Capacity",
        "required_data": ["molecular_weight"],
        "code_template": (
            "import math\n"
            "C = {buffer_concentration}  # mol/L\n"
            "pKa = {pKa}\n"
            "pH = {pH}\n"
            "x = 10 ** (pH - pKa)\n"
            "beta = 2.303 * C * x / (1 + x) ** 2\n"
            "print(round(beta, 6))"
        ),
        "question_hint": "At pH={pH}, what is the buffer capacity (mol/L) of a {buffer_concentration} M solution of this compound (pKa={pKa})?",
        "answer_unit": "mol/L",
        "reasoning_depth": 3,
    },
    "protein_charge_at_ph": {
        "level": 2,
        "type": "single",
        "entity_type": "protein",
        "label": "Estimated Net Charge at pH",
        "required_data": ["amino_acid_counts"],
        "code_template": (
            "import math\n"
            "pH = {pH}\n"
            "# Ionizable residues and their pKa\n"
            "n_asp = {asp_count}; n_glu = {glu_count}\n"
            "n_lys = {lys_count}; n_arg = {arg_count}; n_his = {his_count}\n"
            "n_cys = {cys_count}; n_tyr = {tyr_count}\n"
            "# Henderson-Hasselbalch: fraction charged\n"
            "def neg(pKa, n): return -n / (1 + 10**(pKa - pH))\n"
            "def pos(pKa, n): return n / (1 + 10**(pH - pKa))\n"
            "charge = (pos(10.5, n_lys) + pos(12.5, n_arg) + pos(6.0, n_his)\n"
            "         + neg(3.9, n_asp) + neg(4.1, n_glu) + neg(8.3, n_cys)\n"
            "         + neg(10.1, n_tyr) + pos(8.0, 1) + neg(3.1, 1))\n"
            "print(round(charge, 2))"
        ),
        "question_hint": "What is the estimated net charge of this protein at pH {pH}?",
        "answer_unit": "",
        "reasoning_depth": 4,
    },
    "half_life_remaining": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Fraction Remaining After Time",
        "required_data": ["molecular_weight"],
        "code_template": (
            "import math\n"
            "t_half = {half_life_hours}\n"
            "t = {elapsed_hours}\n"
            "fraction = 0.5 ** (t / t_half)\n"
            "print(round(fraction * 100, 2))"
        ),
        "question_hint": "If this compound has a half-life of {half_life_hours} hours, what percentage remains after {elapsed_hours} hours?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "competitive_inhibition_rate": {
        "level": 2,
        "type": "single",
        "entity_type": "compound",
        "label": "Rate with Competitive Inhibitor",
        "required_data": ["molecular_weight"],
        "code_template": (
            "Vmax = {Vmax}\n"
            "Km = {Km}\n"
            "S = {substrate_conc}\n"
            "I = {inhibitor_conc}\n"
            "Ki = {Ki}\n"
            "Km_app = Km * (1 + I / Ki)\n"
            "v = Vmax * S / (Km_app + S)\n"
            "print(round(v, 4))"
        ),
        "question_hint": "With competitive inhibitor at {inhibitor_conc} mM (Ki={Ki} mM), Vmax={Vmax} umol/min, Km={Km} mM, and [S]={substrate_conc} mM, what is the reaction rate?",
        "answer_unit": "umol/min",
        "reasoning_depth": 4,
    },

    # --- Organism Templates ---
    "genome_size_mbp": {
        "level": 1,
        "type": "single",
        "entity_type": "organism",
        "label": "Genome Size in Mbp",
        "required_data": ["genome_size"],
        "code_template": "print(round({genome_size} / 1e6, 2))",
        "question_hint": "What is the genome size of this organism in megabase pairs (Mbp)?",
        "answer_unit": "Mbp",
        "reasoning_depth": 2,
    },
    "gene_density": {
        "level": 1,
        "type": "single",
        "entity_type": "organism",
        "label": "Gene Density (genes per Mbp)",
        "required_data": ["genome_size", "gene_count"],
        "code_template": "print(round({gene_count} / ({genome_size} / 1e6), 2))",
        "question_hint": "What is the gene density (genes per Mbp) of this organism?",
        "answer_unit": "genes/Mbp",
        "reasoning_depth": 2,
    },
    "gc_to_at_ratio": {
        "level": 1,
        "type": "single",
        "entity_type": "organism",
        "label": "GC to AT Ratio",
        "required_data": ["gc_content"],
        "code_template": "print(round({gc_content} / (100 - {gc_content}), 4))",
        "question_hint": "What is the GC-to-AT ratio of this organism's genome?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "coding_fraction": {
        "level": 1,
        "type": "single",
        "entity_type": "organism",
        "label": "Protein-Coding Gene Fraction",
        "required_data": ["protein_coding_genes", "gene_count"],
        "code_template": "print(round({protein_coding_genes} / {gene_count} * 100, 2))",
        "question_hint": "What percentage of this organism's genes are protein-coding?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "genome_size_gb": {
        "level": 1,
        "type": "single",
        "entity_type": "organism",
        "label": "Genome Size in Gbp",
        "required_data": ["genome_size"],
        "code_template": "print(round({genome_size} / 1e9, 4))",
        "question_hint": "What is the genome size of this organism in gigabase pairs (Gbp)?",
        "answer_unit": "Gbp",
        "reasoning_depth": 2,
    },

    # --- Organism Comparative ---
    "genome_size_ratio": {
        "level": 1,
        "type": "comparative",
        "entity_type": "organism",
        "label": "Genome Size Ratio",
        "required_data": ["genome_size"],
        "code_template": "print(round({genome_size_b} / {genome_size_a}, 4))",
        "question_hint": "How many times larger is the second organism's genome compared to the first?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "gc_content_difference": {
        "level": 1,
        "type": "comparative",
        "entity_type": "organism",
        "label": "GC Content Difference",
        "required_data": ["gc_content"],
        "code_template": "print(round(abs({gc_content_b} - {gc_content_a}), 2))",
        "question_hint": "What is the difference in GC content (%) between these two organisms?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
}


# ===========================================================================
# Prompts
# ===========================================================================

BIOCHEM_QA_DEVELOPER = (
    "You are a biochemistry expert creating research questions that test "
    "the ability to identify biological/chemical entities from descriptions "
    "and perform quantitative calculations with their properties."
)

BIOCHEM_QA_PROMPT = """Compose a biochemistry question that:
1. Uses clue facts to describe an unnamed entity (protein, compound, or organism) — readers must figure out which entity
2. Then asks for a specific quantitative computation requiring the solver to look up the entity's data

CLUE FACTS (about the unnamed entity — do NOT name it directly):
{facts_text}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 2-4 sentences total.
- First 1-2 sentences: describe the entity using 3+ clue facts from different topics, without naming it directly.
- Last 1-2 sentences: pose the quantitative question.
- CRITICAL: Do NOT reveal ANY quantitative values from the entity's data (MW, length, LogP, etc.) in the question — the solver must look up ALL data themselves.
- CRITICAL: Do NOT name the entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real biochemistry research question.
- Include all necessary parameters for the computation (e.g., mass, volume, temperature) — these are NOT entity-specific data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your biochemistry question>
Used_Facts: <comma-separated fact_ids used, e.g. W1, W2, W3>
Reasoning: <brief chain: clues identify entity -> look up data -> computation gives answer>
"""

BIOCHEM_COMPARATIVE_QA_PROMPT = """Compose a comparative biochemistry question that:
1. Describes TWO unnamed entities (proteins, compounds, or organisms) using clue facts about each
2. Asks for a specific quantitative comparison requiring the solver to look up data for both

CLUE FACTS FOR ENTITY A (do NOT name this entity directly):
{facts_text_a}

CLUE FACTS FOR ENTITY B (do NOT name this entity directly):
{facts_text_b}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 4-6 sentences total.
- First 1-2 sentences: describe the first entity using 3+ clue facts from different topics, without naming it directly.
- Next 1-2 sentences: describe the second entity using 3+ clue facts from different topics, without naming it directly.
- Last 1-2 sentences: pose the comparative quantitative question.
- Refer to them as "the first protein/compound/organism" and "the second protein/compound/organism" (or similar distinct references).
- CRITICAL: Do NOT reveal ANY quantitative values from either entity's data (MW, length, LogP, etc.) in the question — the solver must look up ALL data themselves.
- CRITICAL: Do NOT name either entity directly anywhere in the question. Use descriptive clues only.
- Include all necessary parameters for the computation (mass, volume, temperature, etc.) — these are NOT entity-specific data.
- Sound natural and conversational, like a real biochemistry research question.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your comparative biochemistry question>
Used_Facts_A: <comma-separated fact_ids for entity A, e.g. A_C1_F1, A_C2_F1>
Used_Facts_B: <comma-separated fact_ids for entity B, e.g. B_C1_F1, B_C2_F1>
Reasoning: <brief chain: clues identify both entities -> look up data -> comparison gives answer>

REQUIREMENTS:
- Use 3+ clue facts per entity from different chains
- Do NOT name either entity in the question
- Do NOT include quantitative entity data in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST have exactly ONE unambiguous answer"""

BIOCHEM_V1_PROMPT = """Solve this biochemistry question. It requires:
1. Identifying a biological or chemical entity from its description
2. Looking up its quantitative properties (molecular weight, sequence, etc.)
3. Performing the requested computation

Provide ONLY the final numerical answer (with units if applicable).
Do not show your work.

Question: {question}

Answer:"""

BIOCHEM_V2_DEVELOPER = """You are an expert biochemistry research assistant.
You have THREE tools available:
- **Bio tool**: Query UniProt (proteins), PubChem (compounds), PDB (structures)
  - bio.search_protein(query) — find proteins by name/function
  - bio.get_protein(uniprot_id) — get protein properties (MW, length, sequence, etc.)
  - bio.search_compound(query) — find compounds by name
  - bio.get_compound(name_or_cid) — get compound properties (MW, formula, LogP, etc.)
  - bio.get_structure(pdb_id) — get crystal structure data
  - bio.compare_compounds(names, property) — compare compounds on a metric
- **Browser tool**: Search Wikipedia for entity identification
- **Python tool**: Execute calculations (math, numpy available)

Recommended approach:
1. Use the browser to search for and identify the entity from the description clues
2. Use the bio tool to look up quantitative properties
3. Use Python for computation
Give your final answer as a single number (with units if applicable) on the last line."""

# ===========================================================================
# Answer Checker
# ===========================================================================

def is_biochem_answer_correct(predicted, gold, tolerance=0.05):
    """Biochemistry answer comparison with 5% tolerance.

    Handles: numbers with units (kDa, g/mol, mol/L), percentages,
    scientific notation, integer violations, ratios.

    Args:
        predicted: Predicted answer string.
        gold: Gold answer string.
        tolerance: Relative tolerance (default 5%).

    Returns:
        True if answer is correct within tolerance.
    """
    if not predicted or not predicted.strip():
        return False

    pred_clean = _normalize_biochem_text(predicted)
    gold_clean = _normalize_biochem_text(gold)

    # 1. Exact match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric comparison
    pred_num = _try_parse_biochem_number(pred_clean)
    gold_num = _try_parse_biochem_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num == 0:
            return abs(pred_num) < 0.01
        rel_error = abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num))
        if rel_error <= tolerance:
            return True

    return False


def _normalize_biochem_text(text):
    """Normalize text for biochemistry comparison."""
    text = text.strip()
    # Normalize Unicode whitespace (narrow no-break space, non-breaking space, etc.)
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Normalize Unicode superscript minus signs
    text = text.replace('\u207b', '^-').replace('\u00b9', '1')
    # Strip units longest-first to avoid partial matches (e.g. "M" inside "M^-1 cm^-1")
    # Multi-char suffixes: safe to use str.replace
    for suffix in ["M^-1 cm^-1", "M^-1cm^-1", "M⁻¹ cm⁻¹", "M⁻¹cm⁻¹",
                    "J/(mol*K)", "umol/min", "μmol/min",
                    "kJ/mol", "g/mol", "mol/L",
                    "kDa", "Da", "mg", "A^2", "Å²",
                    "bonds", "violations", "times", "ratio"]:
        text = text.replace(suffix, "").strip()
    # Single-char suffixes: only strip at word boundary (end of string or before whitespace)
    for suffix in ["%", "M", "L", "g", "V"]:
        text = re.sub(rf'(?<=[\d\s]){re.escape(suffix)}(?:\s|$)', '', text).strip()
    text = text.replace(",", "")
    text = text.strip().strip("'\"")
    return text


def _try_parse_biochem_number(text):
    """Parse a number from biochemistry text. Returns float or None."""
    text = text.strip()
    # Normalize Unicode whitespace before parsing
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Handle scientific notation variants: "2.4 x 10^-4", "2.4×10^-4", etc.
    text = re.sub(r'\s*[×x]\s*10\^', 'e', text)

    multipliers = {
        "million": 1e6, "thousand": 1e3, "kilo": 1e3,
        "micro": 1e-6, "nano": 1e-9, "milli": 1e-3,
    }
    for word, mult in multipliers.items():
        # Word-boundary match to avoid "milli" matching "millimolar"
        if re.search(rf'\b{word}\b', text, re.IGNORECASE):
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE).strip()
            try:
                return float(text) * mult
            except ValueError:
                continue

    try:
        return float(text)
    except ValueError:
        match = re.search(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None


def _llm_judge_biochem_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold for biochem questions."""
    prompt = f"""Compare these two answers to the same biochemistry question.
The gold answer is computed from verified API data (UniProt/PubChem).
The predicted answer may use different units, rounding, or phrasing.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Are these answers equivalent (same value within ~5% tolerance, possibly different formatting)?
Respond with exactly YES or NO."""

    try:
        response = gen_from_prompt_harmony(prompt, temperature=0.0, max_tokens=10)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


# ===========================================================================
# QA Generation Pipeline
# ===========================================================================

def _execute_computation_code(code: str) -> Optional[str]:
    """Execute Python computation code and return printed output."""
    import io
    import contextlib
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {"__builtins__": __builtins__, "math": math, "round": round, "abs": abs})
        return f.getvalue().strip()
    except Exception:
        return None


def _biochem_entity_names(name: str, data: dict) -> List[str]:
    """Build extended entity identifiers for name-leak detection.

    Includes the entity name plus IUPAC name and molecular formula from
    PubChem data so that these synonyms are also caught as leaks.
    """
    names = [name]
    iupac = data.get("iupac_name")
    if iupac and isinstance(iupac, str) and len(iupac) >= 5:
        names.append(iupac)
    formula = data.get("molecular_formula")
    if formula and isinstance(formula, str) and len(formula) >= 4:
        names.append(formula)
    return names


def _generate_template_params(template_id: str, tmpl: dict,
                              entity_data: dict) -> Optional[Dict[str, Any]]:
    """Generate random parameters for level-2 templates that need extra params.

    Returns dict of parameters to fill the code_template, or None if not applicable.
    """
    params = {}

    if template_id == "molar_concentration":
        params["mass_g"] = round(random.uniform(0.5, 50.0), 1)
        params["volume_L"] = round(random.choice([0.1, 0.25, 0.5, 1.0, 2.0]), 2)
    elif template_id == "dilution_factor":
        params["C1"] = round(random.uniform(0.01, 1.0), 3)
        params["V1"] = random.choice([1, 5, 10, 25, 50])
        params["V2"] = params["V1"] * random.choice([2, 5, 10, 20])
    elif template_id == "mass_from_moles":
        params["moles"] = round(random.uniform(0.01, 5.0), 2)
    elif template_id == "henderson_hasselbalch":
        params["pKa"] = round(random.uniform(2.0, 12.0), 1)
        params["acid_base_ratio"] = round(random.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]), 1)
    elif template_id == "beer_lambert":
        params["absorbance"] = round(random.uniform(0.1, 2.0), 2)
        params["epsilon"] = random.choice([500, 1000, 5000, 10000, 18000, 50000])
        params["path_length"] = random.choice([1.0, 2.0, 5.0])
    elif template_id == "michaelis_menten":
        params["Vmax"] = round(random.uniform(10, 500), 1)
        params["Km"] = round(random.uniform(0.1, 50.0), 1)
        params["substrate_conc"] = round(random.uniform(0.05, 100.0), 1)
    elif template_id == "gibbs_free_energy":
        params["delta_H"] = round(random.uniform(-200, 200), 1)
        params["delta_S"] = round(random.uniform(-500, 500), 1)
        params["temperature"] = random.choice([273, 298, 310, 373])
    elif template_id == "nernst_equation":
        params["E_standard"] = round(random.uniform(-1.5, 1.5), 3)
        params["n_electrons"] = random.choice([1, 2, 3, 4])
        params["temperature"] = random.choice([298, 310, 273])
        params["reaction_quotient"] = round(random.choice([0.001, 0.01, 0.1, 1, 10, 100, 1000]), 3)
    elif template_id == "drug_dosage":
        params["patient_weight"] = random.choice([50, 60, 70, 80, 90, 100])
        params["target_concentration"] = round(random.uniform(1, 100), 1)
        params["vd"] = round(random.uniform(0.1, 2.0), 1)
    elif template_id == "therapeutic_index":
        params["ED50"] = round(random.uniform(1, 100), 1)
        params["LD50"] = round(params["ED50"] * random.uniform(2, 50), 1)
    elif template_id == "binding_energy_from_kd":
        params["Kd"] = round(random.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]), 10)
        params["temperature"] = random.choice([298, 310, 273])
    elif template_id == "protein_aa_percentage":
        aa_counts = entity_data.get("amino_acid_counts", {})
        if not aa_counts:
            return None
        # Pick a random amino acid that's present
        present_aas = [(aa, count) for aa, count in aa_counts.items() if count > 0]
        if not present_aas:
            return None
        target_aa, target_count = random.choice(present_aas)
        params["target_aa"] = target_aa
        params["target_aa_count"] = target_count
    elif template_id == "protein_extinction_coefficient":
        aa_counts = entity_data.get("amino_acid_counts", {})
        if not aa_counts:
            return None
        params["trp_count"] = aa_counts.get("W", 0)
        params["tyr_count"] = aa_counts.get("Y", 0)
        params["cys_count"] = aa_counts.get("C", 0)
        # Need at least some aromatic residues
        if params["trp_count"] + params["tyr_count"] == 0:
            return None
    elif template_id == "protein_isoelectric_point":
        aa_counts = entity_data.get("amino_acid_counts", {})
        if not aa_counts:
            return None
        params["asp_count"] = aa_counts.get("D", 0)
        params["glu_count"] = aa_counts.get("E", 0)
        params["lys_count"] = aa_counts.get("K", 0)
        params["arg_count"] = aa_counts.get("R", 0)
        params["his_count"] = aa_counts.get("H", 0)
        if params["asp_count"] + params["glu_count"] + params["lys_count"] + params["arg_count"] == 0:
            return None
    elif template_id == "protein_charge_at_ph":
        aa_counts = entity_data.get("amino_acid_counts", {})
        if not aa_counts:
            return None
        params["asp_count"] = aa_counts.get("D", 0)
        params["glu_count"] = aa_counts.get("E", 0)
        params["lys_count"] = aa_counts.get("K", 0)
        params["arg_count"] = aa_counts.get("R", 0)
        params["his_count"] = aa_counts.get("H", 0)
        params["cys_count"] = aa_counts.get("C", 0)
        params["tyr_count"] = aa_counts.get("Y", 0)
        params["pH"] = round(random.uniform(2.0, 12.0), 1)
    elif template_id == "enzyme_turnover_number":
        params["Vmax"] = round(random.uniform(10, 500), 1)
        params["enzyme_concentration"] = round(random.uniform(0.01, 5.0), 2)
    elif template_id == "osmolarity_calculation":
        params["mass_g"] = round(random.uniform(0.5, 50.0), 1)
        params["volume_L"] = round(random.choice([0.1, 0.25, 0.5, 1.0, 2.0]), 2)
        params["van_hoff_factor"] = random.choice([1, 2, 3])
    elif template_id == "buffer_capacity":
        params["buffer_concentration"] = round(random.choice([0.01, 0.05, 0.1, 0.5, 1.0]), 2)
        params["pKa"] = round(random.uniform(2.0, 12.0), 1)
        # Constrain pH near pKa so buffer capacity is meaningfully nonzero
        # ±1.5 keeps 10^|offset| ≤ ~32, avoiding degenerate near-zero answers
        pH_offset = random.uniform(-1.5, 1.5)
        params["pH"] = round(max(2.0, min(12.0, params["pKa"] + pH_offset)), 1)
    elif template_id == "half_life_remaining":
        params["half_life_hours"] = round(random.choice([0.5, 1, 2, 4, 6, 8, 12, 24]), 1)
        # Constrain elapsed to 0.5–5 half-lives so remaining% is 3–71% (not degenerate)
        hl = params["half_life_hours"]
        params["elapsed_hours"] = round(random.uniform(0.5 * hl, 5.0 * hl), 1)
    elif template_id == "competitive_inhibition_rate":
        params["Vmax"] = round(random.uniform(10, 500), 1)
        params["Km"] = round(random.uniform(0.1, 50.0), 1)
        params["substrate_conc"] = round(random.uniform(0.05, 100.0), 1)
        params["inhibitor_conc"] = round(random.uniform(0.1, 50.0), 1)
        params["Ki"] = round(random.uniform(0.01, 10.0), 2)

    return params


def select_biochem_template(entity_data: dict, entity_type: str,
                            used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a template that the entity's data can support.

    Args:
        entity_data: Entity data dict (from get_protein_data/get_compound_data).
        entity_type: "protein" or "compound".
        used_templates: Set of template IDs already used.

    Returns:
        Dict with template info and computed gold answer, or None.
    """
    template_ids = list(BIOCHEM_TEMPLATES.keys())
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = BIOCHEM_TEMPLATES[tmpl_id]

        # Skip comparatives (handled separately)
        if tmpl["type"] == "comparative":
            continue

        # Check entity type compatibility
        if tmpl["entity_type"] != entity_type:
            continue

        # Check required data availability
        required = tmpl["required_data"]
        all_available = True
        base_params = {}
        for key in required:
            val = entity_data.get(key)
            if val is None or (isinstance(val, (int, float)) and val == 0):
                all_available = False
                break
            if isinstance(val, dict):
                continue  # Complex types like amino_acid_counts handled in param generation
            base_params[key] = val

        if not all_available:
            continue

        # Generate extra parameters for level-2 templates
        extra_params = _generate_template_params(tmpl_id, tmpl, entity_data)
        if extra_params is None:
            continue

        params = {**base_params, **extra_params}

        # Fill code template and compute gold answer
        try:
            code = tmpl["code_template"].format(**params)
        except KeyError:
            continue

        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        # Validate gold answer
        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if abs(gold_float) < 1e-4:
                continue  # Near-zero answers are degenerate (correct regardless of entity)
        except ValueError:
            continue

        # Fill question hint
        try:
            question_hint = tmpl["question_hint"].format(**params)
        except KeyError:
            question_hint = tmpl["question_hint"]

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "question_hint": question_hint,
        }

    return None


def select_comparative_template(data_a: dict, data_b: dict, entity_type: str,
                                 used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a comparative template for two entities.

    Returns:
        Template context dict with gold answer, or None.
    """
    template_ids = [tid for tid, t in BIOCHEM_TEMPLATES.items()
                    if t["type"] == "comparative" and t["entity_type"] == entity_type]
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = BIOCHEM_TEMPLATES[tmpl_id]
        required = tmpl["required_data"]

        params = {}
        all_available = True
        for key in required:
            val_a = data_a.get(key)
            val_b = data_b.get(key)
            if val_a is None or val_b is None:
                all_available = False
                break
            if isinstance(val_a, (int, float)) and val_a == 0:
                all_available = False
                break
            if isinstance(val_b, (int, float)) and val_b == 0:
                all_available = False
                break
            params[f"{key}_a"] = val_a
            params[f"{key}_b"] = val_b

        if not all_available:
            continue

        try:
            code = tmpl["code_template"].format(**params)
        except KeyError:
            continue

        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if abs(gold_float) < 1e-4:
                continue  # Near-zero answers are degenerate (correct regardless of entity)
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "question_hint": tmpl["question_hint"],
        }

    return None


def compose_biochem_question(clues, reasoning_ctx, entity_name,
                              prev_questions):
    """Compose a biochemistry QA question using LLM.

    Args:
        clues: List of clue fact dicts from fetch_entity_clues().
        reasoning_ctx: Template context from select_*_template().
        entity_name: Entity name (for validation only — should NOT appear in question).
        prev_questions: List of previously generated question strings.

    Returns:
        Dict with "question", "used_facts" keys, or None on failure.
    """
    facts_text = "\n".join(
        f"[{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) {c['fact']}"
        for c in clues
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    prompt = BIOCHEM_QA_PROMPT.format(
        facts_text=facts_text,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=BIOCHEM_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Composition LLM error: {e}", flush=True)
        return None

    # Parse response
    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    facts_match = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response)

    if not question_match:
        return None

    question = question_match.group(1).strip()

    # Strip fact ID annotations that the LLM sometimes copies from the prompt
    # e.g. "(C8_F1)", "[C3_F2]", "(A_C1_F1)", "(B_C2_F1)", "(W1)", "[W2]"
    question = re.sub(r'\s*[\(\[]\s*(?:[AB]?_?C\d+_F\d+|W\d+)\s*[\)\]]', '', question)

    used_facts = []
    if facts_match:
        used_facts = [f.strip() for f in facts_match.group(1).split(",") if f.strip()]

    return {"question": question, "used_facts": used_facts}


def compose_biochem_comparative_question(clues_a, clues_b, reasoning_ctx,
                                         entity_name_a, entity_name_b,
                                         prev_questions):
    """Compose a comparative biochemistry QA question with per-entity clue blocks.

    Args:
        clues_a: List of clue fact dicts for entity A.
        clues_b: List of clue fact dicts for entity B.
        reasoning_ctx: Template context from select_comparative_template().
        entity_name_a, entity_name_b: Entity names (for validation only).
        prev_questions: List of previously generated question strings.

    Returns:
        Dict with "question", "used_facts", "used_facts_a", "used_facts_b"
        keys, or None on failure.
    """
    facts_text_a = "\n".join(
        f"[A_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Entity A] {c['fact']}"
        for c in clues_a
    )
    facts_text_b = "\n".join(
        f"[B_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Entity B] {c['fact']}"
        for c in clues_b
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    prompt = BIOCHEM_COMPARATIVE_QA_PROMPT.format(
        facts_text_a=facts_text_a,
        facts_text_b=facts_text_b,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=BIOCHEM_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Comparative composition LLM error: {e}", flush=True)
        return None

    q_match = re.search(
        r'Question:\s*(.+?)(?:\nUsed_Facts|\nReasoning|\Z)', response, re.DOTALL
    )
    if not q_match:
        return None

    question = q_match.group(1).strip()
    question = re.sub(r'\s*[\(\[]\s*(?:[AB]_)?(?:C\d+_F\d+|W\d+|F\d+)\s*[\)\]]', '', question)

    used_facts_a = []
    used_facts_b = []
    m_a = re.search(r'Used_Facts_A:\s*(.+?)(?:\n|$)', response)
    m_b = re.search(r'Used_Facts_B:\s*(.+?)(?:\n|$)', response)
    if m_a:
        used_facts_a = [f.strip() for f in m_a.group(1).split(",") if f.strip()]
    if m_b:
        used_facts_b = [f.strip() for f in m_b.group(1).split(",") if f.strip()]

    all_used = used_facts_a + used_facts_b
    if not all_used:
        m_plain = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response)
        if m_plain:
            all_used = [f.strip() for f in m_plain.group(1).split(",") if f.strip()]
            used_facts_a = [f for f in all_used if f.startswith("A_")]
            used_facts_b = [f for f in all_used if f.startswith("B_")]

    return {
        "question": question,
        "used_facts": used_facts_a + used_facts_b,
        "used_facts_a": used_facts_a,
        "used_facts_b": used_facts_b,
    }


def gen_biochem_qa_from_entity(entity_data, agent_info, num_questions=5,
                                prev_questions=None, used_templates=None,
                                articles=None, category=None):
    """Generate biochem QA pairs from a single entity using chain-based clues.

    Parallels gen_financial_qa_from_entity with Phases 0/1/1.5/2/3.

    Args:
        entity_data: Dict with 'name', 'entity_id', 'entity_type', 'data',
                     'chains', 'articles', optionally 'wikidata_id'.
        agent_info: (lm, tokenizer, client) tuple.
        num_questions: Questions to generate per entity.
        prev_questions: Previously generated question strings.
        used_templates: Set of template IDs already used.
        articles: Wikipedia article dicts (overrides entity_data['articles']).
        category: Category key.

    Returns:
        List of QA pair dicts.
    """
    if prev_questions is None:
        prev_questions = []
    if used_templates is None:
        used_templates = set()

    entity_name = entity_data["name"]
    entity_type_val = entity_data.get("entity_type", "compound")
    data = entity_data.get("data", {})
    chains = entity_data.get("chains", [])
    wikidata_id = entity_data.get("wikidata_id", "")
    if articles is None:
        articles = entity_data.get("articles", [])

    if not chains or len(chains) < 2:
        print(f"  No KG chains for {entity_name}", flush=True)
        return []

    qa_pairs = []
    max_attempts = num_questions * 3
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{entity_name}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select template and compute gold answer
        reasoning_ctx = select_biochem_template(data, entity_type_val, used_templates)
        if reasoning_ctx is None:
            print(f"  No valid template remaining for {entity_name}", flush=True)
            single_templates = [t for t, d in BIOCHEM_TEMPLATES.items()
                                if d["type"] == "single" and d["entity_type"] == entity_type_val]
            if len(used_templates & set(single_templates)) >= len(single_templates):
                used_templates -= set(single_templates)
                continue
            break

        print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

        # Phase 1: Extract clue facts from KG chains
        extracted_facts, _ = extract_chain_clue_facts(chains, entity_name, agent_info)
        if extracted_facts is None:
            continue

        # Phase 1.5: Verify facts against Wikipedia grounding documents
        grounded_facts = verify_facts_grounding(extracted_facts, chains, articles, agent_info)
        if grounded_facts is None:
            continue
        extracted_facts = grounded_facts

        # Filter out poisoned facts from previously rejected QAs
        extracted_facts = filter_poisoned_facts(extracted_facts, poisoned_props)
        if len(extracted_facts) < 3:
            print(f"  Too few facts after poisoned-fact filtering", flush=True)
            continue

        # Phase 2: Compose question from grounded clues
        clues_for_composition = [
            {
                "fact_id": f.get("fact_id", f"C{f.get('chain_num', 0)}_F1"),
                "topic": f.get("property", "description"),
                "fact": f.get("fact", ""),
            }
            for f in extracted_facts
        ]
        composition = compose_biochem_question(
            clues_for_composition, reasoning_ctx, entity_name, prev_questions,
        )
        if composition is None:
            print(f"  Phase 2: Composition failed", flush=True)
            continue

        question = composition["question"]
        used_facts = composition["used_facts"]
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation
        bio_values = [
            data[k] for k in reasoning_ctx["template"]["required_data"]
            if k in data and isinstance(data[k], (int, float))
        ]
        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=reasoning_ctx["gold_answer"],
            computation_code=reasoning_ctx["code"],
            entity_label=entity_name,
            entity_names=_biochem_entity_names(entity_name, data),
            entity_values=bio_values,
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            entity_type=entity_type_val,
            check_facts=True,
            min_chains=2,
            chains=chains,
            entity_qid=wikidata_id,
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3: {reason}", flush=True)
            continue
        print(f"  Phase 3: PASSED", flush=True)

        # Build output fields
        source_triples = [c["path_description"] for c in chains[:5]]
        _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
        facts_for_output = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts
        ]

        grounding_articles = build_multiskill_grounding_articles(
            entity_name, reasoning_ctx["gold_answer"],
            reasoning_ctx["template"]["answer_unit"],
            extracted_facts, chains, articles or [],
        )

        _grounding_keys = {"molecular_weight", "molecular_formula", "xlogp",
                           "hbond_donor_count", "hbond_acceptor_count",
                           "rotatable_bond_count", "heavy_atom_count", "exact_mass",
                           "tpsa", "complexity", "charge", "mass", "length",
                           "resolution", "sequence_length", "organism",
                           "genome_size", "gc_content", "gene_count",
                           "protein_coding_genes", "chromosome_count",
                           "amino_acid_counts"}
        grounding_entity_data = {k: data[k] for k in _grounding_keys if k in data}

        qa_pair = {
            "question": question,
            "gold_answer": reasoning_ctx["gold_answer"],
            "computation_code": reasoning_ctx["code"],
            "template_id": reasoning_ctx["template_id"],
            "template_label": reasoning_ctx["template"]["label"],
            "template_type": reasoning_ctx["template"]["type"],
            "template_level": reasoning_ctx["template"]["level"],
            "answer_unit": reasoning_ctx["template"]["answer_unit"],
            "entity_name": entity_name,
            "entity_type": entity_type_val,
            "category": category,
            "used_facts": used_facts,
            "extracted_facts": facts_for_output,
            "source_triples": source_triples,
            "grounding_articles": grounding_articles,
            "wikidata_entity": wikidata_id,
            "data_source": "biochem_api",
            "grounding_clues": clues_for_composition,
            "grounding_entity_data": grounding_entity_data,
        }
        qa_pair.update(compute_cci_fields(reasoning_ctx["template"], extracted_facts, is_comparative=False))

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add(reasoning_ctx["template_id"])

    return qa_pairs


def gen_biochem_qa_single(entity_name, entity_data, clues, entity_type,
                           used_templates, prev_questions, category):
    """Generate a single-entity biochemistry QA pair (legacy clue-based fallback).

    Phases:
        0: Select template, compute gold answer from API data
        1: Format clue facts from Wikipedia
        2: Compose question via LLM
        3: Validate (no entity name leak, no data leak, recompute gold)

    Returns:
        QA pair dict, or None on failure.
    """
    # Phase 0: Select template
    reasoning_ctx = select_biochem_template(entity_data, entity_type, used_templates)
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1: Clue facts ready
    if len(clues) < 3:
        print(f"  Phase 1: Insufficient clues ({len(clues)})", flush=True)
        return None

    # Phase 2: Compose question
    composition = compose_biochem_question(
        clues, reasoning_ctx, entity_name, prev_questions,
    )
    if composition is None:
        print(f"  Phase 2: Composition failed", flush=True)
        return None

    question = composition["question"]
    used_facts = composition["used_facts"]
    print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

    # Phase 3: Validation (shared utility)
    bio_values = [
        entity_data[k] for k in reasoning_ctx["template"]["required_data"]
        if k in entity_data and isinstance(entity_data[k], (int, float))
    ]
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=entity_name,
        entity_names=_biochem_entity_names(entity_name, entity_data),
        entity_values=bio_values,
        used_facts=used_facts,
        extracted_facts=[],
        gen_fn=gen_from_prompt_harmony,
        entity_type="compound",
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"molecular_weight", "molecular_formula", "xlogp",
                       "hbond_donor_count", "hbond_acceptor_count",
                       "rotatable_bond_count", "heavy_atom_count", "exact_mass",
                       "tpsa", "complexity", "charge", "mass", "length",
                       "resolution", "sequence_length", "organism",
                       "genome_size", "gc_content", "gene_count",
                       "protein_coding_genes", "chromosome_count",
                       "amino_acid_counts"}
    grounding_entity_data = {k: entity_data[k] for k in _grounding_keys if k in entity_data}

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": reasoning_ctx["template"]["type"],
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "entity_name": entity_name,
        "entity_type": entity_type,
        "category": category,
        "used_facts": used_facts,
        "data_source": "biochem_api",
        "grounding_clues": clues,
        "grounding_entity_data": grounding_entity_data,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=False))
    return result


def gen_biochem_qa_comparative(name_a, data_a, clues_a,
                                 name_b, data_b, clues_b,
                                 entity_type, used_templates,
                                 prev_questions, category,
                                 chains_a=None, articles_a=None,
                                 chains_b=None, articles_b=None,
                                 agent_info=None):
    """Generate a comparative biochemistry QA pair (two entities)."""
    # Phase 0: Select comparative template
    reasoning_ctx = select_comparative_template(data_a, data_b, entity_type, used_templates)
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1 + 1.5: Extract and ground chain-based clues
    facts_a = None
    facts_b = None
    if chains_a and chains_b:
        facts_a, _ = extract_chain_clue_facts(chains_a, name_a, agent_info)
        if facts_a is not None and articles_a:
            facts_a = verify_facts_grounding(facts_a, chains_a, articles_a, agent_info)
        facts_b, _ = extract_chain_clue_facts(chains_b, name_b, agent_info)
        if facts_b is not None and articles_b:
            facts_b = verify_facts_grounding(facts_b, chains_b, articles_b, agent_info)

    # Build separate per-entity clue lists — prefer chain-based facts, fall back to legacy clues
    src_a = facts_a if facts_a else clues_a
    src_b = facts_b if facts_b else clues_b

    # Filter cross-entity contamination
    names_a = _biochem_entity_names(name_a, data_a)
    names_b = _biochem_entity_names(name_b, data_b)
    src_a = filter_cross_entity_facts(src_a, names_b)
    src_b = filter_cross_entity_facts(src_b, names_a)

    clues_a_list = src_a[:5]
    clues_b_list = src_b[:5]

    if len(clues_a_list) < 2 or len(clues_b_list) < 2:
        print(f"  Insufficient per-side clues after cross-entity filter "
              f"(A={len(clues_a_list)}, B={len(clues_b_list)}, need 2+ each)", flush=True)
        return None

    # Phase 2: Compose question using comparative-specific composer
    composition = compose_biochem_comparative_question(
        clues_a_list, clues_b_list, reasoning_ctx, name_a, name_b, prev_questions,
    )
    if composition is None:
        return None

    question = composition["question"]

    # Validate per-side used_facts count
    a_used = composition.get("used_facts_a", [])
    b_used = composition.get("used_facts_b", [])
    if len(a_used) < 2 or len(b_used) < 2:
        print(f"  Phase 2: Insufficient per-side used facts "
              f"(A={len(a_used)}, B={len(b_used)}, need 2+ each)", flush=True)
        return None

    # Phase 3: Validation (shared utility)
    all_values = []
    for data_src in [data_a, data_b]:
        for k in reasoning_ctx["template"]["required_data"]:
            val = data_src.get(k)
            if val is not None and isinstance(val, (int, float)):
                all_values.append(val)
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=f"{name_a} / {name_b}",
        entity_names=_biochem_entity_names(name_a, data_a) + _biochem_entity_names(name_b, data_b),
        entity_values=all_values,
        entity_type="compound",
    )
    if passed:
        # 3h (comparative): each side must be uniquely identifiable from its clues.
        # biochem used_facts_a/_b are base ids (no A_/B_ prefix) matching clues_*_list.
        passed, reason = check_kg_uniqueness_comparative(
            a_used, clues_a_list, chains_a or [],
            b_used, clues_b_list, chains_b or [],
        )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Build source triples from chains if available
    source_triples_a = [c["path_description"] for c in (chains_a or [])[:3]]
    source_triples_b = [c["path_description"] for c in (chains_b or [])[:3]]

    # Build grounding articles for both entities
    grounding_articles_a = build_multiskill_grounding_articles(
        name_a, reasoning_ctx["gold_answer"],
        reasoning_ctx["template"]["answer_unit"],
        facts_a or [], chains_a or [], articles_a or [],
    ) if chains_a else []
    grounding_articles_b = build_multiskill_grounding_articles(
        name_b, reasoning_ctx["gold_answer"],
        reasoning_ctx["template"]["answer_unit"],
        facts_b or [], chains_b or [], articles_b or [],
    ) if chains_b else []

    # Extracted facts for output
    _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
    facts_out_a = [
        {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
        for f in (facts_a or [])
    ]
    facts_out_b = [
        {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
        for f in (facts_b or [])
    ]

    # Store grounding data for verification
    _grounding_keys = {"molecular_weight", "molecular_formula", "xlogp",
                       "hbond_donor_count", "hbond_acceptor_count",
                       "rotatable_bond_count", "heavy_atom_count", "exact_mass",
                       "tpsa", "complexity", "charge", "mass", "length",
                       "resolution", "sequence_length", "organism",
                       "genome_size", "gc_content", "gene_count",
                       "protein_coding_genes", "chromosome_count",
                       "amino_acid_counts"}
    grounding_entity_data_a = {k: data_a[k] for k in _grounding_keys if k in data_a}
    grounding_entity_data_b = {k: data_b[k] for k in _grounding_keys if k in data_b}

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": "comparative",
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "entity_name_a": name_a,
        "entity_name_b": name_b,
        "entity_type": entity_type,
        "category": category,
        "used_facts": composition.get("used_facts", []),
        "extracted_facts": facts_out_a + facts_out_b,
        "source_triples": source_triples_a + source_triples_b,
        "grounding_articles": grounding_articles_a + grounding_articles_b,
        "data_source": "biochem_api",
        "grounding_clues_a": clues_a,
        "grounding_clues_b": clues_b,
        "grounding_entity_data_a": grounding_entity_data_a,
        "grounding_entity_data_b": grounding_entity_data_b,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=True))
    return result


# ===========================================================================
# V2 Verification: Bio + Browser + Python Tools
# ===========================================================================

def verify_biochem_v2(qa_pairs, num_samples=10, temperature=1.0,
                       max_iterations=200, threshold=0.5,
                       outfile_prefix=None, subarea="",
                       bench_label="biochem_bench"):
    """V2 verification using Bio + Browser + Python tools.

    For each QA pair, runs an agentic loop where the model can:
    - Use the Bio tool to fetch protein/compound/structure data
    - Use the browser tool to search Wikipedia to identify entities
    - Use the Python tool to execute code for computations

    Args:
        qa_pairs: List of V1-filtered QA dicts.
        num_samples: Agentic samples per question.
        temperature: Sampling temperature.
        max_iterations: Max agentic loop iterations per sample.
        threshold: Keep questions with v2 accuracy < threshold.
        outfile_prefix: Cache prefix for output files.
        subarea: Sub-area name for logging.
        bench_label: Label for cache files.

    Returns:
        (filtered_pairs, all_pairs) — filtered has v2_accuracy < threshold.
    """
    all_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2.json" if outfile_prefix else None
    filtered_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2_filtered.json" if outfile_prefix else None

    # Check cache
    if all_cache and os.path.exists(all_cache) and filtered_cache and os.path.exists(filtered_cache):
        print(f"Found cached V2 results: {all_cache}", flush=True)
        with open(all_cache, "r") as f:
            all_pairs = json.load(f)
        with open(filtered_cache, "r") as f:
            filtered_pairs = json.load(f)
        return filtered_pairs, all_pairs

    generator = get_harmony_generator()
    if generator is None:
        raise RuntimeError("Harmony generator not initialized for V2 verification")
    if not _HAS_BROWSER:
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for biochem V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for biochem V2")
    if not _HAS_BIO_TOOL:
        raise RuntimeError("BiochemTool not available for biochem V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== BIOCHEM V2 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Tools: Bio + Browser + Python", flush=True)
    print(f"Verifying {len(qa_pairs)} QA pairs with {num_samples} samples each...", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc=f"V2 {subarea}")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        q_text = qa['question'].strip()
        if not q_text or q_text in ('**', '*'):
            qa_annotated = copy.deepcopy(qa)
            qa_annotated['v2_accuracy'] = 0.0
            qa_annotated['v2_correct_count'] = 0
            qa_annotated['v2_num_samples'] = num_samples
            qa_annotated['v2_sampled_answers'] = []
            qa_annotated['v2_tool_call_count'] = []
            qa_annotated['v2_iteration_count'] = []
            all_pairs.append(qa_annotated)
            continue

        sampled_answers = []
        correct_count = 0
        per_sample_tool_calls = []
        per_sample_iterations = []
        per_sample_errors = []

        def _run_sample(i):
            """One agentic V2 sample; self-contained + exception-safe so samples
            can run concurrently. Returns an ordered per-sample record."""
            backend = None
            python_tool = None
            bio_tool = None
            qid_str = f"{bench_label}_v2_{subarea}_{idx}_{i}"
            try:
                # Initialize tools
                backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
                browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
                python_tool = HybridPythonTool(timeout=60)
                python_tool.set_qid(qid_str)
                bio_tool = BiochemTool()

                system_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.HIGH)
                    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                    .with_tools(browser_tool.tool_config)
                    .with_tools(python_tool.tool_config)
                    .with_tools(bio_tool.tool_config)
                )

                messages = [
                    Message.from_role_and_content(Role.SYSTEM, system_content),
                    Message.from_role_and_content(Role.DEVELOPER, BIOCHEM_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

                tool_call_counter = [0]

                async def _tool_handler(msg, _browser=browser_tool, _python=python_tool,
                                        _bio=bio_tool, _counter=tool_call_counter):
                    _counter[0] += 1
                    recipient = str(getattr(msg, 'recipient', ''))
                    results = []
                    if recipient.startswith("bio"):
                        async for m in _bio.process(msg):
                            results.append(m)
                    elif recipient.startswith("browser."):
                        async for m in _browser.process(msg):
                            results.append(m)
                    elif recipient.startswith("python"):
                        async for m in _python.process(msg):
                            results.append(m)
                    else:
                        error = Message.from_role_and_content(
                            Role.SYSTEM, f"Unknown tool: {recipient}"
                        )
                        results.append(error)
                    return results

                result_messages = generator.generate_agentic_response_sync(
                    messages,
                    _tool_handler,
                    tool_prefix=("bio", "browser.", "python"),
                    tool_configs=[browser_tool.tool_config, python_tool.tool_config, bio_tool.tool_config],
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)

                answer = _extract_biochem_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())

                correct = bool(is_biochem_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_biochem_answer(answer, qa['gold_answer'], qa['question']))

                log = (f"    V2 sample {i+1}/{num_samples}: '{answer}' "
                       f"(gold: '{qa['gold_answer']}') tools={tool_call_counter[0]} "
                       f"iters={iteration_count}")
                return {"answer": answer, "tool_calls": tool_call_counter[0],
                        "iterations": iteration_count, "error": None,
                        "correct": correct, "log": log}

            except Exception as e:
                import traceback
                error_str = f"{type(e).__name__}: {e}"
                tb_str = traceback.format_exc()
                log = (f"    V2 sample {i+1}/{num_samples} error: {error_str}\n{tb_str}")
                return {"answer": "", "tool_calls": 0, "iterations": 0,
                        "error": error_str, "correct": False, "log": log}

            finally:
                if backend is not None:
                    try:
                        import asyncio
                        asyncio.run_coroutine_threadsafe(
                            backend.close(), generator._loop
                        ).result(timeout=5)
                    except Exception:
                        pass
                if python_tool is not None:
                    try:
                        python_tool._terminate_worker(qid_str)
                    except Exception:
                        pass

        conc = _sample_concurrency(num_samples, "DRBENCH_V2_CONCURRENCY", default=4)
        records = run_samples_concurrently(_run_sample, num_samples, conc)
        for r in records:
            sampled_answers.append(r["answer"])
            per_sample_tool_calls.append(r["tool_calls"])
            per_sample_iterations.append(r["iterations"])
            per_sample_errors.append(r["error"])
            if r["correct"]:
                correct_count += 1
            print(r["log"], flush=True)

        accuracy = correct_count / num_samples if num_samples > 0 else 0.0
        import numpy as np
        avg_tool_calls = float(np.mean(per_sample_tool_calls)) if per_sample_tool_calls else 0.0
        avg_iterations = float(np.mean(per_sample_iterations)) if per_sample_iterations else 0.0

        qa_annotated = copy.deepcopy(qa)
        qa_annotated['v2_accuracy'] = accuracy
        qa_annotated['v2_correct_count'] = correct_count
        qa_annotated['v2_num_samples'] = num_samples
        qa_annotated['v2_sampled_answers'] = sampled_answers
        qa_annotated['v2_tool_call_count'] = per_sample_tool_calls
        qa_annotated['v2_iteration_count'] = per_sample_iterations
        qa_annotated['v2_avg_tool_calls'] = avg_tool_calls
        qa_annotated['v2_avg_iterations'] = avg_iterations
        errors_only = [e for e in per_sample_errors if e is not None]
        if errors_only:
            qa_annotated['v2_errors'] = errors_only
        all_pairs.append(qa_annotated)

        if accuracy < threshold:
            filtered_pairs.append(qa_annotated)
            print(f"  [{idx+1}] KEEP - V2 Accuracy: {accuracy:.1%} "
                  f"avg_tools={avg_tool_calls:.1f} avg_iters={avg_iterations:.1f} "
                  f"- Q: {q_text[:60]}...", flush=True)
        else:
            print(f"  [{idx+1}] SOLVED - V2 Accuracy: {accuracy:.1%} "
                  f"avg_tools={avg_tool_calls:.1f} avg_iters={avg_iterations:.1f} "
                  f"- Q: {q_text[:60]}...", flush=True)

    print(f"\nV2 complete: {len(filtered_pairs)}/{len(all_pairs)} passed filter "
          f"(accuracy < {threshold:.0%})", flush=True)

    # Save caches
    if all_cache:
        output_dir = os.path.dirname(all_cache)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(all_cache, "w") as f:
            json.dump(all_pairs, f, indent=2)
    if filtered_cache:
        with open(filtered_cache, "w") as f:
            json.dump(filtered_pairs, f, indent=2)

    return filtered_pairs, all_pairs


def _extract_biochem_answer(messages):
    """Extract the final answer from an agentic response."""
    from .multiskill_utils import unwrap_content
    for msg in reversed(messages):
        if hasattr(msg, 'author') and msg.author.role == Role.ASSISTANT:
            content = unwrap_content(getattr(msg, 'content', ''))
            if not content.strip():
                continue
            lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
            if lines:
                last = lines[-1]
                for prefix in ["Answer:", "Final answer:", "The answer is", "Result:",
                               "Final Answer:"]:
                    if last.lower().startswith(prefix.lower()):
                        last = last[len(prefix):].strip()
                return last
    return ""


# ===========================================================================
# Orchestrator
# ===========================================================================

def _generate_category_qa_pairs(category, per_category, agent_info,
                                 prefix, bench_label,
                                 num_chains=20, questions_per_entity=5):
    """Generate QA pairs for a single entity category.

    Uses the unified chains pipeline:
    1. Fetch entities → quantitative data + resolve Wikidata QID + KG chains + Wikipedia articles
    2. Cache as {prefix}__{category}.biochem_chains.json
    3. Generate QA pairs per entity via chain-based Phase 1/1.5/2/3
    """
    # Determine entity type from category
    CATEGORY_TO_ENTITY_TYPE = {
        "proteins": "protein",
        "enzymes": "protein",       # UniProt IDs
        "receptors": "protein",     # UniProt IDs
        "compounds": "compound",
        "antibiotics": "compound",  # PubChem CIDs
        "vitamins": "compound",     # PubChem CIDs
        "hormones": "compound",     # PubChem CIDs
        "lipids": "compound",       # PubChem CIDs
        "drugs": "compound",        # PubChem CIDs
        "organisms": "organism",
        # ---- NEW TOPICS (2026-03-16) ----
        "neurotransmitters": "compound",
        "analgesics": "compound",
        "antineoplastics": "compound",
        "antidepressants": "compound",
        "antivirals": "compound",
        "amino_acids": "compound",
        "sugars": "compound",
        "nucleotides": "compound",
        "antifungals": "compound",
        "toxins": "compound",
        "metabolites": "compound",
        "steroids": "compound",
        "kinases": "protein",
        "proteases": "protein",
        "transporters": "protein",
        "cytokines": "protein",
        "structural_proteins": "protein",
        "transcription_factors": "protein",
        "viruses": "organism",
        "parasites": "organism",
    }
    entity_type = CATEGORY_TO_ENTITY_TYPE.get(category, "compound")

    # Check chains cache (new pattern, replaces biochem_entity_data.json)
    chains_cache = f"{prefix}__{category}.biochem_chains.json"
    old_data_cache = f"{prefix}__{category}.biochem_entity_data.json"

    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {category}...", flush=True)
        with open(chains_cache) as f:
            entity_data_list = json.load(f)

        # Backfill Wikipedia articles for cached entities that lack them
        needs_resave = False
        for ed in entity_data_list:
            if "articles" not in ed or not ed["articles"]:
                print(f"  Fetching Wikipedia articles for cached entity {ed.get('name', '')}...",
                      flush=True)
                ed["articles"] = fetch_wikipedia_for_entities(ed.get("chains", []))
                print(f"  {ed.get('name', '')}: fetched {len(ed['articles'])} Wikipedia articles",
                      flush=True)
                needs_resave = True
        # Append new entities from ENTITY_UNIVERSE that aren't in the cache yet
        cached_names = {ed.get("name", "") for ed in entity_data_list}
        cached_ids = {ed.get("entity_id", "") for ed in entity_data_list}
        universe_entities = get_category_entities(category)
        new_entities = [
            e for e in universe_entities
            if e.get("name", "") not in cached_names and e.get("id", "") not in cached_ids
        ]
        if new_entities:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Found {len(new_entities)} new entities in ENTITY_UNIVERSE "
                  f"not yet cached for {category}, fetching...", flush=True)
            for entity in tqdm.tqdm(new_entities, desc=f"Appending new {category} entities"):
                time.sleep(2)
                name = entity.get("name", "")
                eid = entity.get("id", "")

                # Fetch quantitative data based on entity type
                if entity_type == "protein":
                    data = get_protein_data(eid)
                elif entity_type == "compound":
                    data = get_compound_data(name)
                elif entity_type == "organism":
                    taxid = entity.get("taxid")
                    data = get_organism_data(taxid) if taxid else None
                else:
                    data = None

                if data is None:
                    print(f"  {name}: no data from API, skipping", flush=True)
                    continue
                if entity_type in ("protein", "compound") and not data.get("molecular_weight"):
                    continue
                if entity_type == "organism" and not data.get("genome_size"):
                    continue

                wikidata_id = resolve_biochem_wikidata_id(name, entity_type, entity_id=eid)
                chains = []
                articles = []
                if wikidata_id:
                    chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                                    min_sitelinks=2)
                    if chains and len(chains) >= 2:
                        articles = fetch_wikipedia_for_entities(chains)
                    else:
                        chains = chains or []

                clues = fetch_entity_clues(name, entity_type)
                entity_data_list.append({
                    "name": name,
                    "entity_id": eid,
                    "entity_type": entity_type,
                    "wikidata_id": wikidata_id or "",
                    "data": data,
                    "clues": clues,
                    "chains": chains,
                    "articles": articles,
                    "cached_at": now_str,
                })
                print(f"  {name}: appended ({len(chains)} chains, {len(articles)} articles)",
                      flush=True)
            needs_resave = True

        if needs_resave:
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Re-saved chains cache for {category} ({len(entity_data_list)} entities)",
                  flush=True)

    elif os.path.exists(old_data_cache):
        # Backfill: upgrade old entity_data cache to chains cache
        print(f"Upgrading old entity data cache for {category} to chains format...", flush=True)
        with open(old_data_cache) as f:
            old_list = json.load(f)

        entity_data_list = []
        for ed in tqdm.tqdm(old_list, desc=f"Upgrading {category} cache"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            name = ed.get("name", "")

            # Resolve Wikidata QID
            wikidata_id = resolve_biochem_wikidata_id(name, entity_type, entity_id=ed.get("entity_id", ""))
            if wikidata_id is None:
                print(f"  {name}: no Wikidata QID found, keeping with empty chains", flush=True)
                ed["wikidata_id"] = ""
                ed["chains"] = []
                ed["articles"] = []
                entity_data_list.append(ed)
                continue

            chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                            min_sitelinks=2)
            if not chains or len(chains) < 2:
                ed["wikidata_id"] = wikidata_id
                ed["chains"] = chains or []
                ed["articles"] = []
                entity_data_list.append(ed)
                continue

            articles = fetch_wikipedia_for_entities(chains)
            print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                  f"for {len(chains)} chains", flush=True)

            ed["wikidata_id"] = wikidata_id
            ed["chains"] = chains
            ed["articles"] = articles
            entity_data_list.append(ed)

        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Upgraded and cached {len(entity_data_list)} entities for {category}", flush=True)

    else:
        entities = get_category_entities(category)
        if not entities:
            print(f"No entities for category {category}", flush=True)
            return []

        entity_data_list = []
        for entity in tqdm.tqdm(entities, desc=f"Fetching {category} data"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            name = entity.get("name", "")
            eid = entity.get("id", "")

            # Fetch quantitative data based on entity type
            if entity_type == "protein":
                data = get_protein_data(eid)
            elif entity_type == "compound":
                data = get_compound_data(name)
            elif entity_type == "organism":
                taxid = entity.get("taxid")
                if taxid:
                    data = get_organism_data(taxid)
                else:
                    data = None
            else:
                data = None

            if data is None:
                continue

            # Check minimum data availability (entity-type-aware)
            if entity_type in ("protein", "compound"):
                if not data.get("molecular_weight"):
                    continue
            elif entity_type == "organism":
                if not data.get("genome_size"):
                    continue

            # Resolve Wikidata QID
            wikidata_id = resolve_biochem_wikidata_id(name, entity_type, entity_id=eid)
            if wikidata_id is None:
                print(f"  {name}: no Wikidata QID found, skipping", flush=True)
                continue

            # Fetch multi-hop KG chains
            chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                            min_sitelinks=2)
            if not chains or len(chains) < 2:
                print(f"  {name}: insufficient KG chains ({len(chains) if chains else 0}), skipping",
                      flush=True)
                continue

            # Fetch Wikipedia articles for chain entities
            articles = fetch_wikipedia_for_entities(chains)
            print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                  f"for {len(chains)} chains", flush=True)

            # Also fetch legacy clues as fallback
            clues = fetch_entity_clues(name, entity_type)

            entity_data_list.append({
                "name": name,
                "entity_id": eid,
                "entity_type": entity_type,
                "wikidata_id": wikidata_id,
                "data": data,
                "clues": clues,
                "chains": chains,
                "articles": articles,
            })

            if len(entity_data_list) >= per_category * 2:
                break

        # Save chains cache
        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Cached {len(entity_data_list)} entity data for {category}", flush=True)

    print(f"Category {category}: {len(entity_data_list)} entities available", flush=True)

    # Generate QA pairs
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    # Single-entity questions (per-entity loop)
    target_single = per_category * 2 // 3
    for ed in entity_data_list:
        if len([q for q in all_qa_pairs if q.get("template_type") == "single"]) >= target_single:
            break

        chains = ed.get("chains", [])
        articles = ed.get("articles", [])
        clues = ed.get("clues", [])

        # Use chain-based pipeline if chains available, else legacy clues
        if chains and len(chains) >= 2:
            entity_qas = gen_biochem_qa_from_entity(
                ed, agent_info,
                num_questions=questions_per_entity,
                prev_questions=prev_questions,
                used_templates=used_templates,
                articles=articles,
                category=category,
            )
            for qa in entity_qas:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
            if entity_qas:
                print(f"  {ed['name']}: generated {len(entity_qas)} chain-based QA pairs "
                      f"(total: {len(all_qa_pairs)})", flush=True)
        else:
            # Legacy fallback
            qa = gen_biochem_qa_single(
                ed["name"], ed.get("data", {}), clues, ed.get("entity_type", entity_type),
                used_templates, prev_questions, category,
            )
            if qa:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
                print(f"  {ed['name']}: single QA generated (legacy) "
                      f"(total: {len(all_qa_pairs)})", flush=True)

        # Reset used_templates after cycling
        single_templates = [t for t, d in BIOCHEM_TEMPLATES.items()
                           if d["type"] == "single" and d["entity_type"] == ed.get("entity_type", entity_type)]
        if len(used_templates & set(single_templates)) >= len(single_templates):
            used_templates -= set(single_templates)

    # Comparative questions
    target_comp = per_category // 3
    comp_used = set()
    for i in range(0, len(entity_data_list) - 1, 2):
        if len([q for q in all_qa_pairs if q.get("template_type") == "comparative"]) >= target_comp:
            break

        ed_a = entity_data_list[i]
        ed_b = entity_data_list[i + 1]

        # Both must be same entity type for comparison
        if ed_a.get("entity_type", entity_type) != ed_b.get("entity_type", entity_type):
            continue

        # Retry a rejected pair a few times: comparative composition is
        # one-shot per call, so a single stochastic Phase-3 rejection would
        # otherwise discard a viable pair (single-entity gen gets
        # num_questions*3 attempts; give each comparative pair a few too).
        qa = None
        for _ in range(3):
            qa = gen_biochem_qa_comparative(
                ed_a["name"], ed_a.get("data", {}), ed_a.get("clues", []),
                ed_b["name"], ed_b.get("data", {}), ed_b.get("clues", []),
                ed_a.get("entity_type", entity_type), comp_used, prev_questions, category,
                chains_a=ed_a.get("chains", []), articles_a=ed_a.get("articles", []),
                chains_b=ed_b.get("chains", []), articles_b=ed_b.get("articles", []),
                agent_info=agent_info,
            )
            if qa:
                break
        if qa:
            all_qa_pairs.append(qa)
            prev_questions.append(qa["question"])
            print(f"  {ed_a['name']} vs {ed_b['name']}: comparative QA generated "
                  f"(total: {len(all_qa_pairs)})", flush=True)

    return all_qa_pairs


def run_biochem_bench(args, agent_info):
    """Run the biochemistry benchmark pipeline.

    Per category:
    1. Fetch entities + quantitative data + Wikipedia clues
    2. Generate QA pairs by template type
    3. V1 verification (closed-book)
    4. V2 verification (Bio + Browser + Python tools)
    5. Merge all categories into final output

    Args:
        args: Parsed CLI arguments
        agent_info: (lm, tokenizer, client) tuple
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    categories_arg = args.biochem_categories
    per_category = args.biochem_per_category
    v1_samples = args.biochem_v1_samples
    v2_samples = args.biochem_v2_samples
    v1_threshold = args.biochem_v1_threshold
    v2_threshold = args.biochem_v2_threshold
    num_chains = getattr(args, "biochem_num_chains", 20)
    questions_per_entity = getattr(args, "biochem_questions_per_entity", 5)

    bench_label = "biochem_bench"

    # Parse categories
    if categories_arg:
        categories_to_process = [s.strip() for s in categories_arg.split(",")]
    else:
        categories_to_process = list(ENTITY_UNIVERSE.keys())

    is_subset = categories_arg is not None and len(categories_to_process) < len(ENTITY_UNIVERSE)

    print(f"=== BIOCHEMISTRY BENCH (run_id={run_id}) ===", flush=True)
    print(f"Categories: {categories_to_process}", flush=True)
    print(f"Target QAs per category: {per_category}", flush=True)
    print(f"V1: {v1_samples} samples, threshold < {v1_threshold:.0%}", flush=True)
    print(f"V2: {v2_samples} samples, threshold < {v2_threshold:.0%}", flush=True)
    print("=" * 50, flush=True)

    all_final = []
    summary = {}

    for category in categories_to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Category: {category.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Ensure output directory exists
        output_dir = os.path.dirname(run_prefix) if "/" in run_prefix else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check full cache (V2)
        v2_cache = f"{run_prefix}__{category}.{bench_label}_v2.json"
        v2_filtered_cache = f"{run_prefix}__{category}.{bench_label}_v2_filtered.json"
        if os.path.exists(v2_cache) and os.path.exists(v2_filtered_cache):
            print(f"Found cached V2 results for {category}, loading...", flush=True)
            with open(v2_cache) as f:
                v2_all = json.load(f)
            with open(v2_filtered_cache) as f:
                v2_filtered = json.load(f)
            all_final.extend(v2_filtered)
            summary[category] = {
                "generated": "cached", "v1_filtered": "cached",
                "v2_total": len(v2_all), "v2_filtered": len(v2_filtered),
            }
            continue
        else:
            # Check V1 filtered cache
            v1_filtered_cache = f"{run_prefix}__{category}.{bench_label}_v1_filtered.json"
            if os.path.exists(v1_filtered_cache):
                print(f"Found cached V1 filtered for {category}, skipping to V2...", flush=True)
                with open(v1_filtered_cache) as f:
                    v1_filtered = json.load(f)
            else:
                # Check raw bench cache
                bench_cache = f"{run_prefix}__{category}.{bench_label}.json"
                if os.path.exists(bench_cache):
                    print(f"Found cached bench problems for {category}...", flush=True)
                    with open(bench_cache) as f:
                        qa_pairs = json.load(f)
                else:
                    # Step 1: Generate QA pairs
                    qa_pairs = _generate_category_qa_pairs(
                        category, per_category, agent_info, prefix, bench_label,
                        num_chains=num_chains, questions_per_entity=questions_per_entity,
                    )

                    # Save raw bench cache
                    if qa_pairs:
                        with open(bench_cache, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"Saved {len(qa_pairs)} raw QA pairs to {bench_cache}", flush=True)

                if not qa_pairs:
                    print(f"No QA pairs for category {category}", flush=True)
                    summary[category] = {"generated": 0, "v1_filtered": 0,
                                         "v2_total": 0, "v2_filtered": 0}
                    continue

                # Step 2: V1 verification
                v1_filtered, v1_all = verify_bench_v1(
                    qa_pairs, agent_info,
                    num_samples=v1_samples,
                    temperature=0.7,
                    threshold=v1_threshold,
                    outfile_prefix=run_prefix,
                    subarea=category,
                    bench_label=bench_label,
                    answer_checker=is_biochem_answer_correct,
                    llm_judge=_llm_judge_biochem_answer,
                    verification_prompt=BIOCHEM_V1_PROMPT,
                )

            if not v1_filtered:
                print(f"No V1-filtered QA pairs for {category}", flush=True)
                summary[category] = {"generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                                     "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                continue

            # Step 3: V2 verification (Bio + Browser + Python tools)
            v2_filtered, v2_all = verify_biochem_v2(
                v1_filtered,
                num_samples=v2_samples,
                temperature=1.0,
                max_iterations=200,
                threshold=v2_threshold,
                outfile_prefix=run_prefix,
                subarea=category,
                bench_label=bench_label,
            )

        all_final.extend(v2_filtered)
        summary[category] = {
            "generated": len(v1_all) if 'v1_all' in dir() else "cached",
            "v1_filtered": len(v1_filtered) if 'v1_filtered' in dir() else "cached",
            "v2_total": len(v2_all),
            "v2_filtered": len(v2_filtered),
        }

    # Skip final merge when processing a subset (worker job)
    if is_subset:
        print(f"\nSubset mode: processed {categories_to_process}. "
              f"Merge will happen in the merge job.", flush=True)
        _print_summary(summary)
        return all_final

    # Final merge
    final_path = f"{run_prefix}.{bench_label}_final.json"
    if all_final:
        # Optional diversity filter
        if _HAS_DIVERSITY and len(all_final) > 10:
            print(f"\nApplying diversity filter to {len(all_final)} questions...", flush=True)
            all_final = diversity_filter(all_final, target_count=len(all_final), min_distance=0.03)

        with open(final_path, "w") as f:
            json.dump(all_final, f, indent=2)
        print(f"\nSaved {len(all_final)} final QA pairs to {final_path}", flush=True)

        # Diversity report
        if _HAS_DIVERSITY:
            diversity_report(all_final, text_keys=["question"])

    _print_summary(summary)
    return all_final


def _print_summary(summary):
    """Print pipeline summary table."""
    print(f"\n{'='*60}", flush=True)
    print("BIOCHEM BENCH SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for category, stats in summary.items():
        print(f"  {category:20s}: generated={stats.get('generated','?'):>6} "
              f"v1_kept={stats.get('v1_filtered',0):>4} "
              f"v2_kept={stats.get('v2_filtered',0):>4}", flush=True)
    total = sum(s.get("v2_filtered", 0) for s in summary.values())
    print(f"  {'TOTAL':20s}: {total:>4} final questions", flush=True)
    print(f"{'='*60}\n", flush=True)


# ===========================================================================
# CLI __main__
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="biochem_drbencher",
        description="Biochemistry Benchmark: PubChem + UniProt + Wikipedia entity ID",
    )

    # Model args
    parser.add_argument("--agent_modelname", default="openai/gpt-oss-120b")
    parser.add_argument("--use_harmony", type=str, default="no")
    parser.add_argument("--harmony_model_path", type=str, default=None)
    parser.add_argument("--use_vllm_serve", type=str, default="no",
                        help="Use external vllm serve endpoint (yes/no)")
    parser.add_argument("--vllm_serve_url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vllm serve endpoint")
    parser.add_argument("--vllm_serve_model", type=str, default=None,
                        help="Model name for vllm serve (required when use_vllm_serve=yes)")
    parser.add_argument("--use_helm", type=str, default="yes")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Output
    parser.add_argument("--outfile_prefix1", type=str, default="output/biochem/bench")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # Exp mode
    parser.add_argument("--exp_mode", type=str, default="biochem_bench")

    # Biochem-specific
    parser.add_argument("--biochem_categories", type=str, default=None,
                        help="Comma-separated categories (default: all). E.g. 'proteins,compounds'")
    parser.add_argument("--biochem_per_category", type=int, default=50,
                        help="Target QA pairs per category")
    parser.add_argument("--biochem_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--biochem_v2_samples", type=int, default=10,
                        help="V2 (Bio+Browser+Python) sampling attempts per question")
    parser.add_argument("--biochem_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling -- keep below this")
    parser.add_argument("--biochem_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling -- keep below this")
    parser.add_argument("--biochem_num_chains", type=int, default=20,
                        help="Number of KG chains to fetch per entity")
    parser.add_argument("--biochem_questions_per_entity", type=int, default=5,
                        help="Questions to generate per entity (chain-based pipeline)")

    args = parser.parse_args()

    # Initialize Harmony generator
    if args.use_harmony.lower() == "yes":
        import torch
        model_path = args.harmony_model_path or args.agent_modelname
        tp_size = args.tensor_parallel_size or torch.cuda.device_count()
        harmony_gen = HarmonyVLLMGenerator(
            model_path=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        set_harmony_generator(harmony_gen)
    elif args.use_vllm_serve.lower() == "yes":
        from .harmony_serve import HarmonyServeGenerator
        if not args.vllm_serve_model:
            raise ValueError("--vllm_serve_model is required when --use_vllm_serve=yes")
        gen = HarmonyServeGenerator(
            base_url=args.vllm_serve_url,
            model_name=args.vllm_serve_model,
        )
        set_harmony_generator(gen)

    # When Harmony or vllm_serve is enabled, skip loading separate models
    if args.use_harmony.lower() == "yes" or args.use_vllm_serve.lower() == "yes":
        print("Generator mode: using generator for all inference", flush=True)
        agent_info = (None, None, None)
    elif args.use_helm == "yes":
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)
    else:
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)

    # Run benchmark
    if args.exp_mode == "biochem_bench":
        run_biochem_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
