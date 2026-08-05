# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Bio/Chem API utilities for biochemistry benchmark.

Provides functions to fetch protein, compound, and organism data from
PubChem, UniProt, RCSB PDB, and ChEMBL APIs, with local JSON caching
and rate limiting.

API docs:
  PubChem PUG REST: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
  UniProt REST:     https://rest.uniprot.org/
  RCSB PDB:         https://data.rcsb.org/
  ChEMBL:           https://www.ebi.ac.uk/chembl/api/data/
"""

import json
import os
import random
import re
import time
import requests
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIO_CACHE_DIR = "cache/bio_cache"
_RATE_LIMIT_DELAY = 0.2  # 5 req/sec for PubChem
_last_request_time = 0.0

_HEADERS = {
    "User-Agent": "DrBencher/1.0 (research@example.com)",
    "Accept": "application/json",
}

_biochem_wikidata_cache: Dict[str, Optional[str]] = {}

# ---------------------------------------------------------------------------
# Entity Universe
# ---------------------------------------------------------------------------

ENTITY_UNIVERSE: Dict[str, List[Dict[str, Any]]] = {
    "neurotransmitters": [
        {"id": "CID:119", "name": "GABA", "synonyms": ["gamma-aminobutyric acid"]},
        {"id": "CID:187", "name": "Acetylcholine", "synonyms": []},
        {"id": "CID:774", "name": "Histamine", "synonyms": []},
        {"id": "CID:60961", "name": "Adenosine", "synonyms": []},
        {"id": "CID:5281969", "name": "Anandamide", "synonyms": ["arachidonoylethanolamide"]},
        {"id": "CID:5282280", "name": "2-Arachidonoylglycerol", "synonyms": ["2-AG"]},
        {"id": "CID:1123", "name": "Taurine", "synonyms": ["2-aminoethanesulfonic acid"]},
        {"id": "CID:71077", "name": "D-Serine", "synonyms": []},
        {"id": "CID:199", "name": "Agmatine", "synonyms": []},
        {"id": "CID:1001", "name": "Phenethylamine", "synonyms": ["PEA"]},
        {"id": "CID:5610", "name": "Tyramine", "synonyms": []},
        {"id": "CID:4581", "name": "Octopamine", "synonyms": []},
        {"id": "CID:1150", "name": "Tryptamine", "synonyms": []},
        {"id": "CID:3845", "name": "Kynurenic acid", "synonyms": []},
        {"id": "CID:145068", "name": "Nitric oxide", "synonyms": ["NO"]},
        {"id": "CID:65065", "name": "N-Acetylaspartic acid", "synonyms": ["NAA"]},
        {"id": "CID:1045", "name": "Putrescine", "synonyms": ["1,4-diaminobutane"]},
        {"id": "CID:1103", "name": "Spermine", "synonyms": []},
        {"id": "CID:1066", "name": "Quinolinic acid", "synonyms": []},
        {"id": "CID:22880", "name": "NMDA", "synonyms": ["NMDA"]},
    ],
    "analgesics": [
        {"id": "CID:5284371", "name": "Codeine", "synonyms": []},
        {"id": "CID:5284603", "name": "Oxycodone", "synonyms": ["OxyContin"]},
        {"id": "CID:5284569", "name": "Hydrocodone", "synonyms": ["Vicodin"]},
        {"id": "CID:3345", "name": "Fentanyl", "synonyms": []},
        {"id": "CID:33741", "name": "Tramadol", "synonyms": ["Ultram"]},
        {"id": "CID:644073", "name": "Buprenorphine", "synonyms": ["Subutex"]},
        {"id": "CID:5284596", "name": "Naloxone", "synonyms": ["Narcan"]},
        {"id": "CID:2662", "name": "Celecoxib", "synonyms": ["Celebrex"]},
        {"id": "CID:54677470", "name": "Meloxicam", "synonyms": ["Mobic"]},
        {"id": "CID:3826", "name": "Ketorolac", "synonyms": ["Toradol"]},
        {"id": "CID:54676228", "name": "Piroxicam", "synonyms": ["Feldene"]},
        {"id": "CID:3715", "name": "Indomethacin", "synonyms": ["Indocin"]},
        {"id": "CID:4058", "name": "Meperidine", "synonyms": ["Pethidine", "Demerol"]},
        {"id": "CID:5284570", "name": "Hydromorphone", "synonyms": ["Dilaudid"]},
        {"id": "CID:4095", "name": "Methadone", "synonyms": []},
        {"id": "CID:3446", "name": "Gabapentin", "synonyms": ["Neurontin"]},
        {"id": "CID:5486971", "name": "Pregabalin", "synonyms": ["Lyrica"]},
        {"id": "CID:3033", "name": "Diclofenac", "synonyms": ["Voltaren"]},
        {"id": "CID:4044", "name": "Mefenamic acid", "synonyms": ["Ponstel"]},
        {"id": "CID:3821", "name": "Ketamine", "synonyms": []},
    ],
    "antineoplastics": [
        {"id": "CID:5460033", "name": "Cisplatin", "synonyms": ["Platinol"]},
        {"id": "CID:31703", "name": "Doxorubicin", "synonyms": ["Adriamycin"]},
        {"id": "CID:126941", "name": "Methotrexate", "synonyms": ["MTX"]},
        {"id": "CID:3385", "name": "Fluorouracil", "synonyms": ["5-FU"]},
        {"id": "CID:36314", "name": "Paclitaxel", "synonyms": ["Taxol"]},
        {"id": "CID:5291", "name": "Imatinib", "synonyms": ["Gleevec"]},
        {"id": "CID:176870", "name": "Erlotinib", "synonyms": ["Tarceva"]},
        {"id": "CID:2733526", "name": "Tamoxifen", "synonyms": ["Nolvadex"]},
        {"id": "CID:5978", "name": "Vincristine", "synonyms": ["Oncovin"]},
        {"id": "CID:2907", "name": "Cyclophosphamide", "synonyms": ["Cytoxan"]},
        {"id": "CID:36462", "name": "Etoposide", "synonyms": ["VP-16"]},
        {"id": "CID:5360373", "name": "Bleomycin", "synonyms": []},
        {"id": "CID:10339178", "name": "Carboplatin", "synonyms": ["Paraplatin"]},
        {"id": "CID:60750", "name": "Gemcitabine", "synonyms": ["Gemzar"]},
        {"id": "CID:60953", "name": "Capecitabine", "synonyms": ["Xeloda"]},
        {"id": "CID:60838", "name": "Irinotecan", "synonyms": ["Camptosar"]},
        {"id": "CID:216239", "name": "Sorafenib", "synonyms": ["Nexavar"]},
        {"id": "CID:5329102", "name": "Sunitinib", "synonyms": ["Sutent"]},
        {"id": "CID:148124", "name": "Docetaxel", "synonyms": ["Taxotere"]},
        {"id": "CID:5394", "name": "Temozolomide", "synonyms": ["Temodar"]},
    ],
    "antidepressants": [
        {"id": "CID:2771", "name": "Citalopram", "synonyms": ["Celexa"]},
        {"id": "CID:146570", "name": "Escitalopram", "synonyms": ["Lexapro"]},
        {"id": "CID:43815", "name": "Paroxetine", "synonyms": ["Paxil"]},
        {"id": "CID:5656", "name": "Venlafaxine", "synonyms": ["Effexor"]},
        {"id": "CID:60835", "name": "Duloxetine", "synonyms": ["Cymbalta"]},
        {"id": "CID:2160", "name": "Amitriptyline", "synonyms": ["Elavil"]},
        {"id": "CID:4543", "name": "Nortriptyline", "synonyms": ["Pamelor"]},
        {"id": "CID:3696", "name": "Imipramine", "synonyms": ["Tofranil"]},
        {"id": "CID:2995", "name": "Desipramine", "synonyms": ["Norpramin"]},
        {"id": "CID:444", "name": "Bupropion", "synonyms": ["Wellbutrin"]},
        {"id": "CID:4205", "name": "Mirtazapine", "synonyms": ["Remeron"]},
        {"id": "CID:5533", "name": "Trazodone", "synonyms": ["Desyrel"]},
        {"id": "CID:2801", "name": "Clomipramine", "synonyms": ["Anafranil"]},
        {"id": "CID:125017", "name": "Desvenlafaxine", "synonyms": ["Pristiq"]},
        {"id": "CID:6918314", "name": "Vilazodone", "synonyms": ["Viibryd"]},
        {"id": "CID:9966051", "name": "Vortioxetine", "synonyms": ["Trintellix"]},
        {"id": "CID:3158", "name": "Doxepin", "synonyms": ["Sinequan"]},
        {"id": "CID:4011", "name": "Maprotiline", "synonyms": ["Ludiomil"]},
        {"id": "CID:4235", "name": "Moclobemide", "synonyms": ["Manerix"]},
        {"id": "CID:3675", "name": "Phenelzine", "synonyms": ["Nardil"]},
    ],
    "antivirals": [
        {"id": "CID:135398513", "name": "Acyclovir", "synonyms": ["Zovirax"]},
        {"id": "CID:135398742", "name": "Valacyclovir", "synonyms": ["Valtrex"]},
        {"id": "CID:135398740", "name": "Ganciclovir", "synonyms": ["Cytovene"]},
        {"id": "CID:121304016", "name": "Remdesivir", "synonyms": ["Veklury"]},
        {"id": "CID:45375808", "name": "Sofosbuvir", "synonyms": ["Sovaldi"]},
        {"id": "CID:464205", "name": "Tenofovir", "synonyms": ["Viread"]},
        {"id": "CID:35370", "name": "Zidovudine", "synonyms": ["AZT", "Retrovir"]},
        {"id": "CID:60825", "name": "Lamivudine", "synonyms": ["Epivir"]},
        {"id": "CID:64139", "name": "Efavirenz", "synonyms": ["Sustiva"]},
        {"id": "CID:392622", "name": "Ritonavir", "synonyms": ["Norvir"]},
        {"id": "CID:92727", "name": "Lopinavir", "synonyms": []},
        {"id": "CID:492405", "name": "Favipiravir", "synonyms": ["Avigan"]},
        {"id": "CID:145996610", "name": "Molnupiravir", "synonyms": ["Lagevrio"]},
        {"id": "CID:441300", "name": "Abacavir", "synonyms": ["Ziagen"]},
        {"id": "CID:60877", "name": "Emtricitabine", "synonyms": ["Emtriva"]},
        {"id": "CID:25154714", "name": "Daclatasvir", "synonyms": ["Daklinza"]},
        {"id": "CID:135398508", "name": "Entecavir", "synonyms": ["Baraclude"]},
        {"id": "CID:37542", "name": "Ribavirin", "synonyms": ["Virazole"]},
        {"id": "CID:3415", "name": "Foscarnet", "synonyms": ["Foscavir"]},
        {"id": "CID:2130", "name": "Amantadine", "synonyms": ["Symmetrel"]},
    ],
    "amino_acids": [
        {"id": "CID:750", "name": "Glycine", "synonyms": ["Gly"]},
        {"id": "CID:5950", "name": "L-Alanine", "synonyms": ["Ala"]},
        {"id": "CID:6287", "name": "L-Valine", "synonyms": ["Val"]},
        {"id": "CID:6106", "name": "L-Leucine", "synonyms": ["Leu"]},
        {"id": "CID:6306", "name": "L-Isoleucine", "synonyms": ["Ile"]},
        {"id": "CID:145742", "name": "L-Proline", "synonyms": ["Pro"]},
        {"id": "CID:6140", "name": "L-Phenylalanine", "synonyms": ["Phe"]},
        {"id": "CID:6305", "name": "L-Tryptophan", "synonyms": ["Trp"]},
        {"id": "CID:6137", "name": "L-Methionine", "synonyms": ["Met"]},
        {"id": "CID:5951", "name": "L-Serine", "synonyms": ["Ser"]},
        {"id": "CID:6288", "name": "L-Threonine", "synonyms": ["Thr"]},
        {"id": "CID:5862", "name": "L-Cysteine", "synonyms": ["Cys"]},
        {"id": "CID:6057", "name": "L-Tyrosine", "synonyms": ["Tyr"]},
        {"id": "CID:6267", "name": "L-Asparagine", "synonyms": ["Asn"]},
        {"id": "CID:5961", "name": "L-Glutamine", "synonyms": ["Gln"]},
        {"id": "CID:5960", "name": "L-Aspartic acid", "synonyms": ["Asp"]},
        {"id": "CID:33032", "name": "L-Glutamic acid", "synonyms": ["Glu"]},
        {"id": "CID:5962", "name": "L-Lysine", "synonyms": ["Lys"]},
        {"id": "CID:6322", "name": "L-Arginine", "synonyms": ["Arg"]},
        {"id": "CID:6274", "name": "L-Histidine", "synonyms": ["His"]},
    ],
    "sugars": [
        {"id": "CID:2723872", "name": "D-Fructose", "synonyms": ["fructose"]},
        {"id": "CID:6036", "name": "D-Galactose", "synonyms": ["galactose"]},
        {"id": "CID:5988", "name": "Sucrose", "synonyms": ["table sugar"]},
        {"id": "CID:440995", "name": "Lactose", "synonyms": ["milk sugar"]},
        {"id": "CID:439186", "name": "Maltose", "synonyms": ["malt sugar"]},
        {"id": "CID:10975657", "name": "D-Ribose", "synonyms": ["ribose"]},
        {"id": "CID:5460005", "name": "2-Deoxy-D-ribose", "synonyms": ["deoxyribose"]},
        {"id": "CID:18950", "name": "D-Mannose", "synonyms": ["mannose"]},
        {"id": "CID:7427", "name": "Trehalose", "synonyms": []},
        {"id": "CID:135191", "name": "D-Xylose", "synonyms": ["xylose"]},
        {"id": "CID:439195", "name": "L-Arabinose", "synonyms": ["arabinose"]},
        {"id": "CID:5780", "name": "Sorbitol", "synonyms": ["glucitol"]},
        {"id": "CID:6251", "name": "Mannitol", "synonyms": []},
        {"id": "CID:439178", "name": "Cellobiose", "synonyms": []},
        {"id": "CID:439242", "name": "Raffinose", "synonyms": []},
        {"id": "CID:439531", "name": "Stachyose", "synonyms": []},
        {"id": "CID:892", "name": "myo-Inositol", "synonyms": ["inositol"]},
        {"id": "CID:1738118", "name": "N-Acetylglucosamine", "synonyms": ["GlcNAc"]},
        {"id": "CID:94715", "name": "Glucuronic acid", "synonyms": []},
        {"id": "CID:94176", "name": "D-Erythrose", "synonyms": ["erythrose"]},
    ],
    "nucleotides": [
        {"id": "CID:135398633", "name": "GTP", "synonyms": ["GTP"]},
        {"id": "CID:6176", "name": "CTP", "synonyms": ["CTP"]},
        {"id": "CID:6133", "name": "UTP", "synonyms": ["UTP"]},
        {"id": "CID:6022", "name": "ADP", "synonyms": ["ADP"]},
        {"id": "CID:6083", "name": "AMP", "synonyms": ["AMP"]},
        {"id": "CID:135398619", "name": "GDP", "synonyms": ["GDP"]},
        {"id": "CID:6076", "name": "cAMP", "synonyms": ["cAMP"]},
        {"id": "CID:135398570", "name": "cGMP", "synonyms": ["cGMP"]},
        {"id": "CID:5892", "name": "NAD+", "synonyms": ["nicotinamide adenine dinucleotide"]},
        {"id": "CID:439153", "name": "NADH", "synonyms": []},
        {"id": "CID:5885", "name": "NADP+", "synonyms": []},
        {"id": "CID:643975", "name": "FAD", "synonyms": ["flavin adenine dinucleotide"]},
        {"id": "CID:87642", "name": "Coenzyme A", "synonyms": ["CoA"]},
        {"id": "CID:34755", "name": "S-Adenosylmethionine", "synonyms": ["SAM"]},
        {"id": "CID:6030", "name": "UMP", "synonyms": ["UMP"]},
        {"id": "CID:135398631", "name": "GMP", "synonyms": ["GMP"]},
        {"id": "CID:6131", "name": "CMP", "synonyms": ["CMP"]},
        {"id": "CID:9700", "name": "TMP", "synonyms": ["dTMP"]},
        {"id": "CID:135398640", "name": "IMP", "synonyms": ["IMP"]},
        {"id": "CID:6132", "name": "CDP", "synonyms": ["CDP"]},
    ],
    "antifungals": [
        {"id": "CID:3365", "name": "Fluconazole", "synonyms": ["Diflucan"]},
        {"id": "CID:55283", "name": "Itraconazole", "synonyms": ["Sporanox"]},
        {"id": "CID:71616", "name": "Voriconazole", "synonyms": ["Vfend"]},
        {"id": "CID:5280965", "name": "Amphotericin B", "synonyms": ["Fungizone"]},
        {"id": "CID:6433272", "name": "Nystatin", "synonyms": ["Mycostatin"]},
        {"id": "CID:1549008", "name": "Terbinafine", "synonyms": ["Lamisil"]},
        {"id": "CID:16119814", "name": "Caspofungin", "synonyms": ["Cancidas"]},
        {"id": "CID:477468", "name": "Micafungin", "synonyms": ["Mycamine"]},
        {"id": "CID:166548", "name": "Anidulafungin", "synonyms": ["Eraxis"]},
        {"id": "CID:2812", "name": "Clotrimazole", "synonyms": ["Lotrimin"]},
        {"id": "CID:47576", "name": "Ketoconazole", "synonyms": ["Nizoral"]},
        {"id": "CID:4189", "name": "Miconazole", "synonyms": ["Monistat"]},
        {"id": "CID:468595", "name": "Posaconazole", "synonyms": ["Noxafil"]},
        {"id": "CID:6918485", "name": "Isavuconazole", "synonyms": ["Cresemba"]},
        {"id": "CID:441140", "name": "Griseofulvin", "synonyms": ["Grifulvin"]},
        {"id": "CID:3366", "name": "Flucytosine", "synonyms": ["Ancobon"]},
        {"id": "CID:3198", "name": "Econazole", "synonyms": ["Spectazole"]},
        {"id": "CID:5284447", "name": "Natamycin", "synonyms": ["pimaricin"]},
        {"id": "CID:2749", "name": "Ciclopirox", "synonyms": ["Loprox"]},
        {"id": "CID:5510", "name": "Tolnaftate", "synonyms": ["Tinactin"]},
    ],
    "toxins": [
        {"id": "CID:186907", "name": "Aflatoxin B1", "synonyms": []},
        {"id": "CID:11174599", "name": "Tetrodotoxin", "synonyms": ["TTX"]},
        {"id": "CID:56947150", "name": "Saxitoxin", "synonyms": ["STX"]},
        {"id": "CID:441071", "name": "Strychnine", "synonyms": []},
        {"id": "CID:15939", "name": "Paraquat", "synonyms": []},
        {"id": "CID:10666", "name": "Ricinine", "synonyms": []},
        {"id": "CID:6167", "name": "Colchicine", "synonyms": []},
        {"id": "CID:245005", "name": "Aconitine", "synonyms": []},
        {"id": "CID:768", "name": "Hydrogen cyanide", "synonyms": ["HCN"]},
        {"id": "CID:14888", "name": "Arsenic trioxide", "synonyms": []},
        {"id": "CID:89594", "name": "Nicotine", "synonyms": []},
        {"id": "CID:9308", "name": "Muscarine", "synonyms": []},
        {"id": "CID:442530", "name": "Ochratoxin A", "synonyms": []},
        {"id": "CID:445434", "name": "Microcystin-LR", "synonyms": []},
        {"id": "CID:6758", "name": "Rotenone", "synonyms": []},
        {"id": "CID:24085", "name": "Mercuric chloride", "synonyms": ["mercury(II) chloride"]},
        {"id": "CID:1548943", "name": "Capsaicin", "synonyms": []},
        {"id": "CID:6324647", "name": "Batrachotoxin", "synonyms": []},
        {"id": "CID:442021", "name": "Brucine", "synonyms": []},
        {"id": "CID:8223", "name": "Ergotamine", "synonyms": []},
    ],
    "metabolites": [
        {"id": "CID:1060", "name": "Pyruvic acid", "synonyms": ["pyruvate"]},
        {"id": "CID:107689", "name": "L-Lactic acid", "synonyms": ["lactate"]},
        {"id": "CID:1110", "name": "Succinic acid", "synonyms": ["succinate"]},
        {"id": "CID:444972", "name": "Fumaric acid", "synonyms": ["fumarate"]},
        {"id": "CID:222656", "name": "L-Malic acid", "synonyms": ["malate"]},
        {"id": "CID:970", "name": "Oxaloacetic acid", "synonyms": ["oxaloacetate"]},
        {"id": "CID:51", "name": "alpha-Ketoglutaric acid", "synonyms": ["alpha-ketoglutarate"]},
        {"id": "CID:588", "name": "Creatinine", "synonyms": []},
        {"id": "CID:1175", "name": "Uric acid", "synonyms": []},
        {"id": "CID:5280352", "name": "Bilirubin", "synonyms": []},
        {"id": "CID:180", "name": "Acetone", "synonyms": []},
        {"id": "CID:441", "name": "3-Hydroxybutyric acid", "synonyms": ["beta-hydroxybutyrate"]},
        {"id": "CID:971", "name": "Oxalic acid", "synonyms": ["oxalate"]},
        {"id": "CID:753", "name": "Glycerol", "synonyms": ["glycerin"]},
        {"id": "CID:177", "name": "Acetaldehyde", "synonyms": ["ethanal"]},
        {"id": "CID:284", "name": "Formic acid", "synonyms": ["methanoic acid"]},
        {"id": "CID:1198", "name": "Isocitric acid", "synonyms": ["isocitrate"]},
        {"id": "CID:643757", "name": "cis-Aconitic acid", "synonyms": ["aconitate"]},
        {"id": "CID:1005", "name": "Phosphoenolpyruvic acid", "synonyms": ["PEP"]},
        {"id": "CID:668", "name": "Dihydroxyacetone phosphate", "synonyms": ["DHAP"]},
    ],
    "steroids": [
        {"id": "CID:5839", "name": "Aldosterone", "synonyms": []},
        {"id": "CID:5753", "name": "Corticosterone", "synonyms": []},
        {"id": "CID:5881", "name": "Dehydroepiandrosterone", "synonyms": ["DHEA"]},
        {"id": "CID:8955", "name": "Pregnenolone", "synonyms": []},
        {"id": "CID:6128", "name": "Androstenedione", "synonyms": []},
        {"id": "CID:10635", "name": "Dihydrotestosterone", "synonyms": ["DHT"]},
        {"id": "CID:5756", "name": "Estriol", "synonyms": []},
        {"id": "CID:5870", "name": "Estrone", "synonyms": []},
        {"id": "CID:440707", "name": "11-Deoxycortisol", "synonyms": []},
        {"id": "CID:5755", "name": "Prednisolone", "synonyms": []},
        {"id": "CID:6741", "name": "Methylprednisolone", "synonyms": ["Medrol"]},
        {"id": "CID:9782", "name": "Betamethasone", "synonyms": []},
        {"id": "CID:31378", "name": "Fludrocortisone", "synonyms": ["Florinef"]},
        {"id": "CID:31307", "name": "Triamcinolone", "synonyms": ["Kenalog"]},
        {"id": "CID:5281004", "name": "Budesonide", "synonyms": ["Pulmicort"]},
        {"id": "CID:222528", "name": "Deoxycholic acid", "synonyms": []},
        {"id": "CID:10133", "name": "Chenodeoxycholic acid", "synonyms": ["CDCA"]},
        {"id": "CID:31401", "name": "Ursodeoxycholic acid", "synonyms": ["UDCA"]},
        {"id": "CID:9903", "name": "Lithocholic acid", "synonyms": []},
        {"id": "CID:221493", "name": "Cholic acid", "synonyms": []},
    ],
    # --- New protein topics (UniProt) ---
    "kinases": [
        {"id": "P11802", "name": "CDK4", "organism": "Homo sapiens"},
        {"id": "Q00534", "name": "CDK6", "organism": "Homo sapiens"},
        {"id": "P06493", "name": "CDK1", "organism": "Homo sapiens"},
        {"id": "O60674", "name": "JAK2", "organism": "Homo sapiens"},
        {"id": "P12931", "name": "SRC", "organism": "Homo sapiens"},
        {"id": "P31749", "name": "AKT1", "organism": "Homo sapiens"},
        {"id": "P28482", "name": "MAPK1 (ERK2)", "organism": "Homo sapiens"},
        {"id": "P17612", "name": "PRKACA (PKA catalytic alpha)", "organism": "Homo sapiens"},
        {"id": "P17252", "name": "PRKCA (PKC-alpha)", "organism": "Homo sapiens"},
        {"id": "O14965", "name": "Aurora kinase A", "organism": "Homo sapiens"},
        {"id": "P53350", "name": "PLK1", "organism": "Homo sapiens"},
        {"id": "P49841", "name": "GSK3-beta", "organism": "Homo sapiens"},
        {"id": "P68400", "name": "CK2 alpha", "organism": "Homo sapiens"},
        {"id": "Q13464", "name": "ROCK1", "organism": "Homo sapiens"},
        {"id": "Q13153", "name": "PAK1", "organism": "Homo sapiens"},
        {"id": "Q06187", "name": "BTK", "organism": "Homo sapiens"},
        {"id": "P06241", "name": "FYN", "organism": "Homo sapiens"},
        {"id": "P06239", "name": "LCK", "organism": "Homo sapiens"},
        {"id": "P36888", "name": "FLT3", "organism": "Homo sapiens"},
        {"id": "P10721", "name": "KIT", "organism": "Homo sapiens"},
    ],
    "proteases": [
        {"id": "Q99895", "name": "Chymotrypsin-C", "organism": "Homo sapiens"},
        {"id": "P42574", "name": "Caspase-3", "organism": "Homo sapiens"},
        {"id": "P55211", "name": "Caspase-9", "organism": "Homo sapiens"},
        {"id": "P07858", "name": "Cathepsin B", "organism": "Homo sapiens"},
        {"id": "P07339", "name": "Cathepsin D", "organism": "Homo sapiens"},
        {"id": "P14780", "name": "MMP-9", "organism": "Homo sapiens"},
        {"id": "P08253", "name": "MMP-2", "organism": "Homo sapiens"},
        {"id": "P20142", "name": "Gastricsin", "organism": "Homo sapiens"},
        {"id": "P08246", "name": "Neutrophil elastase", "organism": "Homo sapiens"},
        {"id": "P06870", "name": "Kallikrein-1", "organism": "Homo sapiens"},
        {"id": "P78536", "name": "ADAM17", "organism": "Homo sapiens"},
        {"id": "P10144", "name": "Granzyme B", "organism": "Homo sapiens"},
        {"id": "P09958", "name": "Furin", "organism": "Homo sapiens"},
        {"id": "O14672", "name": "ADAM10", "organism": "Homo sapiens"},
        {"id": "P07711", "name": "Cathepsin L1", "organism": "Homo sapiens"},
        {"id": "P43235", "name": "Cathepsin K", "organism": "Homo sapiens"},
        {"id": "P50281", "name": "MMP-14", "organism": "Homo sapiens"},
        {"id": "Q14790", "name": "Caspase-8", "organism": "Homo sapiens"},
        {"id": "Q92743", "name": "HTRA1", "organism": "Homo sapiens"},
        {"id": "P07384", "name": "Calpain-1", "organism": "Homo sapiens"},
    ],
    "transporters": [
        {"id": "P11166", "name": "GLUT1", "organism": "Homo sapiens"},
        {"id": "P14672", "name": "GLUT4", "organism": "Homo sapiens"},
        {"id": "P31639", "name": "SGLT2", "organism": "Homo sapiens"},
        {"id": "P08183", "name": "P-glycoprotein (MDR1)", "organism": "Homo sapiens"},
        {"id": "P29972", "name": "Aquaporin-1", "organism": "Homo sapiens"},
        {"id": "P41181", "name": "Aquaporin-2", "organism": "Homo sapiens"},
        {"id": "P05023", "name": "Na/K-ATPase alpha-1", "organism": "Homo sapiens"},
        {"id": "P31645", "name": "Serotonin transporter", "organism": "Homo sapiens"},
        {"id": "Q01959", "name": "Dopamine transporter", "organism": "Homo sapiens"},
        {"id": "P23975", "name": "Norepinephrine transporter", "organism": "Homo sapiens"},
        {"id": "Q05940", "name": "VMAT2", "organism": "Homo sapiens"},
        {"id": "P02786", "name": "Transferrin receptor 1", "organism": "Homo sapiens"},
        {"id": "P01130", "name": "LDL receptor", "organism": "Homo sapiens"},
        {"id": "Q9UNQ0", "name": "BCRP (ABCG2)", "organism": "Homo sapiens"},
        {"id": "P33527", "name": "MRP1 (ABCC1)", "organism": "Homo sapiens"},
        {"id": "Q9Y6L6", "name": "OATP1B1", "organism": "Homo sapiens"},
        {"id": "P53985", "name": "MCT1", "organism": "Homo sapiens"},
        {"id": "Q9Y6R1", "name": "NBC1 (SLC4A4)", "organism": "Homo sapiens"},
        {"id": "P19634", "name": "NHE1", "organism": "Homo sapiens"},
        {"id": "P30531", "name": "GAT1", "organism": "Homo sapiens"},
    ],
    "cytokines": [
        {"id": "P01584", "name": "IL-1 beta", "organism": "Homo sapiens"},
        {"id": "P60568", "name": "IL-2", "organism": "Homo sapiens"},
        {"id": "P05112", "name": "IL-4", "organism": "Homo sapiens"},
        {"id": "P05231", "name": "IL-6", "organism": "Homo sapiens"},
        {"id": "P22301", "name": "IL-10", "organism": "Homo sapiens"},
        {"id": "P29460", "name": "IL-12 beta", "organism": "Homo sapiens"},
        {"id": "Q16552", "name": "IL-17A", "organism": "Homo sapiens"},
        {"id": "P01579", "name": "IFN-gamma", "organism": "Homo sapiens"},
        {"id": "P01563", "name": "IFN-alpha 2", "organism": "Homo sapiens"},
        {"id": "P01137", "name": "TGF-beta 1", "organism": "Homo sapiens"},
        {"id": "P09919", "name": "G-CSF", "organism": "Homo sapiens"},
        {"id": "P04141", "name": "GM-CSF", "organism": "Homo sapiens"},
        {"id": "P09603", "name": "M-CSF", "organism": "Homo sapiens"},
        {"id": "P10145", "name": "IL-8 (CXCL8)", "organism": "Homo sapiens"},
        {"id": "P13500", "name": "MCP-1 (CCL2)", "organism": "Homo sapiens"},
        {"id": "P15692", "name": "VEGF-A", "organism": "Homo sapiens"},
        {"id": "P01127", "name": "PDGF-B", "organism": "Homo sapiens"},
        {"id": "P09038", "name": "FGF2", "organism": "Homo sapiens"},
        {"id": "P01583", "name": "IL-1 alpha", "organism": "Homo sapiens"},
        {"id": "P35225", "name": "IL-13", "organism": "Homo sapiens"},
    ],
    "structural_proteins": [
        {"id": "P02462", "name": "Collagen alpha-1(IV)", "organism": "Homo sapiens"},
        {"id": "P02461", "name": "Collagen alpha-1(III)", "organism": "Homo sapiens"},
        {"id": "P15502", "name": "Elastin", "organism": "Homo sapiens"},
        {"id": "P25391", "name": "Laminin subunit alpha-1", "organism": "Homo sapiens"},
        {"id": "P08670", "name": "Vimentin", "organism": "Homo sapiens"},
        {"id": "P17661", "name": "Desmin", "organism": "Homo sapiens"},
        {"id": "P02549", "name": "Spectrin alpha chain", "organism": "Homo sapiens"},
        {"id": "Q8WZ42", "name": "Titin", "organism": "Homo sapiens"},
        {"id": "P20929", "name": "Nebulin", "organism": "Homo sapiens"},
        {"id": "P11532", "name": "Dystrophin", "organism": "Homo sapiens"},
        {"id": "P02545", "name": "Lamin A/C", "organism": "Homo sapiens"},
        {"id": "P02533", "name": "Keratin type I 14", "organism": "Homo sapiens"},
        {"id": "Q15149", "name": "Plectin", "organism": "Homo sapiens"},
        {"id": "P21333", "name": "Filamin-A", "organism": "Homo sapiens"},
        {"id": "P18206", "name": "Vinculin", "organism": "Homo sapiens"},
        {"id": "Q9Y490", "name": "Talin-1", "organism": "Homo sapiens"},
        {"id": "P12814", "name": "Alpha-actinin-1", "organism": "Homo sapiens"},
        {"id": "P15924", "name": "Desmoplakin", "organism": "Homo sapiens"},
        {"id": "P06756", "name": "Integrin alpha-V", "organism": "Homo sapiens"},
        {"id": "P05556", "name": "Integrin beta-1", "organism": "Homo sapiens"},
    ],
    "transcription_factors": [
        {"id": "Q16665", "name": "HIF-1-alpha", "organism": "Homo sapiens"},
        {"id": "P01106", "name": "Myc proto-oncogene protein", "organism": "Homo sapiens"},
        {"id": "P05412", "name": "JUN", "organism": "Homo sapiens"},
        {"id": "P01100", "name": "FOS", "organism": "Homo sapiens"},
        {"id": "P48431", "name": "SOX2", "organism": "Homo sapiens"},
        {"id": "Q01860", "name": "POU5F1 (OCT4)", "organism": "Homo sapiens"},
        {"id": "Q9H9S0", "name": "NANOG", "organism": "Homo sapiens"},
        {"id": "Q9BZS1", "name": "FOXP3", "organism": "Homo sapiens"},
        {"id": "Q9H3D4", "name": "TP63", "organism": "Homo sapiens"},
        {"id": "P16220", "name": "CREB1", "organism": "Homo sapiens"},
        {"id": "P08047", "name": "SP1", "organism": "Homo sapiens"},
        {"id": "Q01094", "name": "E2F1", "organism": "Homo sapiens"},
        {"id": "P06400", "name": "RB1", "organism": "Homo sapiens"},
        {"id": "Q13485", "name": "SMAD4", "organism": "Homo sapiens"},
        {"id": "P15976", "name": "GATA-1", "organism": "Homo sapiens"},
        {"id": "P14921", "name": "ETS-1", "organism": "Homo sapiens"},
        {"id": "Q13469", "name": "NFATC2", "organism": "Homo sapiens"},
        {"id": "P46937", "name": "YAP1", "organism": "Homo sapiens"},
        {"id": "Q13950", "name": "RUNX2", "organism": "Homo sapiens"},
        {"id": "Q16236", "name": "NRF2 (NFE2L2)", "organism": "Homo sapiens"},
    ],
    # --- New organism topics (Wikidata) ---
    "viruses": [
        {"id": "Q82069695", "name": "SARS-CoV-2", "taxid": 2697049},
        {"id": "Q18907320", "name": "HIV-1", "taxid": 11676},
        {"id": "Q834390", "name": "Influenza A virus", "taxid": 11320},
        {"id": "Q10538943", "name": "Ebola virus", "taxid": 186538},
        {"id": "Q708693", "name": "Hepatitis C virus", "taxid": 11103},
        {"id": "Q6844", "name": "Hepatitis B virus", "taxid": 10407},
        {"id": "Q69461345", "name": "HPV-16", "taxid": 333760},
        {"id": "Q202864", "name": "Zika virus", "taxid": 64320},
        {"id": "Q476209", "name": "Dengue virus", "taxid": 12637},
        {"id": "Q698976", "name": "Rabies virus", "taxid": 11292},
        {"id": "Q573943", "name": "Measles virus", "taxid": 11234},
        {"id": "Q6900", "name": "Epstein-Barr virus", "taxid": 10376},
        {"id": "Q4902157", "name": "MERS-CoV", "taxid": 1335626},
        {"id": "Q1052913", "name": "RSV", "taxid": 12814},
        {"id": "Q6868", "name": "Varicella-zoster virus", "taxid": 10335},
        {"id": "Q6929", "name": "Cytomegalovirus", "taxid": 10359},
        {"id": "Q158856", "name": "West Nile virus", "taxid": 11082},
        {"id": "Q15794049", "name": "Chikungunya virus", "taxid": 37124},
        {"id": "Q6755280", "name": "Marburg virus", "taxid": 11269},
        {"id": "Q836749", "name": "Yellow fever virus", "taxid": 11089},
    ],
    "parasites": [
        {"id": "Q155630", "name": "Giardia lamblia", "taxid": 5741},
        {"id": "Q131027", "name": "Entamoeba histolytica", "taxid": 5759},
        {"id": "Q132595", "name": "Trichomonas vaginalis", "taxid": 5722},
        {"id": "Q134734", "name": "Cryptosporidium parvum", "taxid": 5807},
        {"id": "Q7777786", "name": "Babesia microti", "taxid": 5868},
        {"id": "Q311109", "name": "Wuchereria bancrofti", "taxid": 6293},
        {"id": "Q139658", "name": "Brugia malayi", "taxid": 6279},
        {"id": "Q137224", "name": "Onchocerca volvulus", "taxid": 6282},
        {"id": "Q468771", "name": "Ascaris lumbricoides", "taxid": 6249},
        {"id": "Q2433913", "name": "Necator americanus", "taxid": 51031},
        {"id": "Q244111", "name": "Strongyloides stercoralis", "taxid": 6248},
        {"id": "Q630200", "name": "Echinococcus granulosus", "taxid": 6210},
        {"id": "Q565708", "name": "Taenia solium", "taxid": 6204},
        {"id": "Q334149", "name": "Fasciola hepatica", "taxid": 6192},
        {"id": "Q135523", "name": "Opisthorchis viverrini", "taxid": 6198},
        {"id": "Q730937", "name": "Trichinella spiralis", "taxid": 6334},
        {"id": "Q2520810", "name": "Ancylostoma duodenale", "taxid": 29170},
        {"id": "Q1502531", "name": "Dracunculus medinensis", "taxid": 318479},
        {"id": "Q134361", "name": "Loa loa", "taxid": 7209},
        {"id": "Q1256989", "name": "Schistosoma japonicum", "taxid": 6182},
    ],
}

# Categories/themes for biochem bench
BIOCHEM_THEMES: Dict[str, Dict[str, Any]] = {
    "proteins": {"min_entities": 30},
    "compounds": {"min_entities": 30},
    "organisms": {"min_entities": 20},
    "enzymes": {"min_entities": 10},
    "receptors": {"min_entities": 10},
    "antibiotics": {"min_entities": 10},
    "vitamins": {"min_entities": 10},
    "hormones": {"min_entities": 10},
    "lipids": {"min_entities": 10},
    "drugs": {"min_entities": 10},
    # New topics (2026-03-16)
    "neurotransmitters": {"min_entities": 10},
    "analgesics": {"min_entities": 10},
    "antineoplastics": {"min_entities": 10},
    "antidepressants": {"min_entities": 10},
    "antivirals": {"min_entities": 10},
    "amino_acids": {"min_entities": 10},
    "sugars": {"min_entities": 10},
    "nucleotides": {"min_entities": 10},
    "antifungals": {"min_entities": 10},
    "toxins": {"min_entities": 10},
    "metabolites": {"min_entities": 10},
    "steroids": {"min_entities": 10},
    "kinases": {"min_entities": 10},
    "proteases": {"min_entities": 10},
    "transporters": {"min_entities": 10},
    "cytokines": {"min_entities": 10},
    "structural_proteins": {"min_entities": 10},
    "transcription_factors": {"min_entities": 10},
    "viruses": {"min_entities": 10},
    "parasites": {"min_entities": 10},
}

# Available properties per data source
PUBCHEM_PROPERTIES = [
    "MolecularWeight", "MolecularFormula", "XLogP", "TPSA",
    "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
    "HeavyAtomCount", "Complexity", "ExactMass", "MonoisotopicMass",
    "Charge", "IsomericSMILES", "CanonicalSMILES", "IUPACName",
]

UNIPROT_FIELDS = [
    "accession", "protein_name", "gene_names", "organism_name",
    "length", "mass", "sequence", "cc_function", "cc_subcellular_location",
    "ec", "ft_domain", "xref_pdb",
]


# ---------------------------------------------------------------------------
# Caching & Rate Limiting
# ---------------------------------------------------------------------------

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(BIO_CACHE_DIR, exist_ok=True)


def _cached_get(url: str, cache_key: str, max_age_hours: int = 168,
                headers: Optional[dict] = None) -> Optional[dict]:
    """Fetch URL with local JSON cache and rate limiting.

    Args:
        url: URL to fetch.
        cache_key: Key for local cache filename (sanitized).
        max_age_hours: Maximum age of cached data in hours (default 1 week).
        headers: Optional custom headers.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    global _last_request_time
    _ensure_cache_dir()

    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(BIO_CACHE_DIR, f"{safe_key}.json")

    # Check cache
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < max_age_hours:
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

    # Rate limit
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)

    hdrs = headers or _HEADERS

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=hdrs, timeout=30)
            _last_request_time = time.time()

            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            elif response.status_code in (429, 503):
                wait = (2 ** attempt) * 5 + random.random() * 5
                print(f"Bio API rate-limited ({response.status_code}) for {cache_key}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"API error {response.status_code} for {cache_key}", flush=True)
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.random() * 5
            print(f"Bio API request error for {cache_key}: {e}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"Unexpected error for {cache_key}: {e}", flush=True)
            return None

    print(f"Bio API request failed after {max_retries} retries for {cache_key}", flush=True)
    return None


# ---------------------------------------------------------------------------
# PubChem API Functions
# ---------------------------------------------------------------------------

def get_compound_data(name_or_cid: str) -> Optional[Dict[str, Any]]:
    """Fetch compound properties from PubChem.

    Args:
        name_or_cid: Compound name (e.g., "aspirin") or CID (e.g., "2244").

    Returns:
        Dict with MW, formula, XLogP, TPSA, HBA, HBD, etc., or None.
    """
    # Determine if input is a CID number or name
    cid_match = re.match(r'^(?:CID:?\s*)?(\d+)$', str(name_or_cid).strip())
    if cid_match:
        cid = cid_match.group(1)
        lookup = f"cid/{cid}"
        cache_key = f"pubchem_cid_{cid}"
    else:
        name = str(name_or_cid).strip()
        lookup = f"name/{requests.utils.quote(name)}"
        cache_key = f"pubchem_name_{name.lower().replace(' ', '_')}"

    props = ",".join(PUBCHEM_PROPERTIES)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{lookup}/property/{props}/JSON"

    data = _cached_get(url, cache_key)
    if not data:
        return None

    properties_list = data.get("PropertyTable", {}).get("Properties", [])
    if not properties_list:
        return None

    props_dict = properties_list[0]

    result = {
        "cid": props_dict.get("CID"),
        "molecular_weight": props_dict.get("MolecularWeight"),
        "molecular_formula": props_dict.get("MolecularFormula"),
        "xlogp": props_dict.get("XLogP"),
        "tpsa": props_dict.get("TPSA"),
        "hbond_donor_count": props_dict.get("HBondDonorCount"),
        "hbond_acceptor_count": props_dict.get("HBondAcceptorCount"),
        "rotatable_bond_count": props_dict.get("RotatableBondCount"),
        "heavy_atom_count": props_dict.get("HeavyAtomCount"),
        "complexity": props_dict.get("Complexity"),
        "exact_mass": props_dict.get("ExactMass"),
        "monoisotopic_mass": props_dict.get("MonoisotopicMass"),
        "charge": props_dict.get("Charge"),
        "isomeric_smiles": props_dict.get("IsomericSMILES"),
        "canonical_smiles": props_dict.get("CanonicalSMILES"),
        "iupac_name": props_dict.get("IUPACName"),
    }

    # Convert MW to float if it's a string
    if result["molecular_weight"] is not None:
        try:
            result["molecular_weight"] = float(result["molecular_weight"])
        except (ValueError, TypeError):
            pass

    return result


def get_compound_properties(name_or_cid: str,
                            property_list: List[str]) -> Optional[Dict[str, Any]]:
    """Fetch specific properties for a compound from PubChem.

    Args:
        name_or_cid: Compound name or CID.
        property_list: List of PubChem property names.

    Returns:
        Dict of property values, or None.
    """
    cid_match = re.match(r'^(?:CID:?\s*)?(\d+)$', str(name_or_cid).strip())
    if cid_match:
        cid = cid_match.group(1)
        lookup = f"cid/{cid}"
        cache_key = f"pubchem_props_{cid}_{'_'.join(sorted(property_list))}"
    else:
        name = str(name_or_cid).strip()
        lookup = f"name/{requests.utils.quote(name)}"
        cache_key = f"pubchem_props_{name.lower().replace(' ', '_')}_{'_'.join(sorted(property_list))}"

    props = ",".join(property_list)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{lookup}/property/{props}/JSON"

    data = _cached_get(url, cache_key)
    if not data:
        return None

    properties_list = data.get("PropertyTable", {}).get("Properties", [])
    if not properties_list:
        return None

    return properties_list[0]


def search_compound(query: str) -> List[Dict[str, Any]]:
    """Search PubChem for compounds matching a query.

    Args:
        query: Search string (compound name or partial name).

    Returns:
        List of {cid, name, molecular_formula, molecular_weight} dicts.
    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(query)}/property/MolecularWeight,MolecularFormula,IUPACName/JSON"
    )
    cache_key = f"pubchem_search_{query.lower().replace(' ', '_')}"

    data = _cached_get(url, cache_key)
    if not data:
        return []

    properties_list = data.get("PropertyTable", {}).get("Properties", [])
    results = []
    for p in properties_list[:10]:
        results.append({
            "cid": p.get("CID"),
            "name": query,
            "iupac_name": p.get("IUPACName", ""),
            "molecular_formula": p.get("MolecularFormula", ""),
            "molecular_weight": p.get("MolecularWeight"),
        })
    return results


# ---------------------------------------------------------------------------
# UniProt API Functions
# ---------------------------------------------------------------------------

def get_protein_data(uniprot_id: str) -> Optional[Dict[str, Any]]:
    """Fetch protein data from UniProt.

    Args:
        uniprot_id: UniProt accession (e.g., "P01308").

    Returns:
        Dict with name, organism, MW, length, sequence, function, etc., or None.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    cache_key = f"uniprot_{uniprot_id}"

    data = _cached_get(url, cache_key)
    if not data:
        return None

    # Extract protein name
    protein_name = ""
    rec_name = data.get("proteinDescription", {}).get("recommendedName", {})
    if rec_name:
        protein_name = rec_name.get("fullName", {}).get("value", "")
    if not protein_name:
        sub_names = data.get("proteinDescription", {}).get("submittedName", [])
        if sub_names:
            protein_name = sub_names[0].get("fullName", {}).get("value", "")

    # Extract gene names
    gene_names = []
    for gene in data.get("genes", []):
        if "geneName" in gene:
            gene_names.append(gene["geneName"].get("value", ""))

    # Extract organism
    organism = data.get("organism", {}).get("scientificName", "")

    # Extract sequence
    sequence = data.get("sequence", {})
    seq_str = sequence.get("value", "")
    seq_length = sequence.get("length", 0)
    seq_mass = sequence.get("molWeight", 0)

    # Extract function
    function_text = ""
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function_text = texts[0].get("value", "")
                break

    # Extract subcellular location
    subcellular_location = ""
    for comment in data.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            locs = comment.get("subcellularLocations", [])
            if locs:
                loc = locs[0].get("location", {}).get("value", "")
                subcellular_location = loc
                break

    # Extract EC number
    ec_numbers = []
    ec_data = data.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", [])
    for ec in ec_data:
        ec_numbers.append(ec.get("value", ""))

    # Extract PDB cross-references
    pdb_ids = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "PDB":
            pdb_ids.append(xref.get("id", ""))

    # Count amino acids in sequence
    aa_counts = {}
    for aa in seq_str:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    result = {
        "uniprot_id": uniprot_id,
        "protein_name": protein_name,
        "gene_names": gene_names,
        "organism": organism,
        "sequence": seq_str,
        "length": seq_length,
        "molecular_weight": seq_mass,  # In Da
        "function": function_text,
        "subcellular_location": subcellular_location,
        "ec_numbers": ec_numbers,
        "pdb_ids": pdb_ids[:5],  # Keep top 5
        "amino_acid_counts": aa_counts,
    }

    return result


def search_protein(query: str) -> List[Dict[str, Any]]:
    """Search UniProt for proteins matching a query.

    Args:
        query: Search string (protein name, gene name, etc.).

    Returns:
        List of {uniprot_id, name, organism, length} dicts.
    """
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query={requests.utils.quote(query)}&format=json&size=10"
        f"&fields=accession,protein_name,organism_name,length"
    )
    cache_key = f"uniprot_search_{query.lower().replace(' ', '_')}"

    data = _cached_get(url, cache_key)
    if not data:
        return []

    results = []
    for entry in data.get("results", []):
        name = ""
        rec = entry.get("proteinDescription", {}).get("recommendedName", {})
        if rec:
            name = rec.get("fullName", {}).get("value", "")
        if not name:
            subs = entry.get("proteinDescription", {}).get("submittedName", [])
            if subs:
                name = subs[0].get("fullName", {}).get("value", "")

        results.append({
            "uniprot_id": entry.get("primaryAccession", ""),
            "name": name,
            "organism": entry.get("organism", {}).get("scientificName", ""),
            "length": entry.get("sequence", {}).get("length", 0),
        })

    return results


# ---------------------------------------------------------------------------
# RCSB PDB API Functions
# ---------------------------------------------------------------------------

def get_pdb_data(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Fetch protein structure data from RCSB PDB.

    Args:
        pdb_id: PDB ID (e.g., "1HHO").

    Returns:
        Dict with resolution, atom count, cell dimensions, etc., or None.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    cache_key = f"pdb_{pdb_id.upper()}"

    data = _cached_get(url, cache_key)
    if not data:
        return None

    # Extract key structure information
    cell = data.get("cell", {})
    exptl = data.get("exptl", [{}])[0] if data.get("exptl") else {}
    refine = data.get("refine", [{}])[0] if data.get("refine") else {}
    struct = data.get("struct", {})

    result = {
        "pdb_id": pdb_id.upper(),
        "title": struct.get("title", ""),
        "method": exptl.get("method", ""),
        "resolution": refine.get("ls_d_res_high"),
        "r_factor": refine.get("ls_R_factor_R_work"),
        "r_free": refine.get("ls_R_factor_R_free"),
        "cell_length_a": cell.get("length_a"),
        "cell_length_b": cell.get("length_b"),
        "cell_length_c": cell.get("length_c"),
        "cell_angle_alpha": cell.get("angle_alpha"),
        "cell_angle_beta": cell.get("angle_beta"),
        "cell_angle_gamma": cell.get("angle_gamma"),
        "space_group": data.get("symmetry", {}).get("space_group_name_H_M", ""),
    }

    # Try to get atom count from polymer entities
    polymer_entities = data.get("rcsb_entry_info", {})
    if polymer_entities:
        result["polymer_entity_count"] = polymer_entities.get("polymer_entity_count")
        result["deposited_atom_count"] = polymer_entities.get("deposited_atom_count")
        result["deposited_modeled_polymer_monomer_count"] = polymer_entities.get(
            "deposited_modeled_polymer_monomer_count"
        )

    return result


# ---------------------------------------------------------------------------
# ChEMBL API Functions
# ---------------------------------------------------------------------------

def get_chembl_data(chembl_id: str) -> Optional[Dict[str, Any]]:
    """Fetch drug/molecule data from ChEMBL.

    Args:
        chembl_id: ChEMBL ID (e.g., "CHEMBL25").

    Returns:
        Dict with properties and bioactivity data, or None.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    cache_key = f"chembl_{chembl_id}"

    data = _cached_get(url, cache_key)
    if not data:
        return None

    props = data.get("molecule_properties", {}) or {}

    result = {
        "chembl_id": chembl_id,
        "pref_name": data.get("pref_name", ""),
        "molecule_type": data.get("molecule_type", ""),
        "max_phase": data.get("max_phase"),
        "molecular_weight": props.get("full_mwt"),
        "alogp": props.get("alogp"),
        "psa": props.get("psa"),
        "hba": props.get("hba"),
        "hbd": props.get("hbd"),
        "num_ro5_violations": props.get("num_ro5_violations"),
        "qed_weighted": props.get("qed_weighted"),
        "aromatic_rings": props.get("aromatic_rings"),
        "heavy_atoms": props.get("heavy_atoms"),
        "molecular_formula": data.get("molecule_structures", {}).get("standard_inchi", "")
        if data.get("molecule_structures") else "",
    }

    return result


# ---------------------------------------------------------------------------
# NCBI Organism Data
# ---------------------------------------------------------------------------

def get_organism_data(taxid: int) -> Optional[Dict[str, Any]]:
    """Fetch organism genome data from NCBI Datasets v2 API.

    Args:
        taxid: NCBI taxonomy ID (e.g., 9606 for Homo sapiens).

    Returns:
        Dict with genome_size, gc_content, chromosome_count, gene_count,
        protein_coding_genes, etc., or None on failure.
    """
    cache_key = f"ncbi_organism_{taxid}"
    _ensure_cache_dir()

    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(BIO_CACHE_DIR, f"{safe_key}.json")

    # Check cache
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 168:
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                if cached and cached.get("genome_size"):
                    return cached
            except (json.JSONDecodeError, IOError):
                pass

    base_url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{taxid}/dataset_report"
    params_strict = {
        "filters.assembly_source": "refseq",
        "filters.assembly_level": "complete_genome,chromosome",
        "page_size": 1,
    }
    params_relaxed = {
        "filters.assembly_source": "refseq",
        "page_size": 1,
    }

    global _last_request_time
    hdrs = {**_HEADERS, "Accept": "application/json"}

    for attempt_params in [params_strict, params_relaxed]:
        elapsed = time.time() - _last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)

        try:
            resp = requests.get(base_url, params=attempt_params,
                                headers=hdrs, timeout=30)
            _last_request_time = time.time()

            if resp.status_code == 429:
                print(f"Rate limited on NCBI taxid={taxid}, sleeping 5s...", flush=True)
                time.sleep(5)
                resp = requests.get(base_url, params=attempt_params,
                                    headers=hdrs, timeout=30)
                _last_request_time = time.time()

            if resp.status_code != 200:
                continue

            data = resp.json()
            reports = data.get("reports", [])
            if not reports:
                continue

            report = reports[0]
            assembly = report.get("assembly_info", {})
            assembly_stats = report.get("assembly_stats", {})
            annotation = report.get("annotation_info", {})
            organism_info = report.get("organism", {})

            genome_size = assembly_stats.get("total_sequence_length")
            gc_content = assembly_stats.get("gc_percent")

            if genome_size is None:
                continue

            result = {
                "genome_size": int(genome_size) if genome_size else None,
                "gc_content": float(gc_content) if gc_content else None,
                "chromosome_count": assembly_stats.get("total_number_of_chromosomes"),
                "scaffold_count": assembly_stats.get("number_of_scaffolds"),
                "gene_count": (annotation.get("stats", {}).get("gene_counts", {})
                               .get("total")),
                "protein_coding_genes": (annotation.get("stats", {})
                                         .get("gene_counts", {})
                                         .get("protein_coding")),
                "assembly_level": assembly.get("assembly_level"),
                "assembly_accession": assembly.get("assembly_accession"),
                "organism_name": organism_info.get("organism_name", ""),
                "taxid": taxid,
            }

            # Cache the result
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)

            return result

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            print(f"Request error for NCBI taxid={taxid}: {e}", flush=True)
        except Exception as e:
            print(f"Unexpected error for NCBI taxid={taxid}: {e}", flush=True)

    return None


# ---------------------------------------------------------------------------
# Wikipedia Entity Clue Fetching
# ---------------------------------------------------------------------------

def fetch_entity_clues(entity_name: str, entity_type: str = "general") -> List[dict]:
    """Fetch descriptive clue facts from Wikipedia for a biochemical entity.

    Extracts non-quantitative facts such as:
    - Discovery/history
    - Biological function/role
    - Classification/taxonomy
    - Medical/industrial applications
    - Notable characteristics

    Args:
        entity_name: Name of the entity (e.g., "Insulin", "Aspirin").
        entity_type: One of "protein", "compound", "organism", "general".

    Returns:
        List of fact dicts: {"fact_id", "fact", "topic", "source"}.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": entity_name,
        "srlimit": 3,
        "utf8": 1,
    }
    headers = {
        "User-Agent": "DrBencher/1.0 (research@example.com)",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        search_data = resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []
    except Exception:
        return []

    # Fetch top article extract
    page_title = search_results[0].get("title", "")
    extract_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "utf8": 1,
    }

    try:
        resp = requests.get(extract_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        pages = resp.json().get("query", {}).get("pages", {})
        extract_text = ""
        for page in pages.values():
            extract_text = page.get("extract", "")
            break
    except Exception:
        return []

    if not extract_text:
        return []

    # Parse sentences into clue facts
    sentences = re.split(r'(?<=[.!?])\s+', extract_text.strip())
    clues = []

    # Topic classification keywords per entity type
    topic_keywords = {
        "protein": {
            "function": ["function", "catalyze", "regulate", "signal", "bind", "encode", "express"],
            "disease": ["disease", "mutation", "deficiency", "cancer", "disorder", "syndrome"],
            "structure": ["structure", "domain", "fold", "helix", "sheet", "subunit"],
            "discovery": ["discovered", "identified", "first", "isolated", "purified", "named"],
            "location": ["membrane", "cytoplasm", "nucleus", "mitochondria", "cell"],
        },
        "compound": {
            "medical": ["drug", "treat", "therapy", "medication", "prescription", "pharmaceutical"],
            "chemistry": ["chemical", "compound", "molecule", "formula", "synthesis"],
            "mechanism": ["mechanism", "inhibit", "receptor", "agonist", "antagonist", "bind"],
            "discovery": ["discovered", "synthesized", "developed", "introduced", "patented"],
            "side_effects": ["side effect", "adverse", "toxicity", "overdose", "contraindication"],
        },
        "organism": {
            "taxonomy": ["kingdom", "phylum", "class", "order", "family", "genus", "species"],
            "habitat": ["habitat", "found in", "native to", "distributed", "environment"],
            "biology": ["genome", "chromosome", "gene", "DNA", "RNA", "protein"],
            "model": ["model organism", "research", "laboratory", "studied", "experiment"],
            "pathogen": ["pathogen", "infect", "disease", "virulence", "transmission"],
        },
    }

    keywords = topic_keywords.get(entity_type, {})
    default_topics = ["description", "background", "characteristics", "history", "applications"]

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if len(clues) >= 10:
            break

        # Assign a topic based on keywords
        sent_lower = sent.lower()
        topic = default_topics[min(i, len(default_topics) - 1)]
        for topic_name, words in keywords.items():
            if any(w in sent_lower for w in words):
                topic = topic_name
                break

        clues.append({
            "fact_id": f"W{i + 1}",
            "fact": sent,
            "topic": topic,
            "source": f"Wikipedia:{page_title}",
        })

    return clues


# ---------------------------------------------------------------------------
# Category Helpers
# ---------------------------------------------------------------------------

def get_category_entities(category: str) -> List[Dict[str, Any]]:
    """Return list of entities in the entity universe for a category."""
    return ENTITY_UNIVERSE.get(category, [])


def resolve_biochem_wikidata_id(
    name: str,
    entity_type: str = "compound",
    entity_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve a biochemical entity name to a Wikidata QID.

    Resolution strategy (in order):
    1. **SPARQL-first**: use structured identifiers when available:
       - Organisms: if *entity_id* starts with ``Q``, return directly.
       - Compounds: SPARQL ``P662`` (PubChem CID) lookup.
       - Proteins: SPARQL ``P352`` (UniProt accession) lookup.
    2. **Text search + P31 validation**: Wikidata search API fallback.
    3. Return ``None`` if no validated match found (no top-result fallback).

    Args:
        name: Entity name (e.g., "Insulin", "Aspirin").
        entity_type: One of "protein", "compound", "organism".
        entity_id: Structured identifier (UniProt accession, ``CID:12345``,
            or Wikidata QID for organisms).

    Returns:
        Wikidata QID string, or None if not found.
    """
    global _last_request_time
    from .tool_util import search_wikidata_entities, get_entity_data

    cache_key = f"{entity_type}:{entity_id or name}"
    if cache_key in _biochem_wikidata_cache:
        return _biochem_wikidata_cache[cache_key]

    p31_valid = {
        "protein": {"Q8054", "Q7187", "Q417841", "Q21174627"},
        "compound": {"Q11173", "Q12140", "Q79529", "Q2393187"},
        "organism": {"Q16521", "Q7239", "Q55983715"},
    }

    # ------------------------------------------------------------------
    # Strategy 1: SPARQL-first lookup using structured identifier
    # ------------------------------------------------------------------
    if entity_id:
        # Organisms: entity_id is already a QID
        if entity_id.startswith("Q"):
            _biochem_wikidata_cache[cache_key] = entity_id
            return entity_id

        sparql_url = "https://query.wikidata.org/sparql"
        sparql_headers = {
            "User-Agent": "DrBencher research@example.com",
            "Accept": "application/json",
        }

        query = None
        if entity_type == "compound" and "CID:" in entity_id.upper():
            cid_num = entity_id.upper().replace("CID:", "").strip()
            query = f'SELECT ?item WHERE {{ ?item wdt:P662 "{cid_num}" . }} LIMIT 5'
        elif entity_type == "protein":
            accession = entity_id.strip()
            query = f'SELECT ?item WHERE {{ ?item wdt:P352 "{accession}" . }} LIMIT 5'

        if query:
            elapsed = time.time() - _last_request_time
            if elapsed < _RATE_LIMIT_DELAY:
                time.sleep(_RATE_LIMIT_DELAY - elapsed)
            try:
                resp = requests.get(
                    sparql_url, params={"query": query},
                    headers=sparql_headers, timeout=15,
                )
                _last_request_time = time.time()
                if resp.status_code == 200:
                    bindings = resp.json().get("results", {}).get("bindings", [])
                    for b in bindings:
                        uri = b.get("item", {}).get("value", "")
                        qid = uri.split("/")[-1] if "/" in uri else ""
                        if qid.startswith("Q"):
                            _biochem_wikidata_cache[cache_key] = qid
                            return qid
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Strategy 2: Text search + P31 validation (fallback)
    # ------------------------------------------------------------------
    results = search_wikidata_entities(name, limit=5)
    if not results:
        _biochem_wikidata_cache[cache_key] = None
        return None

    valid_qids = p31_valid.get(entity_type, set())

    for r in results:
        qid = r.get("id", "")
        if not qid:
            continue
        if not valid_qids:
            _biochem_wikidata_cache[cache_key] = qid
            return qid

        try:
            entity_data = get_entity_data(qid)
            if entity_data:
                claims = entity_data.get("claims", {})
                p31_values = set()
                for claim in claims.get("P31", []):
                    if isinstance(claim, dict):
                        # get_entity_data returns {'type':'entity','id':'Q...','label':'...'}
                        p31_values.add(claim.get("id", claim.get("value", "")))
                    elif isinstance(claim, str):
                        p31_values.add(claim)
                if p31_values & valid_qids:
                    _biochem_wikidata_cache[cache_key] = qid
                    return qid
        except Exception:
            pass

    # No validated match — return None (no top-result fallback)
    _biochem_wikidata_cache[cache_key] = None
    return None
