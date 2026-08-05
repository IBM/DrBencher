# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Historical/Causal API utilities for history benchmark.

Provides functions to fetch temporal data from Wikidata (birth/death dates,
founding dates, conflict start/end), with local JSON caching, rate limiting,
and Wikipedia entity clue extraction.

Data source: Wikidata REST API (via tool_util.get_entity_data)
"""

import json
import os
import random
import re
import time
import requests
from typing import Any, Dict, List, Optional

from .tool_util import get_entity_data, search_wikidata_entities

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_CACHE_DIR = "cache/history_cache"
_WIKIDATA_RATE_LIMIT_DELAY = 0.5
_last_wikidata_request_time = 0.0

_WIKI_HEADERS = {
    "User-Agent": "DrBencher/1.0 (research@example.com)",
    "Accept": "application/json",
}

# Temporal Wikidata properties
TEMPORAL_PROPERTIES = {
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception",
    "P576": "dissolved, abolished or demolished date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
    "P577": "publication date",
    "P1619": "date of official opening",
}


# ---------------------------------------------------------------------------
# Entity Universe (~350 entities across 16 categories)
# ---------------------------------------------------------------------------

ENTITY_UNIVERSE: Dict[str, List[Dict[str, Any]]] = {
    "empires": [
        {"qid": "Q2277", "name": "Roman Empire", "wiki": "Roman Empire"},
        {"qid": "Q12560", "name": "Ottoman Empire", "wiki": "Ottoman Empire"},
        {"qid": "Q12544", "name": "Byzantine Empire", "wiki": "Byzantine Empire"},
        {"qid": "Q12557", "name": "Mongol Empire", "wiki": "Mongol Empire"},
        {"qid": "Q8680", "name": "British Empire", "wiki": "British Empire"},
        {"qid": "Q217230", "name": "Empire of Brazil", "wiki": "Empire of Brazil"},
        {"qid": "Q33296", "name": "Mughal Empire", "wiki": "Mughal Empire"},
        {"qid": "Q389688", "name": "Achaemenid Empire", "wiki": "Achaemenid Empire"},
        {"qid": "Q28513", "name": "Austro-Hungarian Empire", "wiki": "Austria-Hungary"},
        {"qid": "Q28573", "name": "Inca Empire", "wiki": "Inca Empire"},
        {"qid": "Q80702", "name": "Spanish Empire", "wiki": "Spanish Empire"},
        {"qid": "Q71084", "name": "First French Empire", "wiki": "First French Empire"},
        {"qid": "Q8575586", "name": "Umayyad Caliphate", "wiki": "Umayyad Caliphate"},
        {"qid": "Q12536", "name": "Abbasid Caliphate", "wiki": "Abbasid Caliphate"},
        {"qid": "Q12548", "name": "Holy Roman Empire", "wiki": "Holy Roman Empire"},
        {"qid": "Q34266", "name": "Russian Empire", "wiki": "Russian Empire"},
        {"qid": "Q7462", "name": "Song dynasty", "wiki": "Song dynasty"},
        {"qid": "Q83891", "name": "Sassanid Empire", "wiki": "Sasanian Empire"},
        {"qid": "Q2608489", "name": "Aztec Empire", "wiki": "Aztec Empire"},
        {"qid": "Q201705", "name": "Khmer Empire", "wiki": "Khmer Empire"},
    ],
    "assassinations": [
        {"qid": "Q193484", "name": "Assassination of JFK", "wiki": "Assassination of John F. Kennedy"},
        {"qid": "Q1025404", "name": "Assassination of Lincoln", "wiki": "Assassination of Abraham Lincoln"},
        {"qid": "Q192050", "name": "Assassination of Archduke Franz Ferdinand", "wiki": "Assassination of Archduke Franz Ferdinand"},
        {"qid": "Q757963", "name": "Assassination of MLK", "wiki": "Assassination of Martin Luther King Jr."},
        {"qid": "Q3350154", "name": "Assassination of Mahatma Gandhi", "wiki": "Assassination of Mahatma Gandhi"},
        {"qid": "Q1025466", "name": "Assassination of Julius Caesar", "wiki": "Assassination of Julius Caesar"},
        {"qid": "Q1187550", "name": "Assassination of RFK", "wiki": "Assassination of Robert F. Kennedy"},
        {"qid": "Q2756743", "name": "Assassination of Anwar Sadat", "wiki": "Assassination of Anwar Sadat"},
        {"qid": "Q3347903", "name": "Assassination of Indira Gandhi", "wiki": "Assassination of Indira Gandhi"},
        {"qid": "Q3423445", "name": "Assassination of Rajiv Gandhi", "wiki": "Assassination of Rajiv Gandhi"},
        {"qid": "Q2608162", "name": "Assassination of Yitzhak Rabin", "wiki": "Assassination of Yitzhak Rabin"},
        {"qid": "Q1784968", "name": "Assassination of Benazir Bhutto", "wiki": "Assassination of Benazir Bhutto"},
        {"qid": "Q3284177", "name": "Assassination of Olof Palme", "wiki": "Assassination of Olof Palme"},
        {"qid": "Q4468600", "name": "Assassination of Leon Trotsky", "wiki": "Assassination of Leon Trotsky"},
        {"qid": "Q2866985", "name": "Assassination of William McKinley", "wiki": "Assassination of William McKinley"},
        {"qid": "Q482859", "name": "Assassination of Park Chung-hee", "wiki": "Assassination of Park Chung-hee"},
        {"qid": "Q2866972", "name": "Assassination of James A. Garfield", "wiki": "Assassination of James A. Garfield"},
        {"qid": "Q1634609", "name": "Assassination of Aldo Moro", "wiki": "Kidnapping and murder of Aldo Moro"},
        {"qid": "Q2529018", "name": "Assassination of Patrice Lumumba", "wiki": "Assassination of Patrice Lumumba"},
        {"qid": "Q112967003", "name": "Assassination of Shinzo Abe", "wiki": "Assassination of Shinzo Abe"},
    ],
    "sieges": [
        {"qid": "Q160077", "name": "Siege of Constantinople (1453)", "wiki": "Fall of Constantinople"},
        {"qid": "Q151860", "name": "Siege of Leningrad", "wiki": "Siege of Leningrad"},
        {"qid": "Q3555020", "name": "Siege of Masada", "wiki": "Siege of Masada"},
        {"qid": "Q200855", "name": "Siege of Vienna (1683)", "wiki": "Battle of Vienna"},
        {"qid": "Q38789", "name": "Siege of Stalingrad", "wiki": "Battle of Stalingrad"},
        {"qid": "Q3486045", "name": "Siege of Troy", "wiki": "Trojan War"},
        {"qid": "Q844886", "name": "Siege of Jerusalem (70 AD)", "wiki": "Siege of Jerusalem (70 CE)"},
        {"qid": "Q392213", "name": "Siege of Orleans", "wiki": "Siege of Orl%C3%A9ans"},
        {"qid": "Q235344", "name": "Battle of the Alamo", "wiki": "Battle of the Alamo"},
        {"qid": "Q1066253", "name": "Siege of Vicksburg", "wiki": "Siege of Vicksburg"},
        {"qid": "Q459447", "name": "Battle of Yorktown", "wiki": "Siege of Yorktown"},
        {"qid": "Q604897", "name": "Battle of Dien Bien Phu", "wiki": "Battle of Dien Bien Phu"},
        {"qid": "Q154860", "name": "Siege of Sevastopol (1854)", "wiki": "Siege of Sevastopol (1854%E2%80%931855)"},
        {"qid": "Q130847", "name": "Battle of Verdun", "wiki": "Battle of Verdun"},
        {"qid": "Q690489", "name": "Siege of Paris (1870)", "wiki": "Siege of Paris (1870%E2%80%931871)"},
        {"qid": "Q164983", "name": "Gallipoli Campaign", "wiki": "Gallipoli campaign"},
        {"qid": "Q154182", "name": "Battle of Berlin", "wiki": "Battle of Berlin"},
        {"qid": "Q180182", "name": "Battle of Iwo Jima", "wiki": "Battle of Iwo Jima"},
        {"qid": "Q33132", "name": "Battle of Gettysburg", "wiki": "Battle of Gettysburg"},
        {"qid": "Q48314", "name": "Battle of Waterloo", "wiki": "Battle of Waterloo"},
    ],
    "religious_events": [
        {"qid": "Q51649", "name": "First Crusade", "wiki": "First Crusade"},
        {"qid": "Q51655", "name": "Third Crusade", "wiki": "Third Crusade"},
        {"qid": "Q12562", "name": "Protestant Reformation", "wiki": "Reformation"},
        {"qid": "Q51648", "name": "Great Schism (1054)", "wiki": "East%E2%80%93West Schism"},
        {"qid": "Q172991", "name": "Council of Trent", "wiki": "Council of Trent"},
        {"qid": "Q232572", "name": "Council of Nicaea", "wiki": "First Council of Nicaea"},
        {"qid": "Q536822", "name": "Diet of Worms", "wiki": "Diet of Worms"},
        {"qid": "Q184725", "name": "Spanish Inquisition", "wiki": "Spanish Inquisition"},
        {"qid": "Q169789", "name": "Second Vatican Council", "wiki": "Second Vatican Council"},
        {"qid": "Q219698", "name": "Salem witch trials", "wiki": "Salem witch trials"},
        {"qid": "Q202558", "name": "Avignon Papacy", "wiki": "Avignon Papacy"},
        {"qid": "Q759837", "name": "Dissolution of the Monasteries", "wiki": "Dissolution of the Monasteries"},
        {"qid": "Q51654", "name": "Second Crusade", "wiki": "Second Crusade"},
        {"qid": "Q179788", "name": "Edict of Nantes", "wiki": "Edict of Nantes"},
        {"qid": "Q150995", "name": "Peace of Westphalia", "wiki": "Peace of Westphalia"},
        {"qid": "Q234915", "name": "Hajj", "wiki": "Hajj"},
        {"qid": "Q2487", "name": "Thirty Years' War", "wiki": "Thirty Years%27 War"},
        {"qid": "Q51656", "name": "Fourth Crusade", "wiki": "Fourth Crusade"},
        {"qid": "Q1645505", "name": "English Reformation", "wiki": "English Reformation"},
        {"qid": "Q192408", "name": "Taiping Rebellion", "wiki": "Taiping Rebellion"},
    ],
    "coups": [
        {"qid": "Q593774", "name": "1953 Iranian coup", "wiki": "1953 Iranian coup d%27%C3%A9tat"},
        {"qid": "Q856670", "name": "1973 Chilean coup", "wiki": "1973 Chilean coup d%27%C3%A9tat"},
        {"qid": "Q1780431", "name": "1952 Egyptian revolution", "wiki": "Egyptian revolution of 1952"},
        {"qid": "Q8707", "name": "Meiji Restoration", "wiki": "Meiji Restoration"},
        {"qid": "Q620965", "name": "18 Brumaire (Napoleon)", "wiki": "18 Brumaire"},
        {"qid": "Q221382", "name": "1991 Soviet coup attempt", "wiki": "1991 Soviet coup d%27%C3%A9tat attempt"},
        {"qid": "Q1859259", "name": "1960 Turkish coup", "wiki": "1960 Turkish coup d%27%C3%A9tat"},
        {"qid": "Q107187684", "name": "1964 Brazilian coup", "wiki": "1964 Brazilian coup d%27%C3%A9tat"},
        {"qid": "Q36749", "name": "Beer Hall Putsch", "wiki": "Beer Hall Putsch"},
        {"qid": "Q193245", "name": "Carnation Revolution", "wiki": "Carnation Revolution"},
        {"qid": "Q189508", "name": "Glorious Revolution", "wiki": "Glorious Revolution"},
        {"qid": "Q126065", "name": "1979 Iranian Revolution", "wiki": "Iranian Revolution"},
        {"qid": "Q129053", "name": "Partition of India", "wiki": "Partition of India"},
        {"qid": "Q69163529", "name": "Fall of the Berlin Wall", "wiki": "Fall of the Berlin Wall"},
        {"qid": "Q5167679", "name": "Dissolution of the Soviet Union", "wiki": "Dissolution of the Soviet Union"},
        {"qid": "Q154705", "name": "Unification of Germany", "wiki": "Unification of Germany"},
        {"qid": "Q6107218", "name": "1969 Libyan coup", "wiki": "1969 Libyan coup d%27%C3%A9tat"},
        {"qid": "Q16205928", "name": "1966 Nigerian coup", "wiki": "1966 Nigerian coup d%27%C3%A9tat"},
        {"qid": "Q182817", "name": "Velvet Revolution", "wiki": "Velvet Revolution"},
        {"qid": "Q99717", "name": "Tiananmen Square protests", "wiki": "1989 Tiananmen Square protests and massacre"},
    ],
    "migrations": [
        {"qid": "Q493814", "name": "Irish Famine emigration", "wiki": "Great Famine (Ireland)"},
        {"qid": "Q10701282", "name": "Atlantic slave trade", "wiki": "Atlantic slave trade"},
        {"qid": "Q750448", "name": "Trail of Tears", "wiki": "Trail of Tears"},
        {"qid": "Q1506365", "name": "Great Migration (US)", "wiki": "Great Migration (African American)"},
        {"qid": "Q129053", "name": "Partition of India", "wiki": "Partition of India"},
        {"qid": "Q17550", "name": "Gold Rush (California)", "wiki": "California Gold Rush"},
        {"qid": "Q276172", "name": "Jewish exodus from Arab lands", "wiki": "Jewish exodus from the Muslim world"},
        {"qid": "Q131192", "name": "Migration Period (Europe)", "wiki": "Migration Period"},
        {"qid": "Q726501", "name": "Dust Bowl", "wiki": "Dust Bowl"},
        {"qid": "Q130251", "name": "Bantu expansion", "wiki": "Bantu expansion"},
        {"qid": "Q179848", "name": "Scramble for Africa", "wiki": "Scramble for Africa"},
        {"qid": "Q41967248", "name": "Mayflower voyage", "wiki": "Mayflower"},
        {"qid": "Q80034", "name": "Armenian genocide", "wiki": "Armenian genocide"},
        {"qid": "Q2763", "name": "The Holocaust", "wiki": "The Holocaust"},
        {"qid": "Q46333", "name": "Long March (China)", "wiki": "Long March"},
        {"qid": "Q6449297", "name": "Vietnamese boat people", "wiki": "Vietnamese boat people"},
        {"qid": "Q131297", "name": "Rwandan genocide", "wiki": "Rwandan genocide"},
        {"qid": "Q3266633", "name": "Nakba", "wiki": "1948 Palestinian exodus"},
        {"qid": "Q1061030", "name": "Expulsion of Jews from Spain", "wiki": "Alhambra Decree"},
        {"qid": "Q165058", "name": "Holodomor", "wiki": "Holodomor"},
    ],
    "constitutions": [
        {"qid": "Q11698", "name": "US Constitution", "wiki": "Constitution of the United States"},
        {"qid": "Q12519", "name": "Magna Carta", "wiki": "Magna Carta"},
        {"qid": "Q93304", "name": "Code of Hammurabi", "wiki": "Code of Hammurabi"},
        {"qid": "Q169759", "name": "French Declaration of Rights", "wiki": "Declaration of the Rights of Man and of the Citizen"},
        {"qid": "Q219447", "name": "English Bill of Rights", "wiki": "Bill of Rights 1689"},
        {"qid": "Q7813", "name": "Universal Declaration of Human Rights", "wiki": "Universal Declaration of Human Rights"},
        {"qid": "Q391358", "name": "Emancipation Proclamation", "wiki": "Emancipation Proclamation"},
        {"qid": "Q151060", "name": "Napoleon Code", "wiki": "Napoleonic Code"},
        {"qid": "Q156003", "name": "Weimar Constitution", "wiki": "Weimar Constitution"},
        {"qid": "Q52843", "name": "Treaty of Lisbon", "wiki": "Treaty of Lisbon"},
        {"qid": "Q493620", "name": "Articles of Confederation", "wiki": "Articles of Confederation"},
        {"qid": "Q203686", "name": "Twelve Tables (Roman law)", "wiki": "Twelve Tables"},
        {"qid": "Q171328", "name": "Charter of the United Nations", "wiki": "Charter of the United Nations"},
        {"qid": "Q584063", "name": "Petition of Right", "wiki": "Petition of Right"},
        {"qid": "Q237082", "name": "Japanese Constitution (1947)", "wiki": "Constitution of Japan"},
        {"qid": "Q858036", "name": "Federalist Papers", "wiki": "The Federalist Papers"},
        {"qid": "Q477108", "name": "Indian Constitution", "wiki": "Constitution of India"},
        {"qid": "Q181026", "name": "Monroe Doctrine", "wiki": "Monroe Doctrine"},
        {"qid": "Q4576", "name": "Marshall Plan", "wiki": "Marshall Plan"},
        {"qid": "Q133536", "name": "Geneva Conventions", "wiki": "Geneva Conventions"},
    ],
    "independence_movements": [
        {"qid": "Q40949", "name": "American Revolution", "wiki": "American Revolution"},
        {"qid": "Q12444025", "name": "Indian independence movement", "wiki": "Indian independence movement"},
        {"qid": "Q689128", "name": "Haitian Revolution", "wiki": "Haitian Revolution"},
        {"qid": "Q208297", "name": "Irish War of Independence", "wiki": "Irish War of Independence"},
        {"qid": "Q200790", "name": "Algerian War", "wiki": "Algerian War"},
        {"qid": "Q1332160", "name": "Indonesian National Revolution", "wiki": "Indonesian National Revolution"},
        {"qid": "Q11264", "name": "Cuban Revolution", "wiki": "Cuban Revolution"},
        {"qid": "Q8740", "name": "Vietnam War", "wiki": "Vietnam War"},
        {"qid": "Q182062", "name": "Greek War of Independence", "wiki": "Greek War of Independence"},
        {"qid": "Q4583158", "name": "South African anti-apartheid", "wiki": "Internal resistance to apartheid"},
        {"qid": "Q371394", "name": "Bangladesh Liberation War", "wiki": "Bangladesh Liberation War"},
        {"qid": "Q476855", "name": "Mau Mau Uprising", "wiki": "Mau Mau uprising"},
        {"qid": "Q1003", "name": "Solidarity movement (Poland)", "wiki": "Solidarity (Polish trade union)"},
        {"qid": "Q1123201", "name": "Bolivarian independence", "wiki": "Spanish American wars of independence"},
        {"qid": "Q638530", "name": "Texas Revolution", "wiki": "Texas Revolution"},
        {"qid": "Q42388", "name": "Zionist movement", "wiki": "Zionism"},
        {"qid": "Q1992677", "name": "Quit India Movement", "wiki": "Quit India movement"},
        {"qid": "Q239344", "name": "Salt March", "wiki": "Salt March"},
        {"qid": "Q193689", "name": "Easter Rising", "wiki": "Easter Rising"},
        {"qid": "Q422082", "name": "Philippine Revolution", "wiki": "Philippine Revolution"},
    ],
    "civil_wars": [
        {"qid": "Q8676", "name": "American Civil War", "wiki": "American Civil War"},
        {"qid": "Q10859", "name": "Spanish Civil War", "wiki": "Spanish Civil War"},
        {"qid": "Q80330", "name": "English Civil War", "wiki": "English Civil War"},
        {"qid": "Q179975", "name": "Chinese Civil War", "wiki": "Chinese Civil War"},
        {"qid": "Q79911", "name": "Russian Civil War", "wiki": "Russian Civil War"},
        {"qid": "Q208484", "name": "Lebanese Civil War", "wiki": "Lebanese Civil War"},
        {"qid": "Q243620", "name": "Somali Civil War", "wiki": "Somali Civil War"},
        {"qid": "Q426722", "name": "Rwandan Civil War", "wiki": "Rwandan Civil War"},
        {"qid": "Q178810", "name": "Syrian Civil War", "wiki": "Syrian civil war"},
        {"qid": "Q12055176", "name": "Angolan Civil War", "wiki": "Angolan Civil War"},
        {"qid": "Q181533", "name": "Bosnian War", "wiki": "Bosnian War"},
        {"qid": "Q242352", "name": "Yugoslav Wars", "wiki": "Yugoslav Wars"},
        {"qid": "Q829875", "name": "Nigerian Civil War", "wiki": "Nigerian Civil War"},
        {"qid": "Q1783607", "name": "Salvadoran Civil War", "wiki": "Salvadoran Civil War"},
        {"qid": "Q657661", "name": "Mozambican Civil War", "wiki": "Mozambican Civil War"},
        {"qid": "Q127751", "name": "Wars of the Roses", "wiki": "Wars of the Roses"},
        {"qid": "Q211855", "name": "Finnish Civil War", "wiki": "Finnish Civil War"},
        {"qid": "Q188972", "name": "Greek Civil War", "wiki": "Greek Civil War"},
        {"qid": "Q817206", "name": "Cambodian Civil War", "wiki": "Cambodian Civil War"},
        {"qid": "Q213394", "name": "Sri Lankan Civil War", "wiki": "Sri Lankan Civil War"},
    ],
    "genocides_atrocities": [
        {"qid": "Q2763", "name": "The Holocaust", "wiki": "The Holocaust"},
        {"qid": "Q80034", "name": "Armenian genocide", "wiki": "Armenian genocide"},
        {"qid": "Q131297", "name": "Rwandan genocide", "wiki": "Rwandan genocide"},
        {"qid": "Q2885072", "name": "Cambodian genocide", "wiki": "Cambodian genocide"},
        {"qid": "Q165058", "name": "Holodomor", "wiki": "Holodomor"},
        {"qid": "Q192055", "name": "Nanjing Massacre", "wiki": "Nanjing Massacre"},
        {"qid": "Q170334", "name": "Srebrenica massacre", "wiki": "Srebrenica massacre"},
        {"qid": "Q3266633", "name": "Nakba (1948)", "wiki": "1948 Palestinian exodus"},
        {"qid": "Q3288108", "name": "Bosnian genocide", "wiki": "Bosnian genocide"},
        {"qid": "Q134301", "name": "Katyn massacre", "wiki": "Katyn massacre"},
        {"qid": "Q183421", "name": "My Lai massacre", "wiki": "M%E1%BB%B9 Lai massacre"},
        {"qid": "Q190758", "name": "Darfur genocide", "wiki": "War in Darfur"},
        {"qid": "Q108413", "name": "Wounded Knee Massacre", "wiki": "Wounded Knee Massacre"},
        {"qid": "Q208855", "name": "Jallianwala Bagh massacre", "wiki": "Jallianwala Bagh massacre"},
        {"qid": "Q518753", "name": "Sharpeville massacre", "wiki": "Sharpeville massacre"},
        {"qid": "Q799299", "name": "Indonesian mass killings (1965)", "wiki": "Indonesian mass killings of 1965%E2%80%9366"},
        {"qid": "Q28136551", "name": "Rohingya genocide", "wiki": "Rohingya genocide"},
        {"qid": "Q378835", "name": "Unit 731", "wiki": "Unit 731"},
        {"qid": "Q2578778", "name": "Circassian genocide", "wiki": "Circassian genocide"},
        {"qid": "Q312492", "name": "Herero and Namaqua genocide", "wiki": "Herero and Nama genocide"},
    ],
    "economic_crises": [
        {"qid": "Q8698", "name": "Great Depression", "wiki": "Great Depression"},
        {"qid": "Q896666", "name": "2008 financial crisis", "wiki": "Financial crisis of 2007%E2%80%932008"},
        {"qid": "Q219217", "name": "Tulip mania", "wiki": "Tulip mania"},
        {"qid": "Q18643921", "name": "South Sea Bubble", "wiki": "South Sea Company"},
        {"qid": "Q80880", "name": "1997 Asian financial crisis", "wiki": "1997 Asian financial crisis"},
        {"qid": "Q316817", "name": "1973 oil crisis", "wiki": "1973 oil crisis"},
        {"qid": "Q868261", "name": "Black Monday (1987)", "wiki": "Black Monday (1987)"},
        {"qid": "Q79721", "name": "Dot-com bubble", "wiki": "Dot-com bubble"},
        {"qid": "Q844449", "name": "Panic of 1907", "wiki": "Panic of 1907"},
        {"qid": "Q1417847", "name": "Long Depression (1873)", "wiki": "Long Depression"},
        {"qid": "Q217197", "name": "European debt crisis", "wiki": "European debt crisis"},
        {"qid": "Q645426", "name": "Argentine economic crisis (2001)", "wiki": "1998%E2%80%932002 Argentine great depression"},
        {"qid": "Q201684", "name": "Wall Street Crash (1929)", "wiki": "Wall Street Crash of 1929"},
        {"qid": "Q1142572", "name": "1998 Russian financial crisis", "wiki": "1998 Russian financial crisis"},
        {"qid": "Q599806", "name": "Mexican peso crisis (1994)", "wiki": "Mexican peso crisis"},
        {"qid": "Q1326489", "name": "Savings and loan crisis", "wiki": "Savings and loan crisis"},
        {"qid": "Q695566", "name": "Panic of 1873", "wiki": "Panic of 1873"},
        {"qid": "Q6500827", "name": "Japanese asset price bubble", "wiki": "Japanese asset price bubble"},
        {"qid": "Q88599208", "name": "COVID-19 recession", "wiki": "COVID-19 recession"},
        {"qid": "Q1192494", "name": "Mississippi Company bubble", "wiki": "Mississippi Company"},
    ],
    "naval_battles": [
        {"qid": "Q171416", "name": "Battle of Trafalgar", "wiki": "Battle of Trafalgar"},
        {"qid": "Q173034", "name": "Battle of Midway", "wiki": "Battle of Midway"},
        {"qid": "Q165425", "name": "Battle of Lepanto", "wiki": "Battle of Lepanto"},
        {"qid": "Q52418", "name": "Attack on Pearl Harbor", "wiki": "Attack on Pearl Harbor"},
        {"qid": "Q676404", "name": "Spanish Armada", "wiki": "Spanish Armada"},
        {"qid": "Q156554", "name": "Battle of Jutland", "wiki": "Battle of Jutland"},
        {"qid": "Q178850", "name": "Battle of Salamis", "wiki": "Battle of Salamis"},
        {"qid": "Q207165", "name": "Battle of the Coral Sea", "wiki": "Battle of the Coral Sea"},
        {"qid": "Q208127", "name": "Battle of Tsushima", "wiki": "Battle of Tsushima"},
        {"qid": "Q138456543", "name": "Battle of the Nile", "wiki": "Battle of the Nile"},
        {"qid": "Q160387", "name": "Battle of Actium", "wiki": "Battle of Actium"},
        {"qid": "Q16470", "name": "D-Day (Normandy landings)", "wiki": "Normandy landings"},
        {"qid": "Q308999", "name": "Battle of Leyte Gulf", "wiki": "Battle of Leyte Gulf"},
        {"qid": "Q911972", "name": "Dunkirk evacuation", "wiki": "Dunkirk evacuation"},
        {"qid": "Q157627", "name": "Battle of the Atlantic", "wiki": "Battle of the Atlantic"},
        {"qid": "Q217145", "name": "Battle of Guadalcanal", "wiki": "Guadalcanal campaign"},
        {"qid": "Q192660", "name": "Battle of Okinawa", "wiki": "Battle of Okinawa"},
        {"qid": "Q504347", "name": "Battle of the Philippine Sea", "wiki": "Battle of the Philippine Sea"},
        {"qid": "Q6497833", "name": "Sinking of the Bismarck", "wiki": "Last battle of the battleship Bismarck"},
        {"qid": "Q2577588", "name": "RMS Titanic sinking", "wiki": "Sinking of the Titanic"},
    ],
    "peace_accords": [
        {"qid": "Q8736", "name": "Treaty of Versailles", "wiki": "Treaty of Versailles"},
        {"qid": "Q309204", "name": "Camp David Accords", "wiki": "Camp David Accords"},
        {"qid": "Q208958", "name": "Good Friday Agreement", "wiki": "Good Friday Agreement"},
        {"qid": "Q150995", "name": "Peace of Westphalia", "wiki": "Peace of Westphalia"},
        {"qid": "Q46362", "name": "Congress of Vienna", "wiki": "Congress of Vienna"},
        {"qid": "Q1324198", "name": "Treaty of Ghent", "wiki": "Treaty of Ghent"},
        {"qid": "Q4576", "name": "Marshall Plan", "wiki": "Marshall Plan"},
        {"qid": "Q7184", "name": "NATO founding", "wiki": "NATO"},
        {"qid": "Q199820", "name": "Paris Peace Conference (1919)", "wiki": "Paris Peace Conference (1919%E2%80%931920)"},
        {"qid": "Q17013132", "name": "Oslo Accords", "wiki": "Oslo Accords"},
        {"qid": "Q161227", "name": "Yalta Conference", "wiki": "Yalta Conference"},
        {"qid": "Q277476", "name": "Potsdam Conference", "wiki": "Potsdam Conference"},
        {"qid": "Q217450", "name": "Treaty of Paris (1783)", "wiki": "Treaty of Paris (1783)"},
        {"qid": "Q180897", "name": "Treaty of Tordesillas", "wiki": "Treaty of Tordesillas"},
        {"qid": "Q190315", "name": "Dayton Agreement", "wiki": "Dayton Agreement"},
        {"qid": "Q154255", "name": "Munich Agreement", "wiki": "Munich Agreement"},
        {"qid": "Q2862267", "name": "Korean Armistice Agreement", "wiki": "Korean Armistice Agreement"},
        {"qid": "Q392541", "name": "Treaty of San Francisco", "wiki": "Treaty of San Francisco"},
        {"qid": "Q122371", "name": "Treaty of Brest-Litovsk", "wiki": "Treaty of Brest-Litovsk"},
        {"qid": "Q318161", "name": "Helsinki Accords", "wiki": "Helsinki Accords"},
    ],
    "scientific_institutions": [
        {"qid": "Q309751", "name": "NASA", "wiki": "NASA"},
        {"qid": "Q42944", "name": "CERN", "wiki": "CERN"},
        {"qid": "Q123885", "name": "Royal Society", "wiki": "Royal Society"},
        {"qid": "Q49108", "name": "MIT", "wiki": "Massachusetts Institute of Technology"},
        {"qid": "Q13371", "name": "Harvard University", "wiki": "Harvard University"},
        {"qid": "Q158085", "name": "Max Planck Society", "wiki": "Max Planck Society"},
        {"qid": "Q131626", "name": "Smithsonian Institution", "wiki": "Smithsonian Institution"},
        {"qid": "Q127050", "name": "Manhattan Project", "wiki": "Manhattan Project"},
        {"qid": "Q35794", "name": "University of Cambridge", "wiki": "University of Cambridge"},
        {"qid": "Q34433", "name": "University of Oxford", "wiki": "University of Oxford"},
        {"qid": "Q41506", "name": "Stanford University", "wiki": "Stanford University"},
        {"qid": "Q188771", "name": "Academy of Sciences (France)", "wiki": "French Academy of Sciences"},
        {"qid": "Q7817", "name": "WHO", "wiki": "World Health Organization"},
        {"qid": "Q7164", "name": "World Bank", "wiki": "World Bank"},
        {"qid": "Q7804", "name": "International Monetary Fund", "wiki": "International Monetary Fund"},
        {"qid": "Q42262", "name": "European Space Agency", "wiki": "European Space Agency"},
        {"qid": "Q41984", "name": "IAEA", "wiki": "International Atomic Energy Agency"},
        {"qid": "Q21573511", "name": "Harvard Observatory", "wiki": "Harvard College Observatory"},
        {"qid": "Q217365", "name": "Bell Labs", "wiki": "Bell Labs"},
        {"qid": "Q189325", "name": "Jet Propulsion Laboratory", "wiki": "Jet Propulsion Laboratory"},
    ],
    "technological_milestones": [
        {"qid": "Q158075", "name": "Gutenberg printing press", "wiki": "Printing press"},
        {"qid": "Q2269", "name": "Industrial Revolution", "wiki": "Industrial Revolution"},
        {"qid": "Q35820", "name": "First powered flight (Wright brothers)", "wiki": "Wright brothers"},
        {"qid": "Q43653", "name": "Moon landing (Apollo 11)", "wiki": "Apollo 11"},
        {"qid": "Q75", "name": "Internet", "wiki": "Internet"},
        {"qid": "Q80811", "name": "Sputnik launch", "wiki": "Sputnik"},
        {"qid": "Q188770", "name": "First telephone (Bell)", "wiki": "Invention of the telephone"},
        {"qid": "Q207342", "name": "First nuclear test (Trinity)", "wiki": "Trinity (nuclear test)"},
        {"qid": "Q192446", "name": "Completion of Human Genome Project", "wiki": "Human Genome Project"},
        {"qid": "Q20183371", "name": "First railway (Stockton-Darlington)", "wiki": "Stockton and Darlington Railway"},
        {"qid": "Q177", "name": "Pizza", "wiki": "Pizza"},
        {"qid": "Q169399", "name": "First electronic computer (ENIAC)", "wiki": "ENIAC"},
        {"qid": "Q361047", "name": "First transatlantic telegraph", "wiki": "Transatlantic telegraph cable"},
        {"qid": "Q7350", "name": "Panama Canal opening", "wiki": "Panama Canal"},
        {"qid": "Q899", "name": "Suez Canal opening", "wiki": "Suez Canal"},
        {"qid": "Q732410", "name": "First transatlantic flight", "wiki": "Transatlantic flight"},
        {"qid": "Q466", "name": "World Wide Web invention", "wiki": "World Wide Web"},
        {"qid": "Q2766", "name": "First smartphone (iPhone)", "wiki": "IPhone"},
        {"qid": "Q1753108", "name": "Completion of Transcontinental Railroad", "wiki": "First transcontinental railroad"},
        {"qid": "Q2513", "name": "Hubble Space Telescope", "wiki": "Hubble Space Telescope"},
    ],
    "famines": [
        {"qid": "Q188371", "name": "Great Irish Famine", "wiki": "Great Famine (Ireland)"},
        {"qid": "Q522837", "name": "Bengal famine of 1943", "wiki": "Bengal famine of 1943"},
        {"qid": "Q165058", "name": "Holodomor", "wiki": "Holodomor"},
        {"qid": "Q2454958", "name": "Great Chinese Famine", "wiki": "Great Chinese Famine"},
        {"qid": "Q698323", "name": "North Korean famine", "wiki": "North Korean famine"},
        {"qid": "Q1637896", "name": "Ethiopian famine (1983)", "wiki": "1983%E2%80%931985 famine in Ethiopia"},
        {"qid": "Q829875", "name": "Biafran famine", "wiki": "Nigerian Civil War"},
        {"qid": "Q2172846", "name": "Russian famine (1921)", "wiki": "Russian famine of 1921%E2%80%931922"},
        {"qid": "Q2885072", "name": "Cambodian famine", "wiki": "Cambodian famine"},
        {"qid": "Q5957422", "name": "Persian famine of 1917", "wiki": "Persian famine of 1917%E2%80%931919"},
        {"qid": "Q3530500", "name": "Deccan famine (1630)", "wiki": "Deccan famine of 1630%E2%80%931632"},
        {"qid": "Q1558846", "name": "Finnish famine (1866)", "wiki": "Finnish famine of 1866%E2%80%931868"},
        {"qid": "Q1988505", "name": "Great Famine of 1315", "wiki": "Great Famine of 1315%E2%80%931317"},
        {"qid": "Q3534685", "name": "Madras famine (1877)", "wiki": "Great Famine of 1876%E2%80%931878"},
        {"qid": "Q165058", "name": "Soviet famine (1932)", "wiki": "Soviet famine of 1932%E2%80%931933"},
        {"qid": "Q2562594", "name": "Dutch famine (1944)", "wiki": "Dutch famine of 1944%E2%80%931945"},
        {"qid": "Q707698", "name": "Vietnam famine (1945)", "wiki": "Vietnamese famine of 1945"},
        {"qid": "Q223390", "name": "Somali famine (2011)", "wiki": "2011 East Africa drought"},
        {"qid": "Q4855418", "name": "Bangladesh famine (1974)", "wiki": "Bangladesh famine of 1974"},
        {"qid": "Q3530493", "name": "Chalisa famine (1783)", "wiki": "Chalisa famine"},
    ],
    "liberation_leaders": [
        {"qid": "Q1001", "name": "Mahatma Gandhi", "wiki": "Mahatma Gandhi"},
        {"qid": "Q8023", "name": "Nelson Mandela", "wiki": "Nelson Mandela"},
        {"qid": "Q8605", "name": "Simon Bolivar", "wiki": "Sim%C3%B3n Bol%C3%ADvar"},
        {"qid": "Q8573", "name": "Sun Yat-sen", "wiki": "Sun Yat-sen"},
        {"qid": "Q11812", "name": "Thomas Jefferson", "wiki": "Thomas Jefferson"},
        {"qid": "Q36014", "name": "Ho Chi Minh", "wiki": "Ho Chi Minh"},
        {"qid": "Q8620", "name": "Kwame Nkrumah", "wiki": "Kwame Nkrumah"},
        {"qid": "Q205783", "name": "Toussaint Louverture", "wiki": "Toussaint Louverture"},
        {"qid": "Q173563", "name": "Jomo Kenyatta", "wiki": "Jomo Kenyatta"},
        {"qid": "Q539", "name": "Giuseppe Garibaldi", "wiki": "Giuseppe Garibaldi"},
        {"qid": "Q1500", "name": "Jose Rizal", "wiki": "Jos%C3%A9 Rizal"},
        {"qid": "Q23", "name": "George Washington", "wiki": "George Washington"},
        {"qid": "Q5152", "name": "Mustafa Kemal Ataturk", "wiki": "Mustafa Kemal Atat%C3%BCrk"},
        {"qid": "Q134160", "name": "Jose de San Martin", "wiki": "Jos%C3%A9 de San Mart%C3%ADn"},
        {"qid": "Q444", "name": "Lech Walesa", "wiki": "Lech Wa%C5%82%C4%99sa"},
        {"qid": "Q76127", "name": "Sukarno", "wiki": "Sukarno"},
        {"qid": "Q36740", "name": "Aung San Suu Kyi", "wiki": "Aung San Suu Kyi"},
        {"qid": "Q37610", "name": "David Ben-Gurion", "wiki": "David Ben-Gurion"},
        {"qid": "Q323419", "name": "Michael Collins", "wiki": "Michael Collins (Irish leader)"},
        {"qid": "Q186525", "name": "Julius Nyerere", "wiki": "Julius Nyerere"},
    ],
    "cold_war_events": [
        {"qid": "Q69163529", "name": "Fall of the Berlin Wall", "wiki": "Fall of the Berlin Wall"},
        {"qid": "Q128160", "name": "Cuban Missile Crisis", "wiki": "Cuban Missile Crisis"},
        {"qid": "Q5167679", "name": "Dissolution of the Soviet Union", "wiki": "Dissolution of the Soviet Union"},
        {"qid": "Q151349", "name": "Berlin Blockade", "wiki": "Berlin Blockade"},
        {"qid": "Q8740", "name": "Vietnam War", "wiki": "Vietnam War"},
        {"qid": "Q8663", "name": "Korean War", "wiki": "Korean War"},
        {"qid": "Q1932", "name": "Space Race", "wiki": "Space Race"},
        {"qid": "Q191721", "name": "Bay of Pigs Invasion", "wiki": "Bay of Pigs Invasion"},
        {"qid": "Q162401", "name": "Prague Spring", "wiki": "Prague Spring"},
        {"qid": "Q221382", "name": "1991 Soviet coup attempt", "wiki": "1991 Soviet coup d%27%C3%A9tat attempt"},
        {"qid": "Q164348", "name": "Hungarian Revolution (1956)", "wiki": "Hungarian Revolution of 1956"},
        {"qid": "Q5086", "name": "Berlin Wall construction", "wiki": "Berlin Wall"},
        {"qid": "Q7184", "name": "NATO", "wiki": "NATO"},
        {"qid": "Q41644", "name": "Warsaw Pact", "wiki": "Warsaw Pact"},
        {"qid": "Q83085", "name": "Soviet-Afghan War", "wiki": "Soviet%E2%80%93Afghan War"},
        {"qid": "Q273659", "name": "Iran-Contra affair", "wiki": "Iran%E2%80%93Contra affair"},
        {"qid": "Q171375", "name": "Truman Doctrine", "wiki": "Truman Doctrine"},
        {"qid": "Q862054", "name": "Nixon visit to China", "wiki": "Richard Nixon%27s 1972 visit to China"},
        {"qid": "Q80811", "name": "Sputnik launch", "wiki": "Sputnik"},
        {"qid": "Q748379", "name": "U-2 incident", "wiki": "1960 U-2 incident"},
    ],
    "world_fairs_olympics": [
        {"qid": "Q8150", "name": "1936 Berlin Olympics", "wiki": "1936 Summer Olympics"},
        {"qid": "Q59667269", "name": "1896 Athens Olympics", "wiki": "1896 Summer Olympics"},
        {"qid": "Q181278", "name": "1964 Tokyo Olympics", "wiki": "1964 Summer Olympics"},
        {"qid": "Q8438", "name": "1972 Munich Olympics", "wiki": "1972 Summer Olympics"},
        {"qid": "Q57176052", "name": "2008 Beijing Olympics", "wiki": "2008 Summer Olympics"},
        {"qid": "Q8450", "name": "1980 Moscow Olympics", "wiki": "1980 Summer Olympics"},
        {"qid": "Q285406", "name": "1893 World's Columbian Exposition", "wiki": "World%27s Columbian Exposition"},
        {"qid": "Q273095", "name": "1851 Great Exhibition", "wiki": "Great Exhibition"},
        {"qid": "Q957317", "name": "1889 Paris Exposition (Eiffel Tower)", "wiki": "Exposition Universelle (1889)"},
        {"qid": "Q1131758", "name": "1984 Los Angeles Olympics", "wiki": "1984 Summer Olympics"},
        {"qid": "Q8098", "name": "1904 St. Louis Olympics", "wiki": "1904 Summer Olympics"},
        {"qid": "Q207420", "name": "1939 New York World's Fair", "wiki": "1939 New York World%27s Fair"},
        {"qid": "Q57678450", "name": "1912 Stockholm Olympics", "wiki": "1912 Summer Olympics"},
        {"qid": "Q8150", "name": "1936 Berlin Olympics", "wiki": "1936 Winter Olympics"},
        {"qid": "Q28958341", "name": "1970 Osaka Expo", "wiki": "Expo %2770"},
        {"qid": "Q995653", "name": "1900 Paris Olympics", "wiki": "1900 Summer Olympics"},
        {"qid": "Q8444", "name": "1976 Montreal Olympics", "wiki": "1976 Summer Olympics"},
        {"qid": "Q8411", "name": "1956 Melbourne Olympics", "wiki": "1956 Summer Olympics"},
        {"qid": "Q8403", "name": "1948 London Olympics", "wiki": "1948 Summer Olympics"},
        {"qid": "Q8577", "name": "2012 London Olympics", "wiki": "2012 Summer Olympics"},
    ],
    "trade_routes": [
        {"qid": "Q58027", "name": "Silk Road", "wiki": "Silk Road"},
        {"qid": "Q465279", "name": "Trans-Saharan trade", "wiki": "Trans-Saharan trade"},
        {"qid": "Q135453380", "name": "Spice trade", "wiki": "Spice trade"},
        {"qid": "Q647110", "name": "Triangular trade", "wiki": "Triangular trade"},
        {"qid": "Q42908", "name": "Hanseatic League", "wiki": "Hanseatic League"},
        {"qid": "Q83164", "name": "East India Company", "wiki": "East India Company"},
        {"qid": "Q159766", "name": "Dutch East India Company", "wiki": "Dutch East India Company"},
        {"qid": "Q7350", "name": "Panama Canal", "wiki": "Panama Canal"},
        {"qid": "Q899", "name": "Suez Canal", "wiki": "Suez Canal"},
        {"qid": "Q862312", "name": "Oregon Trail", "wiki": "Oregon Trail"},
        {"qid": "Q1064041", "name": "Transcontinental Railroad", "wiki": "First transcontinental railroad"},
        {"qid": "Q58767", "name": "Trans-Siberian Railway", "wiki": "Trans-Siberian Railway"},
        {"qid": "Q2903677", "name": "Cape Route", "wiki": "Cape Route"},
        {"qid": "Q81136", "name": "Northwest Passage", "wiki": "Northwest Passage"},
        {"qid": "Q239574", "name": "Amber Road", "wiki": "Amber Road"},
        {"qid": "Q754999", "name": "Incense Route", "wiki": "Incense Route"},
        {"qid": "Q683846", "name": "Manila galleon trade", "wiki": "Manila galleon"},
        {"qid": "Q361047", "name": "Transatlantic cable", "wiki": "Transatlantic telegraph cable"},
        {"qid": "Q300701", "name": "Grand Trunk Road", "wiki": "Grand Trunk Road"},
        {"qid": "Q911953", "name": "Camino Real", "wiki": "Camino Real (road system)"},
    ],
}

# Categories/themes for history bench
HISTORY_THEMES: Dict[str, Dict[str, Any]] = {
    "conflicts": {"min_entities": 15},
    "organizations": {"min_entities": 15},
    "figures": {"min_entities": 15},
    "milestones": {"min_entities": 15},
    "treaties": {"min_entities": 10},
    "revolutions": {"min_entities": 8},
    "dynasties": {"min_entities": 10},
    "explorations": {"min_entities": 8},
    "inventions": {"min_entities": 8},
    "pandemics": {"min_entities": 8},
    "space_missions": {"min_entities": 10},
    "natural_disasters": {"min_entities": 10},
    "scientific_discoveries": {"min_entities": 8},
    "cultural_movements": {"min_entities": 8},
    "colonial_events": {"min_entities": 8},
    "archaeological_discoveries": {"min_entities": 8},
    # ---- NEW TOPICS (2026-03-16) ----
    "empires": {"min_entities": 12},
    "assassinations": {"min_entities": 12},
    "sieges": {"min_entities": 12},
    "religious_events": {"min_entities": 12},
    "coups": {"min_entities": 12},
    "migrations": {"min_entities": 12},
    "constitutions": {"min_entities": 12},
    "independence_movements": {"min_entities": 12},
    "civil_wars": {"min_entities": 12},
    "genocides_atrocities": {"min_entities": 12},
    "economic_crises": {"min_entities": 12},
    "naval_battles": {"min_entities": 12},
    "peace_accords": {"min_entities": 12},
    "scientific_institutions": {"min_entities": 12},
    "technological_milestones": {"min_entities": 12},
    "famines": {"min_entities": 12},
    "liberation_leaders": {"min_entities": 12},
    "cold_war_events": {"min_entities": 12},
    "world_fairs_olympics": {"min_entities": 12},
    "trade_routes": {"min_entities": 12},
}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(HISTORY_CACHE_DIR, exist_ok=True)


def _cached_get(url: str, cache_key: str, max_age_hours: int = 168) -> Optional[dict]:
    """Fetch URL with local JSON cache and rate limiting.

    Args:
        url: URL to fetch.
        cache_key: Key for local cache filename (sanitized).
        max_age_hours: Maximum age of cached data in hours (default 1 week).

    Returns:
        Parsed JSON dict, or None on failure.
    """
    global _last_wikidata_request_time
    _ensure_cache_dir()

    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(HISTORY_CACHE_DIR, f"{safe_key}.json")

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
    elapsed = time.time() - _last_wikidata_request_time
    if elapsed < _WIKIDATA_RATE_LIMIT_DELAY:
        time.sleep(_WIKIDATA_RATE_LIMIT_DELAY - elapsed)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
            _last_wikidata_request_time = time.time()

            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            elif response.status_code in (429, 503):
                wait = (2 ** attempt) * 5 + random.random() * 5
                print(f"History API rate-limited ({response.status_code}) for {cache_key}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"API error {response.status_code} for {cache_key}", flush=True)
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.random() * 5
            print(f"History API request error for {cache_key}: {e}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"Unexpected error for {cache_key}: {e}", flush=True)
            return None

    print(f"History API request failed after {max_retries} retries for {cache_key}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Wikidata Temporal Data Extraction
# ---------------------------------------------------------------------------

def parse_wikidata_time(time_str: str) -> Optional[int]:
    """Parse a Wikidata time value to extract the year.

    Handles ISO 8601 format from Wikidata: +1945-10-24T00:00:00Z
    Also handles BCE dates with negative years: -0100-03-15T00:00:00Z

    Args:
        time_str: Wikidata time string.

    Returns:
        Year as integer (negative for BCE), or None if unparseable.
    """
    if not time_str:
        return None

    # Wikidata format: +YYYY-MM-DDT... or -YYYY-MM-DDT...
    match = re.match(r'^([+-]?\d+)-\d{2}-\d{2}T', time_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # Fallback: just extract year
    match = re.match(r'^([+-]?\d+)', time_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


def get_entity_temporal_data(qid: str) -> Optional[Dict[str, Any]]:
    """Fetch temporal data for a Wikidata entity.

    Uses tool_util.get_entity_data() to get all claims, then extracts
    temporal property values.

    Args:
        qid: Wikidata entity ID (e.g., "Q362").

    Returns:
        Dict with parsed temporal fields, or None on failure.
    """
    cache_key = f"hist_temporal_{qid}"
    _ensure_cache_dir()

    # Check cache first
    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(HISTORY_CACHE_DIR, f"{safe_key}.json")
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 168:
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                # Invalidate cache entries with empty raw_dates (from old bug)
                if cached.get("raw_dates"):
                    return cached
            except (json.JSONDecodeError, IOError):
                pass

    # Fetch from Wikidata
    entity_data = get_entity_data(qid)
    if not entity_data:
        return None

    claims = entity_data.get("claims", {})
    result = {"qid": qid, "raw_dates": {}}

    # Extract all temporal properties
    # Handles both raw Wikidata format (mainsnak.datavalue) and
    # simplified format from tool_util.get_entity_data ({'type': 'time', 'value': ...})
    for prop_id, prop_label in TEMPORAL_PROPERTIES.items():
        if prop_id in claims:
            claim_list = claims[prop_id]
            # Sort by rank: preferred > normal > deprecated
            _RANK_ORDER = {"preferred": 0, "normal": 1, "deprecated": 2}
            sorted_claims = sorted(
                claim_list,
                key=lambda c: _RANK_ORDER.get(c.get("rank", "normal"), 1),
            )
            for claim in sorted_claims:
                time_val = None
                precision = 11  # default: day-level
                if "mainsnak" in claim:
                    # Raw Wikidata claim format
                    mainsnak = claim["mainsnak"]
                    if mainsnak.get("snaktype") != "value":
                        continue
                    dv = mainsnak.get("datavalue", {})
                    if dv.get("type") == "time":
                        time_val = dv.get("value", {}).get("time", "")
                        precision = dv.get("value", {}).get("precision", 11)
                elif claim.get("type") == "time":
                    # Simplified format from tool_util.get_entity_data
                    time_val = claim.get("value", "")
                    precision = claim.get("precision", 11)

                if time_val:
                    year = parse_wikidata_time(time_val)
                    if year is not None:
                        result["raw_dates"][prop_id] = {
                            "year": year,
                            "time_string": time_val,
                            "label": prop_label,
                            "precision": precision,
                        }
                        break  # Use first valid value per property (highest rank)

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(result, f)

    return result


def normalize_entity_dates(temporal_data: dict, entity_type: str) -> Dict[str, Any]:
    """Normalize temporal data into canonical start/end fields.

    Maps entity-type-specific Wikidata properties to canonical fields:
    - figures: birth_year, death_year
    - conflicts: start_year, end_year
    - organizations: inception_year, dissolution_year
    - milestones: event_year (from P585 > P580 > P571 > P577 > P1619)

    Args:
        temporal_data: Raw temporal data from get_entity_temporal_data().
        entity_type: One of "conflict", "organization", "figure", "milestone".

    Returns:
        Dict with normalized year fields.
    """
    raw = temporal_data.get("raw_dates", {})
    result = {"qid": temporal_data.get("qid", "")}

    def _get_year(prop_id: str, min_precision: int = 9) -> Optional[int]:
        """Return year for *prop_id* only if stored at year-level precision or better.

        Wikidata precision: 7=century, 8=decade, 9=year, 10=month, 11=day.
        Decade/century entries (precision < 9) are not specific enough and
        must be skipped to avoid using e.g. "1970" for "the 1970s".
        """
        entry = raw.get(prop_id)
        if entry and entry.get("precision", 11) >= min_precision:
            return entry.get("year")
        return None

    if entity_type == "figure":
        result["birth_year"] = _get_year("P569")
        result["death_year"] = _get_year("P570")

    elif entity_type == "conflict":
        result["start_year"] = _get_year("P580")
        result["end_year"] = _get_year("P582")

    elif entity_type == "organization":
        result["inception_year"] = _get_year("P571")
        result["dissolution_year"] = _get_year("P576")

    elif entity_type == "milestone":
        # Collect all candidate years from temporal properties
        _candidate_years = []
        for prop in ("P585", "P580", "P571", "P577", "P1619"):
            year = _get_year(prop)
            if year is not None:
                _candidate_years.append(year)
        if _candidate_years:
            # If there are multiple candidates and the first one looks like a
            # Wikidata precision artifact (tiny positive year while others are
            # much larger), skip it in favour of the next plausible candidate.
            chosen = _candidate_years[0]
            if (len(_candidate_years) >= 2
                    and 0 < chosen < 100
                    and any(abs(y) > 100 for y in _candidate_years[1:])):
                # Pick the first candidate with abs > 100
                for alt in _candidate_years[1:]:
                    if abs(alt) > 100:
                        chosen = alt
                        break
            result["event_year"] = chosen

    # Also populate generic start/end for cross-category templates
    if entity_type == "figure":
        result["start_year"] = result.get("birth_year")
        result["end_year"] = result.get("death_year")
    elif entity_type == "organization":
        result["start_year"] = result.get("inception_year")
        result["end_year"] = result.get("dissolution_year")
    elif entity_type == "milestone":
        result["start_year"] = result.get("event_year")
        result["end_year"] = result.get("event_year")

    # Store all raw year values for leakage checking (year-level precision only)
    all_years = set()
    for entry in raw.values():
        y = entry.get("year")
        if y is not None and entry.get("precision", 11) >= 9:
            all_years.add(y)
    result["_all_years"] = sorted(all_years)

    return result


# ---------------------------------------------------------------------------
# Wikipedia Entity Clue Fetching
# ---------------------------------------------------------------------------

def fetch_entity_clues(entity_name: str, entity_type: str = "conflict") -> List[dict]:
    """Fetch descriptive clue facts from Wikipedia for a historical entity.

    Args:
        entity_name: Wikipedia article title of the entity.
        entity_type: One of "conflict", "organization", "figure", "milestone".

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

    try:
        resp = requests.get(search_url, params=params, headers=_WIKI_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        search_data = resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []
    except Exception:
        return []

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
        resp = requests.get(extract_url, params=params, headers=_WIKI_HEADERS, timeout=15)
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

    sentences = re.split(r'(?<=[.!?])\s+', extract_text.strip())
    clues = []

    topic_keywords = {
        "conflict": {
            "causes": ["caused", "triggered", "arose", "resulted from", "origins"],
            "parties": ["fought", "between", "against", "allied", "coalition"],
            "timeline": ["began", "started", "ended", "lasted", "years"],
            "outcome": ["resulted", "led to", "victory", "defeat", "treaty"],
            "impact": ["casualties", "deaths", "destroyed", "devastated", "affected"],
        },
        "organization": {
            "founding": ["founded", "established", "created", "formed", "inaugurated"],
            "purpose": ["aims", "purpose", "mission", "mandate", "promote"],
            "members": ["members", "countries", "nations", "states", "signatories"],
            "location": ["headquartered", "based", "located", "Geneva", "New York"],
            "achievements": ["achieved", "awarded", "Nobel", "notable", "recognized"],
        },
        "figure": {
            "birth": ["born", "birthplace", "childhood", "grew up", "early life"],
            "career": ["known for", "famous", "contributed", "invented", "discovered"],
            "works": ["wrote", "published", "authored", "composed", "painted"],
            "legacy": ["legacy", "influence", "honored", "remembered", "recognized"],
            "death": ["died", "death", "passed away", "assassinated", "executed"],
        },
        "milestone": {
            "context": ["occurred", "took place", "happened", "during", "era"],
            "significance": ["significant", "landmark", "turning point", "pivotal", "marked"],
            "impact": ["changed", "transformed", "revolutionized", "led to", "sparked"],
            "location": ["in", "at", "near", "located", "took place"],
            "participants": ["led by", "organized by", "involved", "participants", "attended"],
        },
    }

    keywords = topic_keywords.get(entity_type, {})
    default_topics = ["description", "background", "details", "history", "context"]

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if len(clues) >= 10:
            break

        sent_lower = sent.lower()
        topic = default_topics[min(i, len(default_topics) - 1)]
        for topic_name, words in keywords.items():
            if any(w.lower() in sent_lower for w in words):
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


def get_all_entity_data(qid: str, entity_type: str) -> Optional[Dict[str, Any]]:
    """Fetch and normalize all temporal data for an entity.

    Combines get_entity_temporal_data() + normalize_entity_dates().

    Args:
        qid: Wikidata entity ID.
        entity_type: One of "conflict", "organization", "figure", "milestone".

    Returns:
        Dict with normalized temporal fields, or None.
    """
    temporal = get_entity_temporal_data(qid)
    if not temporal:
        return None
    return normalize_entity_dates(temporal, entity_type)


def search_entity(query: str) -> List[Dict[str, Any]]:
    """Search for historical entities in Wikidata.

    Args:
        query: Search text.

    Returns:
        List of {'id', 'label', 'description'} dicts.
    """
    return search_wikidata_entities(query, limit=10)


def get_entity_info(qid: str) -> Optional[Dict[str, Any]]:
    """Get basic entity information from Wikidata.

    Args:
        qid: Wikidata entity ID.

    Returns:
        Dict with label, description, and temporal data.
    """
    entity_data = get_entity_data(qid)
    if not entity_data:
        return None

    result = {
        "qid": qid,
        "label": entity_data.get("label", ""),
        "description": entity_data.get("description", ""),
    }

    # Also fetch temporal data
    temporal = get_entity_temporal_data(qid)
    if temporal:
        result["raw_dates"] = temporal.get("raw_dates", {})

    return result


def compare_entities(qid_a: str, qid_b: str) -> Optional[Dict[str, Any]]:
    """Compare temporal data between two entities.

    Args:
        qid_a: First entity QID.
        qid_b: Second entity QID.

    Returns:
        Dict with both entities' temporal data side by side.
    """
    data_a = get_entity_temporal_data(qid_a)
    data_b = get_entity_temporal_data(qid_b)
    if not data_a or not data_b:
        return None

    return {
        "entity_a": {"qid": qid_a, "raw_dates": data_a.get("raw_dates", {})},
        "entity_b": {"qid": qid_b, "raw_dates": data_b.get("raw_dates", {})},
    }


# ---------------------------------------------------------------------------
# Temporal grounding verification
# ---------------------------------------------------------------------------

# Field → human-readable claim templates
_TEMPORAL_CLAIM_TEMPLATES = {
    "birth_year": "This person was born in {year}",
    "death_year": "This person died in {year}",
    "start_year": "This event started in {year}",
    "end_year": "This event ended in {year}",
    "event_year": "This event occurred in {year}",
    "inception_year": "This organization was founded in {year}",
    "dissolution_year": "This organization was dissolved in {year}",
}

_TEMPORAL_GROUNDING_DEVELOPER = (
    "You are a fact-verification assistant that checks whether specific "
    "temporal claims are supported by Wikipedia article text."
)

_TEMPORAL_GROUNDING_PROMPT = """Determine whether the following temporal claim about "{entity_name}" is supported by the Wikipedia article excerpt below.

CLAIM: {claim_text}

WIKIPEDIA ARTICLE: "{article_title}"
---
{article_excerpt}
---

Is this temporal claim supported by the article text?
- SUPPORTED: The article explicitly states or directly implies this year.
- PARTIALLY_SUPPORTED: The article mentions related dates that are consistent with this claim.
- UNSUPPORTED: The article does not contain information supporting this year, or contradicts it.

Respond with exactly one word: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED"""


def verify_temporal_grounding(
    entity_data: dict,
    entity_name: str,
    qid: str,
    articles: list,
    agent_info,
) -> Optional[dict]:
    """Verify temporal year values from Wikidata against Wikipedia articles.

    For each key temporal field (birth_year, start_year, etc.), checks whether
    the year appears in the entity's Wikipedia article.  Uses a fast string
    check first (year substring in article text); falls back to LLM
    verification only when the fast check fails.

    Args:
        entity_data: Dict with normalized temporal fields (from get_all_entity_data).
        entity_name: Human-readable entity name.
        qid: Wikidata QID of the entity.
        articles: List of article dicts (from fetch_wikipedia_for_entities).
        agent_info: (lm, tokenizer, client) tuple for LLM calls, or None.

    Returns:
        entity_data unchanged if all key years are grounded, or None if any
        key year fails verification.
    """
    if not entity_data:
        return None

    # Collect temporal fields that have values
    temporal_facts = {}
    for field, template in _TEMPORAL_CLAIM_TEMPLATES.items():
        val = entity_data.get(field)
        if val is not None:
            temporal_facts[field] = (val, template.format(year=val))

    if not temporal_facts:
        # No temporal data to verify
        return entity_data

    # Find the entity's own Wikipedia article from articles list (match by QID)
    article = None
    for art in articles:
        if art.get("qid", "") == qid:
            article = art
            break

    # Fallback: try matching by entity name in title
    if article is None:
        name_lower = entity_name.lower().strip()
        for art in articles:
            if art.get("title", "").lower().strip() == name_lower:
                article = art
                break

    if article is None or not article.get("paragraph", "").strip():
        # No Wikipedia article available — be lenient, accept
        print(f"    Temporal grounding: no Wikipedia article for {entity_name} ({qid}), "
              f"skipping verification", flush=True)
        return entity_data

    article_text = article.get("paragraph", "")

    # Check each temporal fact
    for field, (year_val, claim_text) in temporal_facts.items():
        year_str = str(year_val)

        # Fast path: year string appears in article text
        if year_str in article_text:
            continue

        # Slow path: LLM verification
        # Prepare article excerpt (cap at 3000 chars, prioritize paragraphs mentioning dates)
        paragraphs = [p.strip() for p in article_text.split("\n") if p.strip()]
        scored = []
        for para in paragraphs:
            score = 0
            # Boost paragraphs with any 4-digit year
            if re.search(r"\b\d{4}\b", para):
                score += 1
            # Boost paragraphs mentioning the entity name
            if entity_name.lower() in para.lower():
                score += 1
            scored.append((score, para))
        scored.sort(key=lambda x: x[0], reverse=True)

        excerpt_parts = []
        total_chars = 0
        for _score, para in scored:
            if total_chars + len(para) > 3000 and excerpt_parts:
                break
            excerpt_parts.append(para)
            total_chars += len(para)
        article_excerpt = "\n\n".join(excerpt_parts)

        prompt_text = _TEMPORAL_GROUNDING_PROMPT.format(
            entity_name=entity_name,
            claim_text=claim_text,
            article_title=article.get("title", "Unknown"),
            article_excerpt=article_excerpt,
        )

        is_grounded = False
        try:
            from .wikidata_harmony import get_harmony_generator
            use_harmony = get_harmony_generator() is not None

            if use_harmony:
                gen = get_harmony_generator()
                response = gen.generate_response_sync(
                    developer_content=_TEMPORAL_GROUNDING_DEVELOPER,
                    user_content=prompt_text,
                    temperature=0.0,
                    max_tokens=256,
                    reasoning_effort="low",
                )
            elif agent_info is not None:
                from .util import gen_from_prompt
                agent_lm, agent_tokenizer, agent_client = agent_info
                full_prompt = _TEMPORAL_GROUNDING_DEVELOPER + "\n\n" + prompt_text
                result = gen_from_prompt(
                    model=agent_lm, tokenizer=agent_tokenizer,
                    prompt=[full_prompt],
                    echo_prompt=False, temperature=0.0, max_tokens=32,
                    process_func=None, service=agent_client,
                    terminate_by_linebreak="no", verbose=False,
                )
                response = result.completions[0].text
            else:
                # No LLM available — be lenient
                print(f"    Temporal grounding: no LLM available for {entity_name} "
                      f"{field}={year_val}, accepting", flush=True)
                continue

            verdict = response.strip().upper()
            # Must match exact verdicts — "UNSUPPORTED" contains the
            # substring "SUPPORTED", so a naive `in` check always passes.
            is_grounded = verdict.startswith("SUPPORTED") or verdict.startswith("PARTIALLY")
        except Exception as e:
            # On error, be lenient
            print(f"    Temporal grounding LLM error for {entity_name} "
                  f"{field}={year_val}: {e}", flush=True)
            continue

        if not is_grounded:
            print(f"    Temporal grounding FAILED: {entity_name} {field}={year_val} "
                  f"not supported by Wikipedia (verdict: {verdict})", flush=True)
            return None

    return entity_data


def calculate_duration(start_year: int, end_year: int) -> int:
    """Calculate duration in years between two years.

    Handles BCE dates correctly: -44 - (-100) = 56.

    Args:
        start_year: Start year (negative for BCE).
        end_year: End year (negative for BCE).

    Returns:
        Duration in years (always non-negative).
    """
    return abs(end_year - start_year)
