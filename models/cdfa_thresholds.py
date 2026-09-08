# -*- coding: utf-8 -*-
"""
CDFA-Inspired Economic Injury Level (EIL) & Action Threshold Engine
===================================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Synthesizes regulatory scouting methodologies from:
- California Department of Food and Agriculture (CDFA - PDEP)
- Kenya Agricultural and Livestock Research Organization (KALRO)
- Pest Control Products Board (PCPB) Kenya

Calculates actionable agronomic thresholds based on pest density, phenological stage,
and damage severity, preventing unnecessary chemical sprays and preserving natural predators.
"""

from typing import Any, Dict, Optional


# Standardized CDFA / KALRO Action Thresholds for Major East African Field Pests
ACTION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "fall_armyworm": {
        "common_name": "Fall Armyworm (Spodoptera frugiperda)",
        "unit": "larvae per 10 plants or % whorl damage",
        "thresholds": {
            "early_vegetative": {
                "action_threshold": 20.0,  # 20% plants showing windowpane damage
                "action": "SPRAY_BIOPESTICIDE",
                "recommendation": "Exceeds 20% early whorl threshold. Apply Bacillus thuringiensis (Bt) or Neem extract to whorls before larvae penetrate deeply."
            },
            "mid_to_late_whorl": {
                "action_threshold": 40.0,  # 40% whorl damage
                "action": "TARGETED_PCPB_INTERVENTION",
                "recommendation": "Exceeds 40% mid-whorl threshold. Apply PCPB-registered diamides (Chlorantraniliprole) directly into the funnel early morning or late evening."
            },
            "tasseling_silking": {
                "action_threshold": 10.0,  # 10% infested ears/silks
                "action": "IMMEDIATE_PROTECTION",
                "recommendation": "Exceeds 10% silk threshold. Larvae will penetrate ears directly. Immediate treatment required to prevent Cob Rot and ear destruction."
            }
        },
        "quarantine_threat": "HIGH",
        "cultural_controls": "Intercropping with Desmodium (Push-Pull strategy), handpicking egg masses in smallholder plots (<0.5 ha)."
    },
    "aphids": {
        "common_name": "Maize / Bean Aphids (Rhopalosiphum maidis / Aphis fabae)",
        "unit": "aphid colonies per leaf or % canopy infestation",
        "thresholds": {
            "vegetative": {
                "action_threshold": 50.0,  # >50 aphids per leaf on >20% plants
                "action": "CONSERVE_PREDATORS_OR_NEEM",
                "recommendation": "Observe for Ladybird beetles and Syrphid larvae. If predator count is low and honey-dew covers >20% leaves, spray potassium salts of fatty acids or Azadirachtin."
            },
            "flowering_fruiting": {
                "action_threshold": 25.0,  # Risk of Maize Dwarf Mosaic Virus transmission
                "action": "TARGETED_APHICIDE",
                "recommendation": "High risk of viral vectoring. Exceeds economic threshold. Apply Pirimicarb or systemic flonicamid; avoid broad-spectrum pyrethroids to protect bees."
            }
        },
        "quarantine_threat": "MODERATE",
        "cultural_controls": "Reflective plastic mulches, yellow sticky monitoring traps at 5 per hectare."
    },
    "spider_mites": {
        "common_name": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "unit": "% leaves with stippling or webbing",
        "thresholds": {
            "any_stage": {
                "action_threshold": 15.0,  # >15% leaves with bronzing/webbing
                "action": "MITICIDE_OR_PREDATORY_MITES",
                "recommendation": "Hot/dry weather triggers exponential flare-up. Apply Bifenazate or abamectin targeting leaf undersides, or release Phytoseiulus persimilis."
            }
        },
        "quarantine_threat": "MODERATE",
        "cultural_controls": "Overhead sprinkler irrigation to disrupt webbing, avoid excessive nitrogen fertilization which increases mite fecundity."
    },
    "stem_borer": {
        "common_name": "African Maize Stalk Borer (Busseola fusca / Chilo partellus)",
        "unit": "% plants with pinholes / deadheart",
        "thresholds": {
            "early_whorl": {
                "action_threshold": 10.0,  # >10% plants with pinholes in leaves
                "action": "WHORL_APPLICATION",
                "recommendation": "Apply coarse sand mixed with ash or targeted bio-granules into central whorls before larvae bore into stems."
            },
            "post_tasseling": {
                "action_threshold": 5.0,
                "action": "POST_HARVEST_SANITATION",
                "recommendation": "Larvae already inside stems. Chemical sprays ineffective. Slashing and burning/composting dry stalks post-harvest to destroy diapausing larvae."
            }
        },
        "quarantine_threat": "HIGH",
        "cultural_controls": "Silverleaf Desmodium intercrop with Napier grass border (ICIPE Push-Pull system)."
    }
}


def evaluate_cdfa_threshold(
    pest_name: str,
    pest_count: int,
    crop_stage: str = "vegetative",
    total_plants_sampled: int = 10
) -> Dict[str, Any]:
    """
    Evaluates observed pest count against CDFA/KALRO Economic Injury Levels (EIL).
    Returns clear action advice: NO_ACTION (cultural control only) vs THRESHOLD_EXCEEDED (intervention).
    """
    clean_pest = pest_name.lower().replace(" ", "_")
    matched_key = None
    for k in ACTION_THRESHOLDS:
        if k in clean_pest or clean_pest in k:
            matched_key = k
            break

    if not matched_key:
        return {
            "evaluated": False,
            "pest": pest_name,
            "message": f"Standard economic threshold data not cataloged for {pest_name}. Apply standard KALRO good agricultural practices."
        }

    pest_info = ACTION_THRESHOLDS[matched_key]
    thresholds = pest_info["thresholds"]
    stage_key = "any_stage" if "any_stage" in thresholds else ("early_vegetative" if "veg" in crop_stage.lower() else list(thresholds.keys())[0])
    
    stage_rule = thresholds.get(stage_key, list(thresholds.values())[0])
    action_threshold_val = stage_rule["action_threshold"]

    # Calculate observed infestation rate
    observed_pct = (pest_count / max(1, total_plants_sampled)) * 100.0 if "per 10" in pest_info["unit"] else float(pest_count)
    threshold_exceeded = observed_pct >= action_threshold_val

    return {
        "evaluated": True,
        "pest_common_name": pest_info["common_name"],
        "growth_stage": stage_key,
        "observed_density": pest_count,
        "action_threshold_value": action_threshold_val,
        "threshold_unit": pest_info["unit"],
        "threshold_exceeded": threshold_exceeded,
        "action_level": stage_rule["action"] if threshold_exceeded else "CONSERVE_PRESERVE",
        "regulatory_guidance": stage_rule["recommendation"] if threshold_exceeded else "Density is BELOW Economic Injury Level. DO NOT SPRAY broad-spectrum chemicals. Conserve beneficial predators.",
        "quarantine_level": pest_info["quarantine_threat"],
        "cultural_ipm": pest_info["cultural_controls"],
        "sampling_protocol": "CDFA 5-Point Field Grid: Sample 10 plants at 5 stations across the field (W-pattern)."
    }
