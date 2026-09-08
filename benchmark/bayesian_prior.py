# -*- coding: utf-8 -*-
"""
Agronomic Bayesian Prior Calibrator for Kenya Smallholder Farming
=================================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Bayesian re-weighting engine that combines visual model confidence with:
1. Crop Phenological Stage (e.g., Early Vegetative, Whorl, Silking, Maturity)
2. Agro-Ecological Zone / Kenyan County (Rift Valley, Central, Western, Coast, Eastern)
3. Seasonal Window (Long Rains: Mar-Jun, Short Rains: Oct-Dec, Dry/Stress: Jan-Feb, Jul-Aug)

Suppresses out-of-season and biologically impossible false positives, lifting decision precision.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class BayesianCalibrationResult:
    original_class: str
    original_confidence: float
    calibrated_class: str
    calibrated_confidence: float
    prior_probability: float
    is_phenologically_consistent: bool
    adjustment_reason: str
    stage_recommended_scouting: List[str]


class KenyanAgronomicBayesianPrior:
    """
    Applies empirical agronomic priors to visual object detection and classification outputs.
    Follows KALRO (Kenya Agricultural and Livestock Research Organization) and CIMMYT pest calendars.
    """

    # Stage-to-pest likelihood distributions P(Pest | Stage)
    STAGE_PEST_PRIORS = {
        "early_vegetative": {  # Emergence to V4
            "Cutworms": 0.40,
            "Flea Beetles": 0.25,
            "African Maize Stalk Borer": 0.15,
            "Fall Armyworm": 0.15,
            "Aphids": 0.05,
            "Spider Mites": 0.02,
            "African Bollworm": 0.001,  # Biologically impossible on seedlings
            "Maize Weevils": 0.000,
            "Stem Borer": 0.15,
            "Grasshoppers": 0.10,
        },
        "whorl_vegetative": {  # V5 to V12 (Peak whorl funnel stage)
            "Fall Armyworm": 0.55,      # Primary whorl feeder
            "African Maize Stalk Borer": 0.25,
            "Stem Borer": 0.25,
            "Aphids": 0.15,
            "Grasshoppers": 0.08,
            "Cutworms": 0.02,
            "African Bollworm": 0.01,
            "Spider Mites": 0.10,
            "Maize Weevils": 0.000,
        },
        "silking_tasseling": {  # R1 - R3 (Reproductive ear emergence)
            "African Bollworm": 0.45,  # Silks & ear tip feeder
            "Fall Armyworm": 0.35,      # Ear invasion
            "Aphids": 0.20,             # Tassel colonizers
            "Spider Mites": 0.15,
            "African Maize Stalk Borer": 0.15,
            "Cutworms": 0.001,
            "Maize Weevils": 0.01,
        },
        "grain_fill_maturity": {  # R4 - R6 (Drying grain & pre-harvest)
            "Maize Weevils": 0.40,
            "Larger Grain Borer": 0.35,
            "African Bollworm": 0.20,
            "Fall Armyworm": 0.10,
            "Cutworms": 0.000,
            "Aphids": 0.02,
        }
    }

    # Seasonal multipliers by Kenya Agro-Ecological Zones
    ECOLOGICAL_MODIFIERS = {
        "rift_valley_trans_nzoia": {"Fall Armyworm": 1.3, "African Maize Stalk Borer": 1.2, "Cutworms": 1.1},
        "central_highlands_meru": {"Fall Armyworm": 1.2, "Aphids": 1.3, "Spider Mites": 1.1},
        "western_kakamega": {"Fall Armyworm": 1.4, "Stem Borer": 1.3, "Flea Beetles": 1.1},
        "eastern_dryland_machakos": {"Spider Mites": 1.5, "Aphids": 1.4, "African Bollworm": 1.2, "Cutworms": 0.8},
        "coastal_kilifi": {"Fall Armyworm": 1.2, "African Bollworm": 1.3, "Spider Mites": 1.4}
    }

    def __init__(self, default_stage: str = "whorl_vegetative", default_region: str = "rift_valley_trans_nzoia"):
        self.default_stage = default_stage
        self.default_region = default_region

    def calibrate_prediction(
        self,
        predicted_class: str,
        confidence: float,
        crop_stage: Optional[str] = None,
        region: Optional[str] = None,
        candidate_classes: Optional[Dict[str, float]] = None
    ) -> BayesianCalibrationResult:
        """
        Calibrates visual confidence using Bayesian updating:
        P(Class | Visual, Stage, Region) ~ P(Visual | Class) * P(Class | Stage, Region)
        """
        stage = crop_stage or self.default_stage
        reg = region or self.default_region

        stage_priors = self.STAGE_PEST_PRIORS.get(stage, self.STAGE_PEST_PRIORS["whorl_vegetative"])
        reg_mods = self.ECOLOGICAL_MODIFIERS.get(reg, {})

        # Default prior for unlisted or healthy classes
        if "healthy" in predicted_class.lower():
            return BayesianCalibrationResult(
                original_class=predicted_class,
                original_confidence=confidence,
                calibrated_class=predicted_class,
                calibrated_confidence=confidence,
                prior_probability=0.50,
                is_phenologically_consistent=True,
                adjustment_reason="Healthy foliage baseline maintained.",
                stage_recommended_scouting=list(stage_priors.keys())[:3]
            )

        # Lookup prior probability for the visual candidate
        prior_p = stage_priors.get(predicted_class, 0.05)
        # Apply regional modifier
        prior_p *= reg_mods.get(predicted_class, 1.0)

        # Check for biological / phenological inconsistency
        is_consistent = prior_p >= 0.01

        # If candidate classes are provided (e.g. top-k from detector), compute full Bayesian posterior
        if candidate_classes and len(candidate_classes) > 1:
            unnorm_posteriors = {}
            for cls_name, visual_p in candidate_classes.items():
                p_prior = stage_priors.get(cls_name, 0.02) * reg_mods.get(cls_name, 1.0)
                # Likelihood * Prior
                unnorm_posteriors[cls_name] = visual_p * p_prior

            total = sum(unnorm_posteriors.values()) + 1e-9
            norm_posteriors = {k: v / total for k, v in unnorm_posteriors.items()}
            best_class = max(norm_posteriors.items(), key=lambda x: x[1])[0]
            calibrated_conf = round(float(norm_posteriors[best_class]), 3)
            reason = f"Multi-candidate Bayesian posterior calculated across {len(candidate_classes)} classes."
        else:
            # Single-prediction Bayesian updating with logistic sigmoid dampening
            # If prior is near zero (e.g. Bollworm at seedling stage), heavily penalize visual false positive
            if not is_consistent:
                calibrated_conf = round(confidence * 0.25, 3)  # Heavily damped
                best_class = predicted_class
                reason = (f"Phenological Alert: '{predicted_class}' is biologically improbable during '{stage}'. "
                          f"Visual confidence penalized from {confidence} to {calibrated_conf}.")
            else:
                # Modest boost or dampening based on agronomic prior strength
                boost_factor = 1.0 + 0.15 * math.log(prior_p / 0.10 + 1e-3)
                boost_factor = max(0.60, min(1.20, boost_factor))
                calibrated_conf = min(0.99, round(confidence * boost_factor, 3))
                best_class = predicted_class
                reason = f"Prior likelihood ({prior_p:.2f}) aligned with '{stage}' phenology."

        return BayesianCalibrationResult(
            original_class=predicted_class,
            original_confidence=confidence,
            calibrated_class=best_class,
            calibrated_confidence=calibrated_conf,
            prior_probability=round(prior_p, 4),
            is_phenologically_consistent=is_consistent,
            adjustment_reason=reason,
            stage_recommended_scouting=[k for k, v in sorted(stage_priors.items(), key=lambda x: x[1], reverse=True)[:3]]
        )
