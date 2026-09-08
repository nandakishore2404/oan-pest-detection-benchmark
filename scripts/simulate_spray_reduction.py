# -*- coding: utf-8 -*-
"""
CDFA Economic Injury Level (EIL) Spray Reduction Simulator
==========================================================
Program Manager -> AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla (nandakishore.kakulla9@gmail.com)
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Simulates chemical spray recommendations across 1,000 randomized field scouting events
comparing:
1. Naive Binary Computer Vision Policy (spray whenever any pest bounding box >= 1 is detected).
2. CDFA / KALRO Economic Injury Level (EIL) Action Threshold Policy (spray ONLY when infestation exceeds stage threshold).

Generates results/reports/cdfa_spray_simulation.json with empirical evidence.
"""

import os
import sys
import json
import random
from typing import Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.cdfa_thresholds import evaluate_cdfa_threshold, ACTION_THRESHOLDS

def run_simulation(num_events: int = 1000, seed: int = 42) -> Dict[str, Any]:
    random.seed(seed)
    
    stages = [
        ("early_vegetative", 0.30),
        ("mid_to_late_whorl", 0.40),
        ("tasseling_silking", 0.20),
        ("grain_fill_maturity", 0.10)
    ]
    stage_choices = [s[0] for s in stages]
    stage_weights = [s[1] for s in stages]
    
    pests = ["fall_armyworm", "aphids", "spider_mites", "stem_borer"]
    
    naive_sprays = 0
    cdfa_sprays = 0
    sub_threshold_detections = 0
    healthy_scouts = 0
    
    breakdown_by_stage = {s: {"scouts": 0, "naive_sprays": 0, "cdfa_sprays": 0} for s in stage_choices}
    breakdown_by_pest = {p: {"scouts": 0, "naive_sprays": 0, "cdfa_sprays": 0} for p in pests}
    
    for _ in range(num_events):
        stage = random.choices(stage_choices, weights=stage_weights, k=1)[0]
        pest = random.choice(pests)
        breakdown_by_stage[stage]["scouts"] += 1
        breakdown_by_pest[pest]["scouts"] += 1
        
        has_pest = random.random() < 0.75
        
        if not has_pest:
            pest_count = 0
            healthy_scouts += 1
        else:
            roll = random.random()
            if roll < 0.60:
                pest_count = random.randint(1, 2)
            elif roll < 0.85:
                pest_count = random.randint(3, 4)
            else:
                pest_count = random.randint(5, 10)
        
        naive_triggered = (pest_count >= 1)
        if naive_triggered:
            naive_sprays += 1
            breakdown_by_stage[stage]["naive_sprays"] += 1
            breakdown_by_pest[pest]["naive_sprays"] += 1
            
        cdfa_eval = evaluate_cdfa_threshold(pest, pest_count, crop_stage=stage, total_plants_sampled=10)
        cdfa_triggered = cdfa_eval.get("threshold_exceeded", False)
        
        if cdfa_triggered:
            cdfa_sprays += 1
            breakdown_by_stage[stage]["cdfa_sprays"] += 1
            breakdown_by_pest[pest]["cdfa_sprays"] += 1
        else:
            if naive_triggered:
                sub_threshold_detections += 1
                
    spray_reduction_pct = round(((naive_sprays - cdfa_sprays) / max(1, naive_sprays)) * 100.0, 1)
    false_alarm_rate_naive = round((sub_threshold_detections / max(1, naive_sprays)) * 100.0, 1)
    
    cost_per_spray_kes = 2500
    avoided_sprays_est = (naive_sprays - cdfa_sprays) / num_events * 4.0
    savings_per_ha_kes = round(avoided_sprays_est * cost_per_spray_kes, 0)
    savings_per_ha_usd = round(savings_per_ha_kes / 130.0, 2)
    
    report = {
        "simulation_metadata": {
            "title": "CDFA Economic Injury Level (EIL) Spray Reduction Simulation",
            "author": "Program Manager -> AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>",
            "total_simulated_field_scouting_events": num_events,
            "random_seed": seed,
            "sampling_grid": "CDFA 5-Point Field Grid (10 plants per station, W-pattern)",
            "standards_referenced": ["CDFA-PDEP California", "KALRO Kenya", "PCPB Kenya"]
        },
        "simulation_results": {
            "naive_binary_policy_sprays_triggered": naive_sprays,
            "cdfa_eil_policy_sprays_triggered": cdfa_sprays,
            "unnecessary_sprays_prevented": naive_sprays - cdfa_sprays,
            "spray_reduction_percentage": spray_reduction_pct,
            "sub_threshold_detections_conserved": sub_threshold_detections,
            "naive_policy_premature_spray_rate_pct": false_alarm_rate_naive,
            "cdfa_policy_actionable_spray_rate_pct": round((cdfa_sprays / max(1, naive_sprays)) * 100.0, 1)
        },
        "economic_protection_kenya": {
            "cost_per_chemical_spray_kes": cost_per_spray_kes,
            "avoided_sprays_per_season_per_ha": round(avoided_sprays_est, 2),
            "estimated_farmer_savings_per_ha_kes": savings_per_ha_kes,
            "estimated_farmer_savings_per_ha_usd": savings_per_ha_usd
        },
        "breakdown_by_crop_stage": breakdown_by_stage,
        "breakdown_by_pest": breakdown_by_pest
    }
    
    out_dir = os.path.join(REPO_ROOT, "results", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cdfa_spray_simulation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Simulation completed across {num_events} scouting events.")
    print(f"Naive Binary Sprays: {naive_sprays} | CDFA EIL Sprays: {cdfa_sprays}")
    print(f"Chemical Spray Reduction: {spray_reduction_pct}%")
    print(f"Naive Policy Premature Spray Rate (Sub-EIL): {false_alarm_rate_naive}%")
    print(f"Estimated Farmer Savings: KES {savings_per_ha_kes:,} (~${savings_per_ha_usd} USD) per ha/season")
    print(f"Saved artifact to: {out_path}")
    return report

if __name__ == "__main__":
    run_simulation()
