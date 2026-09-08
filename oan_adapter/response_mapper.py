# -*- coding: utf-8 -*-
"""
Beckn / OAN Response Mapper
===========================
Transforms normalized AI inference predictions into standard Beckn /on_search catalog items.
"""

from datetime import datetime, timezone
from typing import Dict, Any

def map_inference_to_beckn_on_search(
    internal_pred: Dict[str, Any],
    beckn_context: Dict[str, Any],
    bpp_id: str = "bpp.pest-ai.oan.kenya.deloitte.org"
) -> Dict[str, Any]:
    """
    Packages a NormalizedPrediction into a Beckn 1.1.0 /on_search catalog response.
    """
    is_uncertain = internal_pred.get("uncertain", False)
    diag = internal_pred.get("diagnosis", "Unknown")
    conf = internal_pred.get("confidence", 0.0)

    # Build Beckn catalog item
    item = {
        "id": f"diag_{internal_pred.get('request_id', '001')}",
        "descriptor": {
            "name": f"Diagnosis: {diag}",
            "code": "DIAG_RESULT",
            "short_desc": f"Automated confidence: {round(conf * 100, 1)}%. Advisory: {internal_pred.get('advisory', 'N/A')}"
        },
        "price": {
            "currency": "KES",
            "value": "0.00"
        },
        "tags": {
            "confidence_score": str(conf),
            "uncertain_status": str(is_uncertain),
            "recommend_expert_review": str(internal_pred.get("recommend_human_review", False)),
            "inference_latency_ms": str(internal_pred.get("inference_time_ms", 0.0)),
            "model_engine": internal_pred.get("model_id", "mobilenetv4_5class")
        }
    }

    ctx = dict(beckn_context)
    ctx["action"] = "on_search"
    ctx["timestamp"] = datetime.now(timezone.utc).isoformat()
    ctx["bpp_id"] = bpp_id

    return {
        "context": ctx,
        "message": {
            "catalog": {
                "bpp/descriptor": {
                    "name": "OAN Kenya AI Diagnostic Provider"
                },
                "bpp/providers": [
                    {
                        "id": "oan-pest-ai-provider-kenya-01",
                        "descriptor": {"name": "Pest & Disease Advisory Unit"},
                        "items": [item]
                    }
                ]
            }
        }
    }
