# -*- coding: utf-8 -*-
"""
Beckn / OAN Request Mapper
==========================
Transforms incoming Beckn protocol search/select payloads into internal AI inference requests.
"""

from typing import Dict, Any, Optional

def map_beckn_search_to_inference_req(beckn_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses Beckn /search intent and extracts image URI, crop context, and farmer location.
    """
    message = beckn_payload.get("message", {})
    intent = message.get("intent", {})
    item = intent.get("item", {})
    tags = item.get("tags", {})

    image_url = tags.get("image_url") or tags.get("photo_uri")
    crop = tags.get("crop") or "Unspecified"
    location = tags.get("county") or "Western Kenya"

    return {
        "image_url": image_url,
        "crop": crop,
        "location": location,
        "transaction_id": beckn_payload.get("context", {}).get("transaction_id"),
        "bap_id": beckn_payload.get("context", {}).get("bap_id")
    }
