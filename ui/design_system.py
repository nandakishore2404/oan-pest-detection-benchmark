# -*- coding: utf-8 -*-
"""
Design System & Line-SVG Icon Library
====================================
OpenAgriNet (OAN) Kenya · Pest & Disease Lab
Two-Audience Frontend System (Farmer Mobile + Engineering Desktop Lab)

Author: Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
"""

import streamlit as st

# ==============================================================================
# 1. COLOR PALETTE TOKENS
# ==============================================================================
COLOR_GROUND = "#070c0a"       # Page background
COLOR_SURFACE = "#0d1712"      # Cards & containers
COLOR_SURFACE_HOVER = "#13221b"
COLOR_ACCENT = "#22c08a"       # Primary / Teal-green
COLOR_ACCENT_MUTED = "rgba(34, 192, 138, 0.15)"
COLOR_CAUTION = "#e7b458"      # Amber / Moderate risk
COLOR_CAUTION_MUTED = "rgba(231, 180, 88, 0.15)"
COLOR_ALERT = "#e2847a"        # Rose / Red / High risk
COLOR_ALERT_MUTED = "rgba(226, 132, 122, 0.15)"
COLOR_TEXT = "#eef3ef"         # Primary text
COLOR_MUTED = "#7c8a82"        # Secondary / captions
COLOR_MUTED_LIGHT = "#8b9a91"
COLOR_BORDER = "#1c2620"       # Thin component border
COLOR_BORDER_FOCUS = "#284235"

# Confidence Badge Palettes
CONF_HIGH_BG = "#173226"
CONF_HIGH_FG = "#5fe0ac"
CONF_MED_BG = "#332813"
CONF_MED_FG = "#e7b458"
CONF_LOW_BG = "#33201c"
CONF_LOW_FG = "#e2847a"


# ==============================================================================
# 2. LINE-STYLE SVG ICONOGRAPHY (STROKE-ONLY, ~1.8px WEIGHT, ZERO EMOJI)
# ==============================================================================
def svg_leaf(color: str = COLOR_ACCENT, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
        <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>"""

def svg_sun(color: str = COLOR_CAUTION, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="4"/>
        <path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/>
        <path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>
    </svg>"""

def svg_camera(color: str = COLOR_GROUND, size: int = 22) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
        <circle cx="12" cy="13" r="3"/>
    </svg>"""

def svg_shield(color: str = COLOR_ACCENT, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>"""

def svg_shield_alert(color: str = COLOR_CAUTION, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="M12 8v4"/><path d="M12 16h.01"/>
    </svg>"""

def svg_checkmark(color: str = COLOR_ACCENT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 6 9 17l-5-5"/>
    </svg>"""

def svg_chat_bubble(color: str = COLOR_ACCENT, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>"""

def svg_settings(color: str = COLOR_MUTED, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>"""

def svg_chevron_left(color: str = COLOR_TEXT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round">
        <path d="m15 18-6-6 6-6"/>
    </svg>"""

def svg_chevron_right(color: str = COLOR_MUTED, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round">
        <path d="m9 18 6-6-6-6"/>
    </svg>"""

def svg_chevron_down(color: str = COLOR_MUTED, size: int = 16) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round">
        <path d="m6 9 6 6 6-6"/>
    </svg>"""

def svg_sliders(color: str = COLOR_MUTED, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/>
        <path d="M1 14h6"/><path d="M9 8h6"/><path d="M17 16h6"/>
    </svg>"""

def svg_microscope(color: str = COLOR_ACCENT, size: int = 20) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M6 18h8"/><path d="M3 22h18"/><path d="m14 22 3-3-3-3"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>
        <path d="m12 6 1-4 4 1-1 4"/>
    </svg>"""

def svg_cpu(color: str = COLOR_MUTED, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect width="16" height="16" x="4" y="4" rx="2"/>
        <rect width="6" height="6" x="9" y="9" rx="1"/>
        <path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/>
    </svg>"""

def svg_clock(color: str = COLOR_MUTED, size: int = 16) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
    </svg>"""

def svg_bookmark(color: str = COLOR_TEXT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>
    </svg>"""

def svg_user(color: str = COLOR_TEXT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 1 0-16 0"/>
    </svg>"""

def svg_zap(color: str = COLOR_ACCENT, size: int = 16) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>"""

def svg_chart_bar(color: str = COLOR_ACCENT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="20" x2="12" y2="10"/>
        <line x1="18" y1="20" x2="18" y2="4"/>
        <line x1="6" y1="20" x2="6" y2="16"/>
    </svg>"""

def svg_activity(color: str = COLOR_ACCENT, size: int = 18) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>"""


# ==============================================================================
# 3. GLOBAL THEME & TYPOGRAPHY INJECTION
# ==============================================================================
def inject_theme():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Work+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root Variables */
        :root {{
            --ground: {COLOR_GROUND};
            --surface: {COLOR_SURFACE};
            --surface-hover: {COLOR_SURFACE_HOVER};
            --accent: {COLOR_ACCENT};
            --caution: {COLOR_CAUTION};
            --alert: {COLOR_ALERT};
            --text-primary: {COLOR_TEXT};
            --text-muted: {COLOR_MUTED};
            --text-muted-light: {COLOR_MUTED_LIGHT};
            --border: {COLOR_BORDER};
            --border-focus: {COLOR_BORDER_FOCUS};
            --font-display: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-body: 'Work Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        /* Global App Styling */
        .stApp {{
            background-color: var(--ground) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--font-display) !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }}

        /* Mobile Container Constraint */
        .mobile-viewport-wrapper {{
            max-width: 410px;
            margin: 0 auto;
            background-color: var(--ground);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 16px;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45);
            position: relative;
        }}

        /* Cards & Radii */
        .oan-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 14px;
            transition: border-color 0.2s ease, transform 0.15s ease;
        }}
        .oan-card:hover {{
            border-color: var(--border-focus);
        }}

        /* Stat Tile Component */
        .stat-tile {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .stat-label {{
            font-size: 11px;
            font-family: var(--font-display);
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .stat-value {{
            font-size: 22px;
            font-family: var(--font-display);
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.15;
        }}
        .stat-sub {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        /* Confidence Badges (Pills) */
        .conf-badge {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 5px 12px;
            border-radius: 999px;
            font-family: var(--font-display);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.02em;
            line-height: 1;
        }}
        .conf-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }}
        .conf-high {{
            background: {CONF_HIGH_BG};
            color: {CONF_HIGH_FG};
            border: 1px solid rgba(95, 224, 172, 0.35);
        }}
        .conf-high .conf-dot {{
            background: {CONF_HIGH_FG};
            box-shadow: 0 0 6px {CONF_HIGH_FG};
        }}
        .conf-med {{
            background: {CONF_MED_BG};
            color: {CONF_MED_FG};
            border: 1px solid rgba(231, 180, 88, 0.35);
        }}
        .conf-med .conf-dot {{
            background: {CONF_MED_FG};
            box-shadow: 0 0 6px {CONF_MED_FG};
        }}
        .conf-low {{
            background: {CONF_LOW_BG};
            color: {CONF_LOW_FG};
            border: 1px solid rgba(226, 132, 122, 0.35);
        }}
        .conf-low .conf-dot {{
            background: {CONF_LOW_FG};
            box-shadow: 0 0 6px {CONF_LOW_FG};
        }}

        /* Safety Brake Banner */
        .safety-brake-banner {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--caution);
            border-radius: 14px;
            padding: 16px;
            margin: 14px 0;
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }}
        .safety-brake-title {{
            font-family: var(--font-display);
            font-size: 14px;
            font-weight: 700;
            color: var(--caution);
            margin-bottom: 4px;
        }}
        .safety-brake-body {{
            font-size: 12px;
            color: var(--text-muted-light);
            line-height: 1.45;
        }}

        /* Action Step Card */
        .action-step-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 10px;
        }}
        .action-number-bubble {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(34, 192, 138, 0.15);
            border: 1.5px solid var(--accent);
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 12px;
            flex-shrink: 0;
        }}
        .action-step-title {{
            font-family: var(--font-display);
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 3px;
        }}
        .action-step-desc {{
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.4;
        }}

        /* Primary & Secondary Pill Buttons */
        .btn-pill-primary {{
            background: var(--accent) !important;
            color: var(--ground) !important;
            font-family: var(--font-display) !important;
            font-weight: 700 !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 10px 22px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            text-decoration: none !important;
            transition: opacity 0.2s ease, transform 0.1s ease !important;
            cursor: pointer !important;
        }}
        .btn-pill-primary:hover {{
            opacity: 0.9 !important;
            transform: translateY(-1px);
        }}
        .btn-pill-secondary {{
            background: transparent !important;
            color: var(--text-primary) !important;
            font-family: var(--font-display) !important;
            font-weight: 600 !important;
            border-radius: 999px !important;
            border: 1px solid var(--border) !important;
            padding: 10px 18px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            text-decoration: none !important;
            transition: background-color 0.2s ease !important;
            cursor: pointer !important;
        }}
        .btn-pill-secondary:hover {{
            background-color: var(--surface) !important;
            border-color: var(--border-focus) !important;
        }}

        /* Sample Chip */
        .sample-chip {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-right: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .sample-chip:hover {{
            border-color: var(--accent);
            background: var(--surface-hover);
        }}
        .sample-chip.active {{
            border-color: var(--accent);
            background: rgba(34, 192, 138, 0.12);
        }}

        /* Trust Strip Pills */
        .trust-pill {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 11px;
            color: var(--text-muted-light);
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
        }}

        /* Sticky Bottom Shamba AI Bar */
        .sticky-shamba-bar {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 18px;
            cursor: pointer;
            transition: border-color 0.2s ease;
        }}
        .sticky-shamba-bar:hover {{
            border-color: var(--accent);
        }}

        /* Model Trust Footer Strip */
        .model-trust-strip {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 20px;
        }}

        /* Streamlit Input Override for Dark Theme */
        .stSelectbox, .stSlider, .stFileUploader {{
            font-family: var(--font-body) !important;
        }}

        /* -------------------------------------------------------------
           Device Screensize & Strict Image Constraints
           Ensures uploaded and analyzed photos fit within screen bounds
           ------------------------------------------------------------- */
        [data-testid="stImage"] {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            margin: 6px auto !important;
        }}

        [data-testid="stImage"] img {{
            max-height: 270px !important;
            max-width: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
            background-color: #040806 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        }}

        [data-testid="stImage"] [data-testid="stCaptionContainer"] {{
            font-size: 11px !important;
            color: var(--text-muted) !important;
            text-align: center !important;
            margin-top: 5px !important;
        }}

        /* Lab View 3-panel side-by-side images */
        .lab-photo-panel [data-testid="stImage"] img {{
            max-height: 220px !important;
            height: 220px !important;
            object-fit: contain !important;
        }}

        /* Mobile Device Mockup Frame for Farmer Experience.
           .mobile-device-shell is kept for any legacy raw-HTML usage; the
           .st-key-* selectors are the real target now that the Farmer screens
           use st.container(key=...) so this style actually wraps the
           Streamlit-native widgets inside it (a plain unsafe_allow_html div
           does NOT nest later st.* calls in the real DOM - Streamlit renders
           each element as its own sibling block). */
        .mobile-device-shell,
        .st-key-mobile_shell_home,
        .st-key-mobile_shell_result {{
            max-width: 430px;
            width: 100%;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 20px 18px;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55);
        }}

        /* Responsive Breakpoints for Mobile Phones & Tablets */
        @media (max-width: 768px) {{
            .mobile-device-shell,
            .st-key-mobile_shell_home,
            .st-key-mobile_shell_result {{
                max-width: 100% !important;
                border: none !important;
                padding: 10px 4px !important;
                box-shadow: none !important;
                background: transparent !important;
            }}
            .mobile-viewport-wrapper {{
                max-width: 100% !important;
                border: none !important;
                padding: 10px 4px !important;
                box-shadow: none !important;
            }}
            [data-testid="stImage"] img {{
                max-height: 240px !important;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. REUSABLE HTML COMPONENT BUILDERS
# ==============================================================================
def render_confidence_badge(confidence: float) -> str:
    """Renders high / medium / low confidence badge pill."""
    conf_pct = round(confidence * 100, 1)
    if conf_pct >= 75.0:
        css_class = "conf-high"
        label = f"High Confidence · {conf_pct}%"
    elif conf_pct >= 40.0:
        css_class = "conf-med"
        label = f"Moderate · {conf_pct}%"
    else:
        css_class = "conf-low"
        label = f"Uncertain · {conf_pct}%"

    return f'<div class="conf-badge {css_class}"><span class="conf-dot"></span>{label}</div>'


def render_stat_tile(label: str, value: str, subtext: str = "", color: str = None) -> str:
    """Renders a standard bordered stat tile."""
    val_style = f'style="color: {color};"' if color else ""
    sub_html = f'<div class="stat-sub">{subtext}</div>' if subtext else ""
    return f"""
    <div class="stat-tile">
        <div class="stat-label">{label}</div>
        <div class="stat-value" {val_style}>{value}</div>
        {sub_html}
    </div>
    """


def render_safety_brake_banner(threshold: float = 0.40, current_conf: float = 0.0) -> str:
    """Renders the left-accent bordered safety brake banner."""
    shield_icon = svg_shield_alert(COLOR_CAUTION, size=22)
    return f"""
    <div class="safety-brake-banner">
        <div>{shield_icon}</div>
        <div>
            <div class="safety-brake-title">Safety Brake Activated (Confidence Below {threshold*100:.0f}%)</div>
            <div class="safety-brake-body">
                Model confidence ({current_conf*100:.1f}%) is below the Kenya agronomic safety threshold.
                Chemical and pesticide recommendations are withheld to protect crops from unnecessary toxicity.
                Expert human scouting by a Ward Agricultural Officer is advised.
            </div>
        </div>
    </div>
    """


def render_action_step(number: int, title: str, description: str, color: str = COLOR_ACCENT) -> str:
    """Renders a numbered action step card."""
    return f"""
    <div class="action-step-card">
        <div class="action-number-bubble" style="border-color: {color}; color: {color};">{number}</div>
        <div>
            <div class="action-step-title">{title}</div>
            <div class="action-step-desc">{description}</div>
        </div>
    </div>
    """


def render_model_trust_strip(model_name: str, license_str: str, artifact_ref: str) -> str:
    """Renders the model trust badge footer linking to empirical evidence."""
    check_icon = svg_checkmark(COLOR_ACCENT, size=16)
    return f"""
    <div class="model-trust-strip">
        <div style="display: flex; align-items: center; gap: 8px;">
            {check_icon}
            <span><strong>{model_name}</strong> · {license_str} · Verified Empirical Model</span>
        </div>
        <div style="font-family: var(--font-mono); font-size: 11px; color: {COLOR_MUTED};">
            Evidence: <code>{artifact_ref}</code>
        </div>
    </div>
    """
