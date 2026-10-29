"""
app.py
------
VisionMatrix – Real-Time Object Detection System
Powered by OpenCV, Ultralytics YOLOv8, and Streamlit.

Run with:
    streamlit run app.py
"""

import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import torch

# ── Local imports ─────────────────────────────────────────────────────────────
from detector import ObjectDetector, AVAILABLE_MODELS, COCO_CATEGORIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionMatrix | Real-Time YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Global High-Tech Dark CSS Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace;
}

/* ── App Background ── */
.stApp {
    background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.12) 0%, rgba(15, 23, 42, 0) 55%),
                linear-gradient(180deg, #090d16 0%, #0d1322 50%, #090d16 100%);
    color: #e2e8f0;
}

/* ── Sidebar Container & Scrollbar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1322 0%, #080c16 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 18px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
}
[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 4px;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: transparent;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 9999px;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}

/* ── Sidebar Header / Branding Card ── */
.sidebar-brand-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
}
.brand-badge-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
    border-radius: 12px;
    font-size: 1.5rem;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45);
    flex-shrink: 0;
}
.brand-text-wrapper {
    display: flex;
    flex-direction: column;
}
.brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 7px;
}
.brand-pro-tag {
    font-size: 0.62rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 6px;
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
    color: #ffffff;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}
.brand-tagline {
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 1px;
}
.brand-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.status-indicator-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #4ade80;
}
.status-pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: pulseGlow 1.8s infinite;
}
.version-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #64748b;
    background: rgba(255, 255, 255, 0.04);
    padding: 2px 6px;
    border-radius: 4px;
}

/* ── Section Dividers in Sidebar ── */
.sidebar-section-divider {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin: 16px 2px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-section-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.06);
}

/* ── Professional Navigation (Sleek SaaS Nav Cards) ── */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    width: 100% !important;
    padding: 0 !important;
}
/* Style each option as an executive card */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    margin: 0 0 4px 0 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    position: relative !important;
    box-sizing: border-box !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.28) 0%, rgba(56, 189, 248, 0.08) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.55) !important;
    border-left: 4px solid #818cf8 !important;
    box-shadow: 0 4px 18px -4px rgba(99, 102, 241, 0.35) !important;
}

/* Ensure the option wrapper is completely visible */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Hide ONLY the radio circle, never the text container */
[data-testid="stSidebar"] [data-testid="stRadio"] [class*="e1dm8mpf4"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) {
    display: none !important;
}

/* Markdown label styling - Guaranteed Visibility */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    line-height: 1.4 !important;
    letter-spacing: -0.01em !important;
    transition: color 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Sidebar Selectbox ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: rgba(99, 102, 241, 0.45) !important;
    background: rgba(22, 30, 49, 0.75) !important;
}

/* ── Sidebar Sliders (Goodbye harsh red, Hello sleek Cyan/Indigo) ── */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 10px 12px 12px;
    margin-bottom: 6px;
    transition: border-color 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stSlider"]:hover {
    border-color: rgba(99, 102, 241, 0.3);
}
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}
/* Slider value text (replaces default red) */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stThumbValue"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="rgb(255, 75, 75)"] {
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background-color: rgba(255, 255, 255, 0.08) !important;
    height: 5px !important;
    border-radius: 9999px !important;
}
/* Slider filled progress */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div,
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background-color: rgb(255, 75, 75)"] {
    background: linear-gradient(90deg, #6366f1 0%, #38bdf8 100%) !important;
    background-color: #6366f1 !important;
    border-radius: 9999px !important;
}
/* Slider thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background: #ffffff !important;
    border: 2px solid #6366f1 !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.8), 0 2px 6px rgba(0,0,0,0.4) !important;
    width: 16px !important;
    height: 16px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"]:hover {
    transform: scale(1.25) !important;
    box-shadow: 0 0 14px #38bdf8 !important;
}

/* ── Sidebar Expanders ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.2s ease !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
    background: rgba(22, 30, 49, 0.6) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    padding: 11px 14px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #818cf8 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 12px 14px !important;
    background: rgba(10, 15, 28, 0.4) !important;
}

/* ── Hero Banner ── */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 16px 40px -12px rgba(99, 102, 241, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0 0 18px;
    max-width: 780px;
    line-height: 1.5;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}

/* ── Modern Glass Card ── */
.glass-card {
    background: rgba(22, 30, 49, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px 24px;
    backdrop-filter: blur(14px);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    height: 100%;
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-3px);
    box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.25);
}

/* ── Metric Display Card ── */
.metric-box {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-box:hover {
    border-color: rgba(129, 140, 248, 0.4);
    box-shadow: 0 8px 24px -8px rgba(99, 102, 241, 0.3);
}
.metric-box .metric-val {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.metric-box .metric-lbl {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
}

/* ── Section Header ── */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.22rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 28px 0 16px;
    letter-spacing: -0.01em;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 18px;
    background: linear-gradient(180deg, #6366f1 0%, #38bdf8 100%);
    border-radius: 4px;
}

/* ── Status Pill / Badge ── */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-ready {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
}
.badge-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: pulseGlow 2s infinite;
}
@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* ── Buttons Overhaul ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #4338ca 100%);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45);
    border-color: rgba(255, 255, 255, 0.25);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    padding: 9px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
}

/* ── Modern Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    white-space: pre-wrap;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0 16px;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(79, 70, 229, 0.35) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.4);
    border: 2px dashed rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 16px;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(22, 30, 49, 0.6);
}

/* ── Crop Card in Gallery ── */
.crop-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 8px;
    text-align: center;
    transition: transform 0.2s ease;
}
.crop-card:hover {
    transform: scale(1.03);
    border-color: #818cf8;
}

/* ── Modern Notice Box ── */
.alert-box {
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin: 14px 0;
    backdrop-filter: blur(10px);
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "detector": None,
        "loaded_model_name": None,
        "webcam_running": False,
        "selected_sample_image": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar Configuration & Controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── App Logo & Sleek Brand Card ──────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="brand-badge-row">
                <div class="brand-icon-wrapper">
                    <span>🎯</span>
                </div>
                <div class="brand-text-wrapper">
                    <div class="brand-title">VisionMatrix <span class="brand-pro-tag">PRO</span></div>
                    <div class="brand-tagline">Real-Time Computer Vision</div>
                </div>
            </div>
            <div class="brand-status-bar">
                <span class="status-indicator-live">
                    <span class="status-pulse-dot"></span>
                    YOLOv8 Engine Ready
                </span>
                <span class="version-tag">v2.4</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation Section ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-divider">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "🖼️  Image Detection",
            "🎬  Video Detection",
            "📷  Webcam Detection",
            "ℹ️  Architecture & Specs",
        ],
        label_visibility="collapsed",
    )

    # ── Model & Inference Settings ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section-divider">DETECTION PIPELINE</div>', unsafe_allow_html=True)

    selected_model_label = st.selectbox(
        "Model Architecture",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Nano is blazing fast for real-time video/webcam. Medium offers highest precision.",
    )
    selected_model_file = AVAILABLE_MODELS[selected_model_label]

    # Load / reload model when selection changes
    if st.session_state.loaded_model_name != selected_model_file:
        with st.spinner(f"Loading {selected_model_file} …"):
            try:
                st.session_state.detector = ObjectDetector(selected_model_file)
                st.session_state.loaded_model_name = selected_model_file
                st.toast(f"Loaded {selected_model_file}", icon="✅")
            except Exception as exc:
                st.error(f"Model load error: {exc}")
                st.session_state.detector = None
                st.session_state.loaded_model_name = None

    # Confidence & IoU Sliders
    conf_col, iou_col = st.columns(2)
    with conf_col:
        conf_threshold = st.slider(
            "Confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum confidence score required. Lower to 0.15-0.25 to detect small objects like mouse, watch/clock, cup, keyboard.",
        )
    with iou_col:
        iou_threshold = st.slider(
            "NMS IoU",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Non-Maximum Suppression threshold to filter overlapping boxes.",
        )

    # ── Class Filtering & Styling Section ────────────────────────────────────
    st.markdown('<div class="sidebar-section-divider">FILTER & CUSTOMIZATION</div>', unsafe_allow_html=True)

    with st.expander("🎯 Target Class Filter", expanded=False):
        category_choice = st.selectbox(
            "Category Preset",
            list(COCO_CATEGORIES.keys()),
            index=0,
            help="Select a category preset to auto-filter classes.",
        )

        preset_classes = COCO_CATEGORIES[category_choice]

        if st.session_state.detector and hasattr(st.session_state.detector.model, "names"):
            all_coco_names = list(st.session_state.detector.model.names.values())
        else:
            all_coco_names = []

        default_selected = preset_classes if preset_classes is not None else all_coco_names

        selected_classes = st.multiselect(
            "Active Classes",
            options=all_coco_names,
            default=default_selected,
            help="Only objects matching the selected classes will be annotated.",
        )

        filter_active = len(selected_classes) < len(all_coco_names) if all_coco_names else False
        if filter_active:
            st.caption(f"⚡ Filtering active: {len(selected_classes)}/{len(all_coco_names)} classes enabled")

    # ── Visual Customization (UI + Feature) ───────────────────────────────────
    with st.expander("🎨 Annotation Customizer", expanded=False):
        box_thickness = st.slider("Box Thickness", min_value=1, max_value=5, value=2)
        color_palette_choice = st.selectbox(
            "Color Palette",
            options=["default", "neon", "emerald", "amber"],
            format_func=lambda x: {
                "default": "🌈 Dynamic Multi-Color",
                "neon": "⚡ Cyberpunk Neon",
                "emerald": "🌿 Emerald Green",
                "amber": "🌅 Sunset Amber",
            }[x],
        )
        c_lbl, c_cnf = st.columns(2)
        with c_lbl:
            show_labels = st.checkbox("Show Labels", value=True)
        with c_cnf:
            show_conf = st.checkbox("Show Conf %", value=True)

    st.markdown('<div class="sidebar-section-divider">HARDWARE TELEMETRY</div>', unsafe_allow_html=True)

    # ── Live Hardware Telemetry ───────────────────────────────────────────────
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f"""
        <div style="background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:12px 14px; font-size:0.77rem; box-shadow:0 4px 14px rgba(0,0,0,0.25);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                <span style="color:#94a3b8;">Compute Device</span>
                <span style="color:#38bdf8; font-weight:700; background:rgba(56,189,248,0.12); padding:1px 7px; border-radius:9999px; border:1px solid rgba(56,189,248,0.25);">{device_name}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="color:#94a3b8;">PyTorch Backend</span>
                <span style="color:#e2e8f0; font-family:'JetBrains Mono', monospace;">v{torch.__version__.split('+')[0]}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8;">OpenCV Engine</span>
                <span style="color:#e2e8f0; font-family:'JetBrains Mono', monospace;">v{cv2.__version__}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Detector reference
detector: ObjectDetector | None = st.session_state.detector

# ══════════════════════════════════════════════════════════════════════════════
#  Helper Rendering Functions
# ══════════════════════════════════════════════════════════════════════════════

def render_hero(title: str, subtitle: str, tags: list[str] = None):
    """Render the high-tech hero banner."""
    if tags is None:
        tags = ["YOLOv8 Engine", "COCO 80 Classes", "Real-Time OpenCV", "Zero GPU Required"]

    tags_html = "".join([f'<span class="hero-tag">⚡ {tag}</span>' for tag in tags])

    st.markdown(
        f"""
        <div class="hero-banner">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
            <div class="hero-tags">{tags_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(total: int, unique: int, mode_name: str, latency: str):
    """Render modern metric summary cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        col1.markdown(
            f"""<div class="metric-box">
                <div class="metric-val">{total}</div>
                <div class="metric-lbl">Total Detected</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        col2.markdown(
            f"""<div class="metric-box">
                <div class="metric-val">{unique}</div>
                <div class="metric-lbl">Unique Classes</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        col3.markdown(
            f"""<div class="metric-box">
                <div class="metric-val" style="font-size:1.6rem; padding-top:4px;">{mode_name}</div>
                <div class="metric-lbl">Inference Mode</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        col4.markdown(
            f"""<div class="metric-box">
                <div class="metric-val">{latency}</div>
                <div class="metric-lbl">Speed / Latency</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_analytics(counts: dict, detections: list[dict]):
    """Render interactive Plotly distribution charts and details."""
    if not counts:
        st.info("No detections found for the current confidence threshold.")
        return

    df_counts = (
        pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    tab_chart, tab_conf = st.tabs(["📊 Class Distribution", "📈 Confidence Breakdown"])

    with tab_chart:
        fig_bar = px.bar(
            df_counts,
            x="Class",
            y="Count",
            color="Count",
            text="Count",
            color_continuous_scale=["#38bdf8", "#6366f1", "#a855f7"],
            template="plotly_dark",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=25, b=10),
            coloraxis_showscale=False,
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_conf:
        df_conf = pd.DataFrame(detections)
        if not df_conf.empty:
            fig_hist = px.histogram(
                df_conf,
                x="confidence",
                nbins=12,
                color_discrete_sequence=["#6366f1"],
                template="plotly_dark",
                labels={"confidence": "Confidence Score"},
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=25, b=10),
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickformat=".0%"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)


def render_crop_gallery(detections: list[dict]):
    """Render visual gallery of cropped detected objects."""
    crops = [d for d in detections if "crop_rgb" in d]
    if not crops:
        return

    st.markdown('<div class="section-title">🔍 Detected Object Gallery</div>', unsafe_allow_html=True)
    st.caption("Individual cropped objects extracted by VisionMatrix with class and confidence labels.")

    # Show up to 12 crops in a grid of 6
    cols = st.columns(min(6, len(crops)))
    for idx, det in enumerate(crops[:12]):
        col = cols[idx % len(cols)]
        with col:
            col.image(det["crop_rgb"], use_container_width=True)
            col.markdown(
                f"""
                <div style="text-align:center; margin-top:-6px; margin-bottom:12px;">
                    <span style="font-size:0.75rem; font-weight:700; background:rgba(99,102,241,0.25); color:#a5b4fc; padding:2px 8px; border-radius:9999px; border:1px solid rgba(99,102,241,0.4);">
                        {det['class_name']} · {det['confidence']:.0%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def bgr_to_rgb_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def image_to_download_bytes(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame as PNG bytes."""
    success, buf = cv2.imencode(".png", frame_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG.")
    return buf.tobytes()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    render_hero(
        "VisionMatrix AI",
        "High-performance real-time computer vision system powered by Ultralytics YOLOv8 and OpenCV. Analyze webcam feeds, uploaded images, and video sequences with sub-second inference.",
    )

    # 4 Quick Action Cards
    st.markdown('<div class="section-title">🚀 Quick Start Workflows</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    workflows = [
        (c1, "🖼️", "Image Detection", "Upload custom images or try built-in sample benchmarks with instant bbox annotations.", "Upload Photo"),
        (c2, "🎬", "Video Pipeline", "Frame-by-frame video processing with live progress preview and MP4 export.", "Process Video"),
        (c3, "📷", "Webcam Suite", "Browser snapshot camera or high-speed native desktop OpenCV window.", "Open Camera"),
        (c4, "ℹ️", "System Specs", "Explore YOLO architecture, COCO 80 classes, and hardware telemetry.", "View Specs"),
    ]

    for col, icon, title, desc, action in workflows:
        col.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 10px;">{icon}</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.83rem; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">{desc}</div>
                <div style="font-size: 0.82rem; font-weight: 600; color: #818cf8;">Explore in Sidebar →</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:32px 0;'>", unsafe_allow_html=True)

    # Model Specification & Benchmark Table
    st.markdown('<div class="section-title">⚡ YOLOv8 Model Suite Comparison</div>', unsafe_allow_html=True)

    bench_df = pd.DataFrame([
        {"Model Variant": "YOLOv8 Nano (yolov8n)", "Parameters": "3.2M", "Inference Speed": "⚡ ~12-25 ms (CPU)", "mAP (val 50-95)": "37.3", "Size": "6.5 MB", "Recommended For": "Real-time webcam, live video"},
        {"Model Variant": "YOLOv8 Small (yolov8s)", "Parameters": "11.2M", "Inference Speed": "🚀 ~40-70 ms (CPU)", "mAP (val 50-95)": "44.9", "Size": "22.5 MB", "Recommended For": "Balanced speed & accuracy"},
        {"Model Variant": "YOLOv8 Medium (yolov8m)", "Parameters": "25.9M", "Inference Speed": "🎯 ~90-150 ms (CPU)", "mAP (val 50-95)": "50.2", "Size": "52.0 MB", "Recommended For": "High precision image analysis"},
    ])
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    # COCO Category Interactive Explorer
    st.markdown('<div class="section-title">📋 COCO 80-Class Taxonomy</div>', unsafe_allow_html=True)
    with st.expander("🔍 Click to browse all 80 detectable object categories", expanded=False):
        for cat_name, cat_items in COCO_CATEGORIES.items():
            if cat_items:
                st.markdown(f"**{cat_name}:**")
                pills_html = " ".join([
                    f"<span style='background:rgba(255,255,255,0.06); padding:3px 9px; border-radius:6px; font-size:0.8rem; margin:2px; display:inline-block;'>{item}</span>"
                    for item in cat_items
                ])
                st.markdown(pills_html, unsafe_allow_html=True)
                st.write("")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: IMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️  Image Detection":
    render_hero(
        "Image Object Detection",
        "Detect, localize, and classify up to 80 object types in any uploaded image or built-in test sample.",
        tags=["High-Precision Bounding Boxes", "Multi-Class Filtering", "Crop Gallery", "CSV/JSON Export"],
    )

    if not detector:
        st.error("Model engine is currently uninitialized. Please verify settings in sidebar.")
        st.stop()

    # Sample Preset Selectors
    st.markdown('<div class="section-title">📂 Choose an Input Source</div>', unsafe_allow_html=True)
    
    col_sample1, col_sample2, col_custom = st.columns([1, 1, 1.5])
    
    sample_images_dir = Path("sample_images")
    bus_sample = sample_images_dir / "bus.jpg"
    zidane_sample = sample_images_dir / "zidane.jpg"

    with col_sample1:
        if st.button("🚌 Sample 1: City Bus & Pedestrians", use_container_width=True):
            st.session_state.selected_sample_image = str(bus_sample) if bus_sample.exists() else None

    with col_sample2:
        if st.button("👥 Sample 2: Sports & People", use_container_width=True):
            st.session_state.selected_sample_image = str(zidane_sample) if zidane_sample.exists() else None

    with col_custom:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            help="Upload custom JPG/PNG image",
        )

    # Determine which image to process
    active_bgr = None
    source_title = ""

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        active_bgr = pil_to_bgr(pil_img)
        source_title = f"Uploaded File: {uploaded_file.name}"
        st.session_state.selected_sample_image = None
    elif st.session_state.selected_sample_image and Path(st.session_state.selected_sample_image).exists():
        active_bgr = cv2.imread(st.session_state.selected_sample_image)
        source_title = f"Sample Image: {Path(st.session_state.selected_sample_image).name}"

    if active_bgr is not None:
        # Run inference
        target_list = selected_classes if ('selected_classes' in locals() and len(selected_classes) < len(all_coco_names)) else None

        with st.spinner("Executing YOLOv8 inference …"):
            t_start = time.perf_counter()
            result = detector.detect_image(
                active_bgr,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                target_classes=target_list,
                box_thickness=box_thickness,
                show_labels=show_labels,
                show_conf=show_conf,
                color_palette_name=color_palette_choice,
                extract_crops=True,
            )
            total_time_ms = (time.perf_counter() - t_start) * 1000

        avg_conf_pct = (
            f"{sum(d['confidence'] for d in result['detections']) / len(result['detections']):.0%}"
            if result["detections"] else "0%"
        )

        st.success(f"🎯 Analysis complete for **{source_title}** — localized **{result['total']}** objects in **{total_time_ms:.1f} ms**")

        # Top metric cards
        render_metric_cards(
            result["total"],
            len(result["counts"]),
            "Image Analysis",
            f"{result['inference_ms']:.1f} ms",
        )

        # Side-by-Side Visual Comparison
        st.markdown('<div class="section-title">🖼️ Detection Output</div>', unsafe_allow_html=True)
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("<p style='font-weight:600; color:#94a3b8;'>Original Input Frame</p>", unsafe_allow_html=True)
            st.image(bgr_to_rgb_pil(active_bgr), use_container_width=True)
        with col_annot:
            st.markdown("<p style='font-weight:600; color:#818cf8;'>Annotated YOLOv8 Output</p>", unsafe_allow_html=True)
            st.image(bgr_to_rgb_pil(result["annotated_frame"]), use_container_width=True)

        # Crop Gallery (UI +)
        render_crop_gallery(result["detections"])

        # Analytics Charts
        st.markdown('<div class="section-title">📊 Detection Analytics & Insights</div>', unsafe_allow_html=True)
        render_analytics(result["counts"], result["detections"])

        # Detailed Table & Export
        if result["detections"]:
            st.markdown('<div class="section-title">📋 Object Detection Registry</div>', unsafe_allow_html=True)
            df_det = pd.DataFrame([
                {
                    "#": i + 1,
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Score": round(d["confidence"], 4),
                    "Bounding Box (x1, y1, x2, y2)": str(d["bbox"]),
                    "Width (px)": d["bbox"][2] - d["bbox"][0],
                    "Height (px)": d["bbox"][3] - d["bbox"][1],
                }
                for i, d in enumerate(result["detections"])
            ])
            st.dataframe(df_det, use_container_width=True, hide_index=True)

            # Export Action Suite
            st.markdown('<div class="section-title">💾 Export Data Suite</div>', unsafe_allow_html=True)
            exp_c1, exp_c2, exp_c3 = st.columns(3)

            with exp_c1:
                st.download_button(
                    "⬇️ Download Annotated Image (PNG)",
                    data=image_to_download_bytes(result["annotated_frame"]),
                    file_name="visionmatrix_annotated.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with exp_c2:
                csv_bytes = df_det.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📊 Export Detection Table (CSV)",
                    data=csv_bytes,
                    file_name="visionmatrix_detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_c3:
                # Prepare JSON clean metadata
                meta_json = {
                    "model": detector.model_name,
                    "timestamp": time.time(),
                    "total_objects": result["total"],
                    "inference_ms": result["inference_ms"],
                    "detections": [
                        {"class": d["class_name"], "confidence": d["confidence"], "bbox": d["bbox"]}
                        for d in result["detections"]
                    ],
                }
                st.download_button(
                    "📄 Export Metadata (JSON)",
                    data=json.dumps(meta_json, indent=2),
                    file_name="visionmatrix_metadata.json",
                    mime="application/json",
                    use_container_width=True,
                )
    else:
        st.markdown(
            """
            <div class="alert-box">
                💡 <b>Get Started:</b> Click one of the quick test sample buttons above (e.g. <i>City Bus & Pedestrians</i>) or upload your own image to run instant real-time detection.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  Video Detection":
    render_hero(
        "Video Stream Detection",
        "Process full video sequences frame-by-frame with real-time live preview, per-frame object counts, and exportable MP4 stream.",
        tags=["Live Frame Preview", "Frame Throughput Monitor", "Peak Object Counter", "Full MP4 Export"],
    )

    if not detector:
        st.error("Model engine is currently uninitialized.")
        st.stop()

    st.markdown('<div class="section-title">🎬 Select Video Source</div>', unsafe_allow_html=True)

    col_vsample, col_vupload = st.columns([1, 2])

    sample_video_path = Path("sample_videos/sample_traffic.mp4")
    use_sample_video = False

    with col_vsample:
        if sample_video_path.exists():
            if st.button("🚗 Use Built-in Sample Video (Traffic)", use_container_width=True):
                use_sample_video = True
        else:
            st.caption("No sample video found.")

    with col_vupload:
        uploaded_video = st.file_uploader(
            "Upload Custom Video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            help="Upload an MP4, AVI, or MOV video file",
        )

    active_video_path = None
    is_temp_file = False

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            active_video_path = tmp_in.name
            is_temp_file = True
    elif use_sample_video and sample_video_path.exists():
        active_video_path = str(sample_video_path)
        is_temp_file = False

    if active_video_path is not None:
        # Inspect video
        cap_info = cv2.VideoCapture(active_video_path)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
        vid_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / vid_fps if vid_fps > 0 else 0
        cap_info.release()

        # Telemetry columns
        st.markdown('<div class="section-title">📹 Video Telemetry & Stream Properties</div>', unsafe_allow_html=True)
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.markdown(f"""<div class="metric-box"><div class="metric-val">{total_frames}</div><div class="metric-lbl">Total Frames</div></div>""", unsafe_allow_html=True)
        vc2.markdown(f"""<div class="metric-box"><div class="metric-val">{vid_fps:.1f}</div><div class="metric-lbl">Native FPS</div></div>""", unsafe_allow_html=True)
        vc3.markdown(f"""<div class="metric-box"><div class="metric-val">{vid_w}×{vid_h}</div><div class="metric-lbl">Resolution</div></div>""", unsafe_allow_html=True)
        vc4.markdown(f"""<div class="metric-box"><div class="metric-val">{duration_sec:.1f}s</div><div class="metric-lbl">Duration</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Process Video button
        if st.button("⚡ Start Video Processing Pipeline", type="primary", use_container_width=True):
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            out_file_path = str(output_dir / f"annotated_{int(time.time())}.mp4")

            # Progress UI
            progress_bar = st.progress(0, text="Initializing YOLOv8 inference pipeline …")
            live_status = st.empty()
            live_preview = st.empty()

            target_list = selected_classes if ('selected_classes' in locals() and len(selected_classes) < len(all_coco_names)) else None

            def on_progress(p: float, curr: int, tot: int):
                pct = int(p * 100)
                progress_bar.progress(p, text=f"Processing Video Stream: {pct}% ({curr}/{tot} frames)")

            def on_frame(frame_bgr: np.ndarray, curr: int, tot: int, count: int):
                # Live frame display in real time
                rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                live_preview.image(rgb_img, caption=f"Live Preview — Frame {curr}/{tot} (Active Detections: {count})", use_container_width=True)

            try:
                t0 = time.perf_counter()
                summary = detector.process_video(
                    active_video_path,
                    out_file_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    target_classes=target_list,
                    progress_callback=on_progress,
                    frame_callback=on_frame,
                    box_thickness=box_thickness,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    color_palette_name=color_palette_choice,
                )
                wall_elapsed = time.perf_counter() - t0
                progress_bar.progress(1.0, text="✅ Video Processing Complete!")
                live_status.success(
                    f"✨ Finished processing **{summary['frames_processed']}** frames in **{wall_elapsed:.1f}s** "
                    f"({summary['avg_fps']:.1f} FPS processing throughput)"
                )

                # Summary metrics
                render_metric_cards(
                    summary["total_detections"],
                    summary["unique_classes"],
                    "Video Batch",
                    f"{summary['avg_fps']:.1f} FPS",
                )

                # Class breakdown
                render_analytics(summary["counts"], [])

                # Video Download & Player
                if Path(out_file_path).exists():
                    st.markdown('<div class="section-title">🎬 Processed Video Output</div>', unsafe_allow_html=True)
                    st.video(out_file_path)

                    with open(out_file_path, "rb") as vf:
                        v_bytes = vf.read()

                    st.download_button(
                        "⬇️ Download Processed Video (MP4)",
                        data=v_bytes,
                        file_name="visionmatrix_processed_video.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Video processing encountered an error: {e}")
            finally:
                if is_temp_file:
                    try:
                        os.unlink(active_video_path)
                    except OSError:
                        pass
    else:
        st.markdown(
            """
            <div class="alert-box">
                📌 <b>Fast Testing:</b> Click <b>'Use Built-in Sample Video (Traffic)'</b> to test the detection engine instantly, or upload any MP4 video from your computer.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: WEBCAM DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📷  Webcam Detection":
    render_hero(
        "Real-Time Webcam Detection",
        "Stream your local camera directly into VisionMatrix via browser snapshot capture or launch a high-speed native desktop OpenCV window.",
        tags=["Browser Snapshot", "OpenCV Desktop Mode", "Live HUD Overlay", "Sub-second Response"],
    )

    if not detector:
        st.error("Model engine is currently uninitialized.")
        st.stop()

    webcam_mode = st.radio(
        "Select Webcam Interface",
        [
            "🖥️ Desktop Window (OpenCV)",
            "📸 Browser Camera Snapshot",
        ],
        horizontal=True,
        key="webcam_mode_selection",
    )

    if webcam_mode == "🖥️ Desktop Window (OpenCV)":
        st.markdown('<div class="section-title">🖥️ Native High-Speed Desktop Window</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="alert-box">
                ⚡ <b>Live OpenCV Stream:</b> Opens a high-performance desktop window with live bounding boxes and FPS telemetry.
                <br>• <b>Small Object Tip:</b> To detect smaller objects (e.g. <b>computer mouse</b>, <b>watch</b>, <b>cup</b>, <b>keys</b>), keep Sensitivity at <b>0.20 – 0.25</b> and hold the object clearly facing the camera.
                <br>• Press <b>Q</b> on your keyboard inside the OpenCV window to exit.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_cam_idx, col_sens, col_start, col_stop = st.columns([1.1, 1.1, 1, 1])
        with col_cam_idx:
            selected_cam_idx = st.selectbox(
                "Camera Device Index",
                [0, 1, 2],
                index=0,
                format_func=lambda x: f"Camera {x} {'(Built-in / Default)' if x == 0 else '(USB / Secondary)'}",
                help="If camera fails to open, try Index 1 or 2.",
            )
        with col_sens:
            cam_conf = st.slider(
                "Sensitivity",
                min_value=0.05,
                max_value=0.85,
                value=min(conf_threshold, 0.25),
                step=0.05,
                help="Lower value (0.15 - 0.25) makes it easy to detect smaller objects like mouse, watch, cup, keys.",
            )
        with col_start:
            st.write("")
            st.write("")
            start_clicked = st.button("▶ Start Detection", key="cam_start", type="primary", use_container_width=True)
            if start_clicked:
                st.session_state.webcam_running = True
        with col_stop:
            st.write("")
            st.write("")
            stop_clicked = st.button("⏹ Stop Detection", key="cam_stop", use_container_width=True)
            if stop_clicked:
                st.session_state.webcam_running = False

        if st.session_state.webcam_running:
            cam_status = st.empty()
            cam_status.info("⏳ Connecting to camera device …")

            # Try DirectShow first on Windows, then fallback to default backend
            def _open_camera(idx):
                # Attempt DirectShow
                c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if c.isOpened():
                    r, f = c.read()
                    if r and f is not None:
                        return c
                    c.release()

                # Attempt default backend
                c = cv2.VideoCapture(idx)
                if c.isOpened():
                    r, f = c.read()
                    if r and f is not None:
                        return c
                    c.release()

                # Attempt Media Foundation
                c = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                if c.isOpened():
                    r, f = c.read()
                    if r and f is not None:
                        return c
                    c.release()

                return None

            cap = _open_camera(selected_cam_idx)

            if cap is None:
                cam_status.error(
                    f"❌ Camera (Index {selected_cam_idx}) could not be opened. "
                    "Make sure your webcam is plugged in, and try switching to Camera Index 1 or 2 above."
                )
                st.session_state.webcam_running = False
            else:
                cam_status.success("✅ Webcam active — OpenCV desktop window opened. Press 'Q' inside the window to exit.")
                prev_time = time.time()
                consecutive_drops = 0

                try:
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_drops += 1
                            if consecutive_drops > 20:
                                cam_status.error("❌ Lost frame stream from camera.")
                                break
                            time.sleep(0.02)
                            continue

                        consecutive_drops = 0
                        result = detector.detect_image(frame, conf_threshold=cam_conf)

                        # FPS overlay
                        curr_time = time.time()
                        fps = 1.0 / max(curr_time - prev_time, 1e-6)
                        prev_time = curr_time
                        cv2.putText(
                            result["annotated_frame"],
                            f"FPS: {fps:.1f}",
                            (frame.shape[1] - 120, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.imshow("Object Detection — Press Q to quit", result["annotated_frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            st.session_state.webcam_running = False
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    for _ in range(5):
                        cv2.waitKey(1)
                    cam_status.info("Camera stream closed.")

    else:
        # Browser snapshot mode (only rendered when selected, so it never locks camera from OpenCV!)
        st.markdown('<div class="section-title">📸 Browser Camera Snapshot</div>', unsafe_allow_html=True)
        st.caption("Click 'Take Photo' below to capture a frame through your browser and run instant YOLOv8 inference.")

        cam_pic = st.camera_input("Capture Webcam Frame", label_visibility="collapsed")

        if cam_pic is not None:
            file_bytes = np.frombuffer(cam_pic.read(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            target_list = selected_classes if ('selected_classes' in locals() and len(selected_classes) < len(all_coco_names)) else None

            with st.spinner("Processing snapshot …"):
                t_snap0 = time.perf_counter()
                result = detector.detect_image(
                    frame_bgr,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    target_classes=target_list,
                    box_thickness=box_thickness,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    color_palette_name=color_palette_choice,
                    extract_crops=True,
                )
                snap_time_ms = (time.perf_counter() - t_snap0) * 1000

            st.success(f"✅ Found **{result['total']}** object(s) in **{snap_time_ms:.1f} ms**")

            # Metrics
            render_metric_cards(
                result["total"],
                len(result["counts"]),
                "Webcam Snapshot",
                f"{result['inference_ms']:.1f} ms",
            )

            # Visual Comparison
            col_s_orig, col_s_ann = st.columns(2)
            with col_s_orig:
                st.markdown("<p style='font-weight:600; color:#94a3b8;'>Raw Webcam Capture</p>", unsafe_allow_html=True)
                st.image(bgr_to_rgb_pil(frame_bgr), use_container_width=True)
            with col_s_ann:
                st.markdown("<p style='font-weight:600; color:#818cf8;'>Annotated YOLOv8 Output</p>", unsafe_allow_html=True)
                st.image(bgr_to_rgb_pil(result["annotated_frame"]), use_container_width=True)

            # Crop Gallery
            render_crop_gallery(result["detections"])

            # Analytics
            render_analytics(result["counts"], result["detections"])

            # Download
            st.download_button(
                "⬇️ Download Annotated Snapshot (PNG)",
                data=image_to_download_bytes(result["annotated_frame"]),
                file_name="webcam_snapshot_detected.png",
                mime="image/png",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ARCHITECTURE & SPECS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  Architecture & Specs":
    render_hero(
        "System Architecture & Specifications",
        "Deep dive into the neural network architecture, inference pipeline, and runtime engineering behind VisionMatrix.",
        tags=["Ultralytics YOLOv8", "COCO Dataset", "OpenCV Engine", "Technical Documentation"],
    )

    tab_pipe, tab_tech, tab_faq = st.tabs(["🔄 Pipeline Flow", "🛠️ Technology Stack", "❓ Technical FAQ"])

    with tab_pipe:
        st.markdown('<div class="section-title">Neural Inference Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT UI LAYER (app.py)                         │
│  • Custom Glassmorphism Theme    • Real-time Telemetry Dashboard       │
│  • Interactive Class Filter      • BBox Annotation Customizer          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ BGR Frame / Video Stream
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               VISION INFERENCE ENGINE (detector.py)                    │
│  • Input Validation & Normalization                                    │
│  • Non-Maximum Suppression (NMS) with configurable IoU                 │
│  • Multi-Class Filtering & Crop Extraction                             │
│  • High-Contrast Anti-Aliased Annotation Rendering                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tensor / Batch Tensors
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              ULTRALYTICS YOLOv8 NEURAL NETWORK                         │
│  • Backbone: Modified CSPDarknet53 with C2f modules                   │
│  • Neck: Path Aggregation Network (PANet)                              │
│  • Head: Anchor-Free Decoupled Classification & Regression Heads       │
│  • Pretrained Weights: COCO (80 Classes) • Optimized for CPU/CUDA      │
└────────────────────────────────────────────────────────────────────────┘
```
            """
        )

    with tab_tech:
        st.markdown('<div class="section-title">Core Runtime Components</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                """
| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Web UI** | Streamlit | ≥1.32.0 | Reactive modern web application |
| **Object Detection** | Ultralytics YOLOv8 | ≥8.2.0 | State-of-the-art anchor-free detector |
| **Vision I/O** | OpenCV | ≥4.9.0 | Frame capture, transformation, rendering |
| **Deep Learning** | PyTorch | ≥2.0.0 | Tensor compute engine |
                """
            )
        with col_t2:
            st.markdown(
                """
| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Charts** | Plotly | ≥5.18.0 | Interactive dark analytics |
| **DataFrames** | Pandas | ≥2.0.0 | Result tables and export |
| **Imaging** | Pillow | ≥10.0.0 | Format transcoding |
| **Numerics** | NumPy | ≥1.24.0 | High-speed array operations |
                """
            )

    with tab_faq:
        st.markdown('<div class="section-title">Frequently Asked Questions</div>', unsafe_allow_html=True)
        with st.expander("How does YOLOv8 achieve real-time performance on CPU?"):
            st.write(
                "YOLOv8 introduces an anchor-free split-head architecture and optimized C2f feature aggregation "
                "modules that drastically reduce floating-point operations (FLOPs) compared to older YOLO iterations, "
                "allowing YOLOv8n (Nano) to execute at 30-60+ FPS on consumer CPUs."
            )
        with st.expander("Can I detect custom objects beyond the 80 COCO classes?"):
            st.write(
                "Yes! You can train a custom YOLOv8 model using Ultralytics with your annotated dataset in YOLO format, "
                "and drop the resulting `.pt` file into the `models/` directory."
            )
        with st.expander("How does the Target Class Filter work?"):
            st.write(
                "During inference, detections generated by the model are compared against your selected class whitelist. "
                "Unselected classes are eliminated prior to bounding box drawing and metric calculation, allowing you to focus on specific objects."
            )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:28px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
            VisionMatrix Object Detection Suite · Built with Streamlit, Ultralytics YOLOv8 & OpenCV
        </div>
        """,
        unsafe_allow_html=True,
    )
