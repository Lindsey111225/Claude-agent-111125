"""
FDA Document Intelligence Workbench
Complete Streamlit application with OCR, word graphs, agent workflows, and advanced analytics
"""

import os
import io
import time
import base64
import json
import re
import hashlib
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Core libraries
import streamlit as st
import yaml
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Image and PDF processing
from PIL import Image
import fitz  # PyMuPDF

# OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False
    
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

# NLP libraries
try:
    import yake
    YAKE_AVAILABLE = True
except:
    YAKE_AVAILABLE = False

# LLM clients
import google.generativeai as genai
from openai import OpenAI
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as grok_user, system as grok_system, image as grok_image
    GROK_AVAILABLE = True
except:
    GROK_AVAILABLE = False

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

LOCALIZATION = {
    "en": {
        "title": "🏥 FDA Document Intelligence Workbench",
        "subtitle": "Advanced Document Analysis & Multi-Agent Processing System",
        "upload": "Upload Documents",
        "paste": "Paste Text Content",
        "add_paste": "Add Pasted Text",
        "docs": "📄 Documents",
        "ocr": "🔍 OCR Processing",
        "combine": "📊 Combine & Analyze",
        "agents": "🤖 Agent Workflows",
        "dashboard": "📈 Analytics Dashboard",
        "wordgraph": "📊 Word Graph Analysis",
        "settings": "⚙️ Settings",
        "api_keys": "🔑 API Keys",
        "theme": "Theme",
        "language": "Language",
        "style": "Visual Style",
        "upload_hint": "Support: PDF, TXT, MD, CSV, JSON",
        "ocr_mode": "OCR Mode",
        "ocr_python": "Python OCR",
        "ocr_llm": "LLM OCR",
        "ocr_lang": "OCR Language",
        "run_ocr": "Run OCR",
        "preview": "Preview",
        "edit": "Edit",
        "delete": "Delete",
        "page": "Page",
        "keywords": "Keywords",
        "auto_extract": "Auto Extract",
        "generate_combined": "Generate Combined Document",
        "combined_doc": "Combined Document",
        "select_agents": "Select Agents to Run",
        "run_agent": "Execute Agent",
        "agent_output": "Agent Output",
        "metrics": "Metrics",
        "export": "Export",
        "word_freq": "Word Frequency",
        "word_cloud": "Word Cloud",
        "ngram_analysis": "N-gram Analysis",
        "entity_extraction": "Entity Extraction",
        "sentiment": "Sentiment Analysis",
        "compliance_check": "Compliance Check",
        "risk_analysis": "Risk Analysis",
        "timeline": "Timeline Analysis",
        "docs_processed": "Documents Processed",
        "pages_ocr": "Pages OCR'd",
        "tokens": "Total Tokens",
        "agent_runs": "Agent Runs",
        "processing_time": "Processing Time",
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "info": "Info",
        "gemini_key": "Gemini API Key",
        "openai_key": "OpenAI API Key",
        "grok_key": "Grok API Key",
        "apply_keys": "Apply Keys",
        "saved": "Saved successfully",
        "failed": "Operation failed",
        "loading": "Loading...",
        "batch_size": "Batch Size",
        "temperature": "Temperature",
        "max_tokens": "Max Tokens",
        "top_words": "Top Words",
        "bigrams": "Bigrams",
        "trigrams": "Trigrams",
        "co_occurrence": "Co-occurrence Network",
        "fda_features": "🔬 FDA-Specific Features",
        "adverse_events": "Adverse Event Detection",
        "drug_interactions": "Drug Interaction Analysis",
        "regulatory_compliance": "Regulatory Compliance Check"
    },
    "zh-TW": {
        "title": "🏥 FDA 文件智能工作台",
        "subtitle": "進階文件分析與多代理處理系統",
        "upload": "上傳文件",
        "paste": "貼上文字內容",
        "add_paste": "新增貼上文字",
        "docs": "📄 文件",
        "ocr": "🔍 OCR 處理",
        "combine": "📊 合併與分析",
        "agents": "🤖 代理工作流程",
        "dashboard": "📈 分析儀表板",
        "wordgraph": "📊 詞彙圖分析",
        "settings": "⚙️ 設定",
        "api_keys": "🔑 API 金鑰",
        "theme": "主題",
        "language": "語言",
        "style": "視覺風格",
        "upload_hint": "支援格式：PDF、TXT、MD、CSV、JSON",
        "ocr_mode": "OCR 模式",
        "ocr_python": "Python OCR",
        "ocr_llm": "LLM OCR",
        "ocr_lang": "OCR 語言",
        "run_ocr": "執行 OCR",
        "preview": "預覽",
        "edit": "編輯",
        "delete": "刪除",
        "page": "頁",
        "keywords": "關鍵字",
        "auto_extract": "自動擷取",
        "generate_combined": "生成合併文件",
        "combined_doc": "合併文件",
        "select_agents": "選擇要執行的代理",
        "run_agent": "執行代理",
        "agent_output": "代理輸出",
        "metrics": "指標",
        "export": "匯出",
        "word_freq": "詞頻",
        "word_cloud": "詞雲",
        "ngram_analysis": "N-gram 分析",
        "entity_extraction": "實體擷取",
        "sentiment": "情感分析",
        "compliance_check": "合規檢查",
        "risk_analysis": "風險分析",
        "timeline": "時間軸分析",
        "docs_processed": "已處理文件",
        "pages_ocr": "已 OCR 頁數",
        "tokens": "總代幣數",
        "agent_runs": "代理執行次數",
        "processing_time": "處理時間",
        "success": "成功",
        "error": "錯誤",
        "warning": "警告",
        "info": "資訊",
        "gemini_key": "Gemini API 金鑰",
        "openai_key": "OpenAI API 金鑰",
        "grok_key": "Grok API 金鑰",
        "apply_keys": "套用金鑰",
        "saved": "儲存成功",
        "failed": "操作失敗",
        "loading": "載入中...",
        "batch_size": "批次大小",
        "temperature": "溫度",
        "max_tokens": "最大代幣數",
        "top_words": "熱門詞彙",
        "bigrams": "雙詞組",
        "trigrams": "三詞組",
        "co_occurrence": "共現網絡",
        "fda_features": "🔬 FDA 專用功能",
        "adverse_events": "不良事件偵測",
        "drug_interactions": "藥物交互作用分析",
        "regulatory_compliance": "法規合規檢查"
    }
}

FLOWER_THEMES = [
    ("玫瑰石英 Rose Quartz", "#e91e63", "#ffe4ec", "#1a1a1a", "#ffffff"),
    ("薰衣草霧 Lavender Mist", "#9c27b0", "#f3e5f5", "#1a1a1a", "#ffffff"),
    ("向日葵光 Sunflower Glow", "#fbc02d", "#fff8e1", "#1a1a1a", "#ffffff"),
    ("櫻花 Cherry Blossom", "#ec407a", "#fde2ea", "#1a1a1a", "#ffffff"),
    ("蘭花綻放 Orchid Bloom", "#ab47bc", "#f4e1f7", "#1a1a1a", "#ffffff"),
    ("牡丹粉 Peony Pink", "#f06292", "#fde1ee", "#1a1a1a", "#ffffff"),
    ("鳶尾藍 Iris Indigo", "#3f51b5", "#e8eaf6", "#1a1a1a", "#ffffff"),
    ("萬壽菊 Marigold", "#ffa000", "#fff3e0", "#1a1a1a", "#ffffff"),
    ("蓮花 Lotus", "#8e24aa", "#f5e1ff", "#1a1a1a", "#ffffff"),
    ("茶花 Camellia", "#d81b60", "#fde1ea", "#1a1a1a", "#ffffff"),
    ("茉莉 Jasmine", "#43a047", "#e8f5e9", "#1a1a1a", "#ffffff"),
    ("鬱金香紅 Tulip Red", "#e53935", "#ffebee", "#1a1a1a", "#ffffff"),
    ("大麗花紫 Dahlia Plum", "#6a1b9a", "#ede7f6", "#1a1a1a", "#ffffff"),
    ("梔子花 Gardenia", "#009688", "#e0f2f1", "#1a1a1a", "#ffffff"),
    ("繡球花 Hydrangea", "#5c6bc0", "#e3e8fd", "#1a1a1a", "#ffffff"),
    ("錦葵 Lavatera", "#7b1fa2", "#f2e5ff", "#1a1a1a", "#ffffff"),
    ("櫻草 Primrose", "#f57c00", "#fff3e0", "#1a1a1a", "#ffffff"),
    ("風鈴草 Bluebell", "#1e88e5", "#e3f2fd", "#1a1a1a", "#ffffff"),
    ("木蘭 Magnolia", "#8d6e63", "#efebe9", "#1a1a1a", "#ffffff"),
    ("紫藤 Wisteria", "#7e57c2", "#ede7f6", "#1a1a1a", "#ffffff"),
]

ADVANCED_PROMPTS = {
    "ocr": """你是一位精確的 OCR 轉錄專家。請逐字轉錄文本，包括標點符號和換行。

要求：
- 目標語言：{language}
- 保留表格和程式碼區塊（使用 Markdown 表格 / ``` 區塊）
- 不要描述圖片，僅返回轉錄的文本
- 保持原始格式和結構
""",
    "agent_system": """你是一個可靠、安全且高效的專家代理。目標：
- 嚴格遵循系統和使用者指令
- 默默推理；僅返回最終答案（無思考鏈）
- 簡潔、結構化、忠實於輸入
- 避免幻覺；如果證據缺失，請說「未知」
"""
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough heuristic)"""
    return max(1, len(text) // 4)

def hash_content(content: str) -> str:
    """Generate hash for content deduplication"""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def extract_text_from_file(file) -> str:
    """Extract text from uploaded file"""
    suffix = file.name.lower().split(".")[-1]
    content = file.read()
    
    if suffix in ["txt", "md", "markdown"]:
        return content.decode("utf-8", errors="ignore")
    elif suffix == "csv":
        df = pd.read_csv(io.BytesIO(content))
        return df.to_markdown(index=False)
    elif suffix == "json":
        try:
            obj = json.loads(content.decode("utf-8", errors="ignore"))
            return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"
        except:
            return content.decode("utf-8", errors="ignore")
    elif suffix == "pdf":
        return ""  # Handled separately
    else:
        return content.decode("utf-8", errors="ignore")

def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[Dict]:
    """Convert PDF to images"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append({"page": i+1, "image": img})
    doc.close()
    return images

def img_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to bytes"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def python_ocr(image: Image.Image, engine: str = "easyocr", language: str = "en") -> str:
    """Perform Python-based OCR"""
    if engine == "tesseract" and TESSERACT_AVAILABLE:
        lang_map = {"en": "eng", "zh": "chi_tra"}
        try:
            return pytesseract.image_to_string(image, lang=lang_map.get(language, "eng"))
        except:
            pass
    
    if EASYOCR_AVAILABLE:
        lang_map = {"en": "en", "zh": "ch_tra"}
        reader = easyocr.Reader([lang_map.get(language, "en")], gpu=False)
        result = reader.readtext(np.array(image), detail=0, paragraph=True)
        return "\n".join(result)
    
    return "OCR libraries not available"

def extract_keywords_yake(text: str, max_k: int = 20, language: str = "en") -> List[str]:
    """Extract keywords using YAKE"""
    if not YAKE_AVAILABLE:
        return []
    
    lang_map = {"en": "en", "zh": "zh"}
    kw_extractor = yake.KeywordExtractor(lan=lang_map.get(language, "en"), n=1, top=max_k)
    keywords = [k for k, s in kw_extractor.extract_keywords(text)]
    return keywords

def highlight_keywords(text: str, keywords: List[str], color: str = "coral") -> str:
    """Highlight keywords in text"""
    if not keywords:
        return text
    
    for kw in sorted(set(keywords), key=len, reverse=True):
        if kw:
            pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
            text = pattern.sub(
                lambda m: f"<span style='color: {color}; font-weight: 600; background: {color}20; padding: 2px 4px; border-radius: 3px'>{m.group(0)}</span>",
                text
            )
    return text

def create_word_frequency(text: str, top_n: int = 50) -> pd.DataFrame:
    """Create word frequency dataframe"""
    # Simple tokenization
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'this', 'that', 'these', 'those', '的', '了', '在', '是', '我', '有', '和'}
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_counts = Counter(words)
    df = pd.DataFrame(word_counts.most_common(top_n), columns=['Word', 'Frequency'])
    return df

def create_ngrams(text: str, n: int = 2, top_k: int = 20) -> List[tuple]:
    """Create n-grams from text"""
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_counts = Counter([' '.join(ng) for ng in ngrams])
    return ngram_counts.most_common(top_k)

def create_cooccurrence_matrix(text: str, keywords: List[str], window: int = 5) -> pd.DataFrame:
    """Create word co-occurrence matrix"""
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())
    keywords_lower = [k.lower() for k in keywords]
    
    cooccur = defaultdict(lambda: defaultdict(int))
    
    for i, word in enumerate(words):
        if word in keywords_lower:
            for j in range(max(0, i-window), min(len(words), i+window+1)):
                if i != j and words[j] in keywords_lower:
                    cooccur[word][words[j]] += 1
    
    # Convert to dataframe
    df = pd.DataFrame(cooccur).fillna(0)
    return df

# =============================================================================
# LLM CLIENT WRAPPER
# =============================================================================

class LLMRouter:
    """Unified LLM client for multiple providers"""
    
    def __init__(self, google_key=None, openai_key=None, grok_key=None):
        self.google_key = google_key or os.getenv("GOOGLE_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.grok_key = grok_key or os.getenv("XAI_API_KEY")
        
        self._gemini = None
        self._openai = None
        self._grok = None
    
    def _init_gemini(self):
        if self._gemini is None and self.google_key:
            genai.configure(api_key=self.google_key)
            self._gemini = genai
        return self._gemini
    
    def _init_openai(self):
        if self._openai is None and self.openai_key:
            self._openai = OpenAI(api_key=self.openai_key)
        return self._openai
    
    def _init_grok(self):
        if self._grok is None and self.grok_key and GROK_AVAILABLE:
            self._grok = XAIClient(api_key=self.grok_key, timeout=3600)
        return self._grok
    
    def generate_text(self, provider: str, model: str, system_prompt: str, 
                     user_prompt: str, temperature: float = 0.2, 
                     max_tokens: int = 1500) -> str:
        """Generate text completion"""
        provider = provider.lower()
        
        try:
            if provider == "gemini":
                gem = self._init_gemini()
                if not gem:
                    raise ValueError("Gemini not configured")
                
                m = gem.GenerativeModel(model)
                parts = []
                if system_prompt:
                    parts.append({"role": "user", "parts": [f"System: {system_prompt}"]})
                parts.append({"role": "user", "parts": [user_prompt]})
                
                resp = m.generate_content(parts, generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                })
                return resp.text or ""
            
            elif provider == "openai":
                oai = self._init_openai()
                if not oai:
                    raise ValueError("OpenAI not configured")
                
                resp = oai.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return resp.choices[0].message.content
            
            elif provider == "grok":
                client = self._init_grok()
                if not client:
                    raise ValueError("Grok not configured")
                
                chat = client.chat.create(model=model)
                if system_prompt:
                    chat.append(grok_system(system_prompt))
                chat.append(grok_user(user_prompt))
                response = chat.sample()
                return response.content
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ocr_image(self, provider: str, model: str, image_bytes: bytes,
                  prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Perform LLM-based OCR"""
        provider = provider.lower()
        
        try:
            if provider == "gemini":
                gem = self._init_gemini()
                if not gem:
                    raise ValueError("Gemini not configured")
                
                m = gem.GenerativeModel(model)
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                img_part = {"inline_data": {"mime_type": "image/png", "data": b64}}
                
                resp = m.generate_content([prompt, img_part], generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                })
                return resp.text or ""
            
            elif provider == "openai":
                oai = self._init_openai()
                if not oai:
                    raise ValueError("OpenAI not configured")
                
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                resp = oai.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "You are a meticulous OCR transcriber."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ]}
                    ]
                )
                return resp.choices[0].message.content
            
            else:
                return "Provider not supported for OCR"
        
        except Exception as e:
            return f"OCR Error: {str(e)}"

# =============================================================================
# UI STYLING
# =============================================================================

def apply_theme(theme_idx: int, dark_mode: bool):
    """Apply visual theme"""
    name, primary, bg_light, text_dark, text_light = FLOWER_THEMES[theme_idx]
    
    bg_color = "#1a1a1a" if dark_mode else bg_light
    text_color = text_light if dark_mode else text_dark
    card_bg = "#2d2d2d" if dark_mode else "#ffffff"
    border_color = f"{primary}40"
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --primary: {primary};
            --bg: {bg_color};
            --text: {text_color};
            --card-bg: {card_bg};
            --border: {border_color};
        }}
        
        .stApp {{
            background: linear-gradient(135deg, {bg_color} 0%, {primary}15 100%);
            font-family: 'Inter', sans-serif;
            color: var(--text);
        }}
        
        .main-header {{
            background: linear-gradient(90deg, {primary} 0%, {primary}cc 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px {primary}30;
            color: white;
            text-align: center;
        }}
        
        .main-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        
        .main-subtitle {{
            font-size: 1.1rem;
            font-weight: 300;
            margin-top: 0.5rem;
            opacity: 0.95;
        }}
        
        .card {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            border-color: {primary};
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--card-bg) 0%, {primary}10 100%);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: scale(1.05);
            border-color: {primary};
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {primary};
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            font-weight: 500;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .tag {{
            display: inline-block;
            padding: 0.4rem 0.8rem;
            margin: 0.25rem;
            border-radius: 20px;
            background: {primary}20;
            color: {primary};
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid {primary}50;
            transition: all 0.2s ease;
        }}
        
        .tag:hover {{
            background: {primary}30;
            transform: scale(1.05);
        }}
        
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .status-success {{
            background: #4caf5020;
            color: #4caf50;
            border: 1px solid #4caf5050;
        }}
        
        .status-warning {{
            background: #ff980020;
            color: #ff9800;
            border: 1px solid #ff980050;
        }}
        
        .status-error {{
            background: #f4433620;
            color: #f44336;
            border: 1px solid #f4433650;
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(0.8); }}
        }}
        
        .stButton > button {{
            background: linear-gradient(90deg, {primary} 0%, {primary}dd 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px {primary}30;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px {primary}40;
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {primary};
            box-shadow: 0 0 0 3px {primary}20;
        }}
        
        .stSelectbox > div > div {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 8px;
        }}
        
        .plot-container {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }}
        
        .agent-workflow {{
            position: relative;
            padding-left: 2rem;
            border-left: 3px solid {primary}40;
            margin: 1rem 0;
        }}
        
        .agent-step {{
            position: relative;
            margin: 1.5rem 0;
        }}
        
        .agent-step::before {{
            content: '';
            position: absolute;
            left: -2.5rem;
            top: 50%;
            transform: translateY(-50%);
            width: 1rem;
            height: 1rem;
            border-radius: 50%;
            background: {primary};
            border: 3px solid var(--bg);
            box-shadow: 0 0 0 3px {primary}40;
        }}
        
        .expander {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            margin: 0.5rem 0;
        }}
        
        div[data-testid="stExpander"] {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
        }}
        
        .sidebar .sidebar-content {{
            background: var(--card-bg);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text);
            font-weight: 600;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background: {primary}10;
            border-color: {primary};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {primary};
            color: white;
            border-color: {primary};
        }}
        </style>
    """, unsafe_allow_html=True)

def render_header(T: dict, theme_name: str):
    """Render main header"""
    st.markdown(f"""
        <div class="main-header">
            <div class="main-title">{T['title']}</div>
            <div class="main-subtitle">{T['subtitle']}</div>
            <div style="margin-top: 1rem;">
                <span class="tag">{theme_name}</span>
                <span class="tag">v2.0</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: any, icon: str = "📊"):
    """Render metric card"""
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

def render_status(status: str, message: str):
    """Render status indicator"""
    status_map = {
        "success": ("✓", "status-success"),
        "warning": ("⚠", "status-warning"),
        "error": ("✗", "status-error"),
        "info": ("ℹ", "status-success")
    }
    icon, css_class = status_map.get(status, ("•", "status-success"))
    
    st.markdown(f"""
        <div class="status-indicator {css_class}">
            <span class="status-dot"></span>
            <span>{icon} {message}</span>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "docs": [],
        "ocr_results": {},
        "combined_doc": "",
        "keywords": [],
        "agents": [],
        "agents_yaml": "",
        "agent_results": [],
        "metrics": {
            "docs_processed": 0,
            "pages_ocr": 0,
            "total_tokens": 0,
            "agent_runs": 0,
            "processing_times": []
        },
        "api_keys": {
            "gemini": None,
            "openai": None,
            "grok": None
        },
        "settings": {
            "lang": "zh-TW",
            "theme_idx": 0,
            "dark_mode": True,
            "ocr_engine": "easyocr",
            "ocr_language": "zh",
            "default_temperature": 0.2,
            "default_max_tokens": 1500
        },
        "word_analysis": {
            "word_freq": None,
            "bigrams": None,
            "trigrams": None,
            "cooccurrence": None
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="FDA Document Intelligence",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar settings
    with st.sidebar:
        st.title("⚙️ " + LOCALIZATION[st.session_state.settings["lang"]]["settings"])
        
        # Language selection
        lang = st.selectbox(
            "🌐 Language / 語言",
            ["en", "zh-TW"],
            index=0 if st.session_state.settings["lang"] == "en" else 1,
            key="lang_select"
        )
        st.session_state.settings["lang"] = lang
        T = LOCALIZATION[lang]
        
        # Theme selection
        st.subheader(T["theme"])
        theme_idx = st.selectbox(
            T["style"],
            range(len(FLOWER_THEMES)),
            format_func=lambda i: FLOWER_THEMES[i][0],
            index=st.session_state.settings["theme_idx"]
        )
        st.session_state.settings["theme_idx"] = theme_idx
        
        dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.settings["dark_mode"])
        st.session_state.settings["dark_mode"] = dark_mode
        
        # API Keys
        st.subheader(T["api_keys"])
        
        env_gemini = os.getenv("GOOGLE_API_KEY")
        env_openai = os.getenv("OPENAI_API_KEY")
        env_grok = os.getenv("XAI_API_KEY")
        
        gemini_key = st.text_input(
            T["gemini_key"],
            type="password",
            value="" if not env_gemini else "••••••••",
            disabled=bool(env_gemini)
        )
        
        openai_key = st.text_input(
            T["openai_key"],
            type="password",
            value="" if not env_openai else "••••••••",
            disabled=bool(env_openai)
        )
        
        grok_key = st.text_input(
            T["grok_key"],
            type="password",
            value="" if not env_grok else "••••••••",
            disabled=bool(env_grok)
        )
        
        if st.button(T["apply_keys"], use_container_width=True):
            st.session_state.api_keys["gemini"] = gemini_key or env_gemini
            st.session_state.api_keys["openai"] = openai_key or env_openai
            st.session_state.api_keys["grok"] = grok_key or env_grok
            st.success(T["saved"])
        else:
            st.session_state.api_keys["gemini"] = st.session_state.api_keys["gemini"] or env_gemini
            st.session_state.api_keys["openai"] = st.session_state.api_keys["openai"] or env_openai
            st.session_state.api_keys["grok"] = st.session_state.api_keys["grok"] or env_grok
        
        # API Status
        st.markdown("---")
        st.markdown("**API Status:**")
        for name, key in [
            ("Gemini", st.session_state.api_keys["gemini"]),
            ("OpenAI", st.session_state.api_keys["openai"]),
            ("Grok", st.session_state.api_keys["grok"])
        ]:
            status = "✓" if key else "✗"
            color = "#4caf50" if key else "#f44336"
            st.markdown(f"<span style='color: {color}'>{status} {name}</span>", unsafe_allow_html=True)
    
    # Apply theme
    apply_theme(theme_idx, dark_mode)
    
    # Main header
    theme_name = FLOWER_THEMES[theme_idx][0]
    render_header(T, theme_name)
    
    # Main tabs
    tabs = st.tabs([
        T["docs"],
        T["ocr"],
        T["combine"],
        T["wordgraph"],
        T["agents"],
        T["dashboard"]
    ])
    
    # Tab 1: Documents
    with tabs[0]:
        render_documents_tab(T)
    
    # Tab 2: OCR
    with tabs[1]:
        render_ocr_tab(T)
    
    # Tab 3: Combine & Analyze
    with tabs[2]:
        render_combine_tab(T)
    
    # Tab 4: Word Graph Analysis
    with tabs[3]:
        render_wordgraph_tab(T)
    
    # Tab 5: Agents
    with tabs[4]:
        render_agents_tab(T)
    
    # Tab 6: Dashboard
    with tabs[5]:
        render_dashboard_tab(T)

# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_documents_tab(T: dict):
    """Render documents management tab"""
    st.subheader(T["upload"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            T["upload_hint"],
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                doc_id = f"{file.name}_{hash_content(file.name)}_{int(time.time())}"
                
                # Check if already added
                if any(d["id"] == doc_id for d in st.session_state.docs):
                    continue
                
                ext = file.name.lower().split(".")[-1]
                doc = {
                    "id": doc_id,
                    "name": file.name,
                    "type": ext,
                    "timestamp": datetime.now().isoformat(),
                    "content": "",
                    "pdf_bytes": None,
                    "images": None
                }
                
                if ext == "pdf":
                    doc["pdf_bytes"] = file.read()
                else:
                    doc["content"] = extract_text_from_file(file)
                
                st.session_state.docs.append(doc)
                st.session_state.metrics["docs_processed"] = len(st.session_state.docs)
            
            render_status("success", f"Added {len(uploaded_files)} document(s)")
    
    with col2:
        st.markdown("### 📝 " + T["paste"])
        paste_text = st.text_area(T["paste"], height=200, key="paste_input")
        
        if st.button(T["add_paste"], use_container_width=True):
            if paste_text.strip():
                doc_id = f"paste_{hash_content(paste_text)}_{int(time.time())}"
                doc = {
                    "id": doc_id,
                    "name": f"Pasted Text {len(st.session_state.docs)+1}",
                    "type": "txt",
                    "timestamp": datetime.now().isoformat(),
                    "content": paste_text,
                    "pdf_bytes": None,
                    "images": None
                }
                st.session_state.docs.append(doc)
                st.session_state.metrics["docs_processed"] = len(st.session_state.docs)
                render_status("success", "Pasted text added")
    
    # Document list
    st.markdown("---")
    st.subheader(f"📚 {T['docs']} ({len(st.session_state.docs)})")
    
    for idx, doc in enumerate(st.session_state.docs):
        with st.expander(f"📄 {doc['name']}", expanded=False):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**Type:** {doc['type'].upper()}")
                st.markdown(f"**Added:** {doc['timestamp'][:19]}")
            
            with col2:
                if doc["type"] == "pdf" and doc["pdf_bytes"]:
                    if st.button(T["preview"], key=f"preview_{doc['id']}"):
                        if doc["images"] is None:
                            with st.spinner(T["loading"]):
                                doc["images"] = pdf_to_images(doc["pdf_bytes"])
                        render_status("success", f"Rendered {len(doc['images'])} pages")
            
            with col3:
                if st.button(T["delete"], key=f"delete_{doc['id']}", type="secondary"):
                    st.session_state.docs.pop(idx)
                    st.session_state.metrics["docs_processed"] = len(st.session_state.docs)
                    st.rerun()
            
            # Show content or images
            if doc["type"] == "pdf" and doc.get("images"):
                cols = st.columns(4)
                for i, page_data in enumerate(doc["images"][:8]):  # Show first 8 pages
                    with cols[i % 4]:
                        st.image(page_data["image"], caption=f"{T['page']} {page_data['page']}", use_container_width=True)
            elif doc["content"]:
                content = st.text_area(
                    T["edit"],
                    value=doc["content"],
                    height=200,
                    key=f"edit_{doc['id']}"
                )
                doc["content"] = content

def render_ocr_tab(T: dict):
    """Render OCR processing tab"""
    st.subheader(T["ocr"])
    
    pdf_docs = [d for d in st.session_state.docs if d["type"] == "pdf"]
    
    if not pdf_docs:
        st.info("📄 Please upload PDF documents in the Documents tab first")
        return
    
    for doc in pdf_docs:
        with st.expander(f"📄 {doc['name']}", expanded=True):
            # Render pages if not done
            if doc["images"] is None and doc["pdf_bytes"]:
                if st.button(f"🖼️ Render Pages", key=f"render_{doc['id']}"):
                    with st.spinner(T["loading"]):
                        doc["images"] = pdf_to_images(doc["pdf_bytes"])
                    render_status("success", f"Rendered {len(doc['images'])} pages")
            
            if doc.get("images"):
                # OCR settings
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ocr_mode = st.radio(
                        T["ocr_mode"],
                        [T["ocr_python"], T["ocr_llm"]],
                        key=f"ocr_mode_{doc['id']}"
                    )
                
                with col2:
                    ocr_lang = st.selectbox(
                        T["ocr_lang"],
                        ["English", "Traditional Chinese"],
                        index=1 if st.session_state.settings["ocr_language"] == "zh" else 0,
                        key=f"ocr_lang_{doc['id']}"
                    )
                
                with col3:
                    if ocr_mode == T["ocr_llm"]:
                        llm_model = st.selectbox(
                            "LLM Model",
                            [
                                "gemini:gemini-2.5-flash",
                                "gemini:gemini-2.5-flash-lite",
                                "openai:gpt-4o-mini",
                                "openai:gpt-4-turbo"
                            ],
                            key=f"llm_model_{doc['id']}"
                        )
                
                # Page selection
                page_nums = [p["page"] for p in doc["images"]]
                selected_pages = st.multiselect(
                    "Select pages for OCR",
                    page_nums,
                    default=page_nums[:min(5, len(page_nums))],
                    key=f"pages_{doc['id']}"
                )
                
                # Run OCR
                if st.button(T["run_ocr"], key=f"run_ocr_{doc['id']}", type="primary"):
                    lang_code = "zh" if "Chinese" in ocr_lang else "en"
                    
                    with st.status("🔍 Processing OCR...", expanded=True) as status:
                        start_time = time.time()
                        router = LLMRouter(
                            google_key=st.session_state.api_keys["gemini"],
                            openai_key=st.session_state.api_keys["openai"],
                            grok_key=st.session_state.api_keys["grok"]
                        )
                        
                        for page_data in doc["images"]:
                            if page_data["page"] not in selected_pages:
                                continue
                            
                            st.write(f"Processing page {page_data['page']}...")
                            
                            if ocr_mode == T["ocr_python"]:
                                text = python_ocr(
                                    page_data["image"],
                                    engine=st.session_state.settings["ocr_engine"],
                                    language=lang_code
                                )
                            else:
                                provider, model = llm_model.split(":")
                                prompt = ADVANCED_PROMPTS["ocr"].format(
                                    language="Traditional Chinese" if lang_code == "zh" else "English"
                                )
                                image_bytes = img_to_bytes(page_data["image"])
                                text = router.ocr_image(provider, model, image_bytes, prompt)
                            
                            st.session_state.ocr_results[(doc["id"], page_data["page"])] = text
                        
                        elapsed = time.time() - start_time
                        st.session_state.metrics["pages_ocr"] += len(selected_pages)
                        st.session_state.metrics["processing_times"].append(elapsed)
                        
                        status.update(label="✓ OCR Complete", state="complete")
                        render_status("success", f"Processed {len(selected_pages)} pages in {elapsed:.2f}s")
                
                # Show OCR results
                if any((doc["id"], p) in st.session_state.ocr_results for p in page_nums):
                    st.markdown("### OCR Results")
                    for page_num in selected_pages:
                        key = (doc["id"], page_num)
                        if key in st.session_state.ocr_results:
                            text = st.text_area(
                                f"{T['page']} {page_num}",
                                value=st.session_state.ocr_results[key],
                                height=200,
                                key=f"ocr_result_{doc['id']}_{page_num}"
                            )
                            st.session_state.ocr_results[key] = text

def render_combine_tab(T: dict):
    """Render combine and analyze tab"""
    st.subheader(T["combine"])
    
    # Build combined document
    combined_parts = []
    
    for doc in st.session_state.docs:
        if doc["type"] == "pdf":
            # Collect OCR results
            ocr_texts = []
            if doc.get("images"):
                for page_data in doc["images"]:
                    key = (doc["id"], page_data["page"])
                    if key in st.session_state.ocr_results:
                        ocr_texts.append(f"### {T['page']} {page_data['page']}\n\n{st.session_state.ocr_results[key]}")
            
            if ocr_texts:
                combined_parts.append(f"## {doc['name']}\n\n" + "\n\n".join(ocr_texts))
        else:
            if doc["content"]:
                combined_parts.append(f"## {doc['name']}\n\n{doc['content']}")
    
    # Keyword extraction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(T["auto_extract"], use_container_width=True):
            full_text = "\n\n".join(combined_parts)
            lang = "zh" if st.session_state.settings["lang"] == "zh-TW" else "en"
            keywords = extract_keywords_yake(full_text, max_k=30, language=lang)
            st.session_state.keywords = keywords
            render_status("success", f"Extracted {len(keywords)} keywords")
    
    with col2:
        if st.button(T["generate_combined"], type="primary", use_container_width=True):
            combined_text = "\n\n---\n\n".join(combined_parts)
            
            # Highlight keywords
            if st.session_state.keywords:
                theme_color = FLOWER_THEMES[st.session_state.settings["theme_idx"]][1]
                combined_text = highlight_keywords(combined_text, st.session_state.keywords, theme_color)
            
            st.session_state.combined_doc = combined_text
            st.session_state.metrics["total_tokens"] = estimate_tokens(combined_text)
            st.balloons()
            render_status("success", "Combined document generated")
    
    # Show/edit keywords
    if st.session_state.keywords or st.session_state.combined_doc:
        st.markdown("### " + T["keywords"])
        keywords_text = st.text_area(
            "Edit keywords (one per line)",
            value="\n".join(st.session_state.keywords),
            height=150
        )
        st.session_state.keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]
        
        # Display as tags
        if st.session_state.keywords:
            tags_html = "".join([f"<span class='tag'>{kw}</span>" for kw in st.session_state.keywords[:20]])
            st.markdown(tags_html, unsafe_allow_html=True)
    
    # Display combined document
    if st.session_state.combined_doc:
        st.markdown("---")
        st.markdown("### " + T["combined_doc"])
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(st.session_state.combined_doc, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 Download Markdown",
                data=st.session_state.combined_doc,
                file_name=f"combined_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def render_wordgraph_tab(T: dict):
    """Render word graph analysis tab"""
    st.subheader(T["wordgraph"])
    
    if not st.session_state.combined_doc:
        st.info("Please generate a combined document first in the Combine tab")
        return
    
    # Clean text (remove HTML tags)
    clean_text = re.sub(r'<[^>]+>', '', st.session_state.combined_doc)
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Number of top words", 10, 100, 30)
    
    with col2:
        ngram_size = st.selectbox("N-gram size", [2, 3, 4], index=0)
    
    with col3:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    # Word Frequency Analysis
    st.markdown("### " + T["word_freq"])
    word_freq_df = create_word_frequency(clean_text, top_n=top_n)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            word_freq_df.head(20),
            x='Frequency',
            y='Word',
            orientation='h',
            title=f'Top 20 {T["top_words"]}',
            color='Frequency',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(word_freq_df, height=600, use_container_width=True)
    
    # N-gram Analysis
    st.markdown(f"### {T['ngram_analysis']} ({ngram_size}-gram)")
    ngrams = create_ngrams(clean_text, n=ngram_size, top_k=20)
    
    if ngrams:
        ngram_df = pd.DataFrame(ngrams, columns=['N-gram', 'Frequency'])
        
        fig = px.bar(
            ngram_df,
            x='Frequency',
            y='N-gram',
            orientation='h',
            title=f'Top 20 {ngram_size}-grams',
            color='Frequency',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Co-occurrence Network
    if st.session_state.keywords:
        st.markdown("### " + T["co_occurrence"])
        
        cooccur_df = create_cooccurrence_matrix(clean_text, st.session_state.keywords, window=10)
        
        if not cooccur_df.empty:
            # Create network graph
            edges = []
            for col in cooccur_df.columns:
                for idx in cooccur_df.index:
                    weight = cooccur_df.loc[idx, col]
                    if weight > 0 and col != idx:
                        edges.append((idx, col, weight))
            
            # Sort by weight and take top connections
            edges.sort(key=lambda x: x[2], reverse=True)
            edges = edges[:50]  # Top 50 connections
            
            # Create network visualization
            import plotly.graph_objects as go
            
            # Build node positions (circular layout)
            nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
            n = len(nodes)
            
            node_positions = {}
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / n
                node_positions[node] = (np.cos(angle), np.sin(angle))
            
            # Create edges
            edge_trace = []
            for source, target, weight in edges:
                x0, y0 = node_positions[source]
                x1, y1 = node_positions[target]
                
                trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight/max(e[2] for e in edges)*5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_trace.append(trace)
            
            # Create nodes
            node_x = [node_positions[node][0] for node in nodes]
            node_y = [node_positions[node][1] for node in nodes]
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=nodes,
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=20,
                    color=FLOWER_THEMES[st.session_state.settings["theme_idx"]][1],
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace])
            fig.update_layout(
                title='Keyword Co-occurrence Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Export word analysis
    st.markdown("---")
    if st.button("📥 Export Word Analysis", use_container_width=True):
        export_data = {
            "word_frequency": word_freq_df.to_dict('records'),
            "ngrams": [{"ngram": ng, "frequency": freq} for ng, freq in ngrams],
            "keywords": st.session_state.keywords,
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"word_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ) > div > div > input,
        .stTextArea > div > div > textarea {{
            background: var(--card-bg);
            border: 2px solid var(--border);
            border-radius: 8px;
            color: var(--text);
        }}
        
        .stTextInput
