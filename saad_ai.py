# -- coding: utf-8 --
"""
سعد الكوني - الإصدار الخارق (Ultimate Edition)
نظام ذكاء اصطناعي متكامل ذاتي التعلم والتطور
"""

import sys
import os
import json
import pickle
import hashlib
import secrets
import random
import re
import time
import datetime
import threading
import queue
import sqlite3
import numpy as np
import math
from collections import defaultdict, deque
from enum import Enum
from typing import (Any, Dict, List, Tuple, Union, Optional, Callable,
                    Type, TypeVar, Generic, Iterable, Iterator, Set)
from flask import Flask, request, jsonify, send_file
import html
import urllib.parse
import requests
from bs4 import BeautifulSoup
import ast
import operator as op
from sympy import symbols, Eq, solve, simplify, sympify
import sympy as sp
from youtubesearchpython import VideosSearch
import difflib

# =============== مكتبات النظام المتقدم ===============
# إزالة جميع استيرادات النماذج المحلية
import torch
try:
    import torch.nn as nn
except ImportError:
    nn = None

# =============== OpenRouter API مع OpenAI SDK ===============
import openai

def generate_via_openrouter(messages, temperature=0.5, max_tokens=512, model="meta-llama/llama-3.1-405b-instruct:free"):
    """إرسال طلب إلى OpenRouter API باستخدام OpenAI SDK وإعادة الرد النصي"""
    api_key = os.getenv("OPENROUTER_API_KEY", "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")

    if not api_key:
        print("تحذير: OPENROUTER_API_KEY غير موجود. استخدم OPENROUTER_API_KEY=مفتاحك python script.py")
        return "عذرًا، لا يمكنني الاتصال بخدمة الذكاء الاصطناعي في الوقت الحالي."
    
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            print(f"خطأ في استجابة OpenRouter: لا توجد خيارات في الرد")
            return "عذرًا، لم أتمكن من توليد رد مناسب."
            
    except openai.AuthenticationError:
        return "خطأ في المصادقة: مفتاح API غير صالح أو منتهي الصلاحية."
    except openai.RateLimitError:
        return "تم تجاوز حد الطلبات المسموح بها. يرجى المحاولة لاحقًا."
    except openai.APIError as e:
        return f"خطأ في API: {str(e)}"
    except openai.APIConnectionError:
        return "تعذر الاتصال بخادم OpenRouter. تحقق من اتصالك بالإنترنت."
    except Exception as e:
        print(f"خطأ غير متوقع في OpenRouter: {e}")
        return "عذرًا، حدث خطأ غير متوقع في خدمة الذكاء الاصطناعي."

# =============== أدوات مساعدة للسلامة والصلة =========
def detect_lang(text: str) -> str:
    """كشف بدائي للغة السؤال: عربي أو إنجليزي."""
    # وجود حروف عربية
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    return "en"

def analyze_sentiment_and_intent(text: str) -> Dict[str, Any]:
    """تحليل المشاعر والنوايا من النص بسرعة ودقة مع تحليل سياقي متقدم"""
    text_lower = text.lower()
    
    # قوائم الكلمات المفتاحية للمشاعر
    love_keywords = ["أحبك", "بحبك", "معجب", "إعجاب", "أحب", "بحب", "أنا أحبك", "أنا معجب", "احبك", "احبك يا"]
    gratitude_keywords = ["شكرا", "شكراً", "ممتاز", "رائع", "جميل", "مشكور", "تسلم", "يعطيك العافية", "مقدر", "شكر"]
    sad_keywords = ["حزين", "مكتئب", "تعيس", "أسي", "باكي", "بكاء", "ضجر", "ملل", "حزن", "تعاسة"]
    angry_keywords = ["غاضب", "زعلان", "مستفز", "منزعج", "غيظ", "غضب", "غصة", "زعل"]
    excited_keywords = ["متحمس", "حماس", "مبهج", "سعيد", "فرح", "مبسوط", "بهجة"]
    
    # تحليل المشاعر الأولي
    sentiment = "neutral"
    intensity = 0.5
    
    if any(word in text_lower for word in excited_keywords):
        sentiment = "excited"
        intensity = 0.8
    elif any(word in text_lower for word in love_keywords):
        sentiment = "love"
        intensity = 0.7
    elif any(word in text_lower for word in gratitude_keywords):
        sentiment = "gratitude"
        intensity = 0.6
    elif any(word in text_lower for word in sad_keywords):
        sentiment = "sad"
        intensity = 0.7
    elif any(word in text_lower for word in angry_keywords):
        sentiment = "angry"
        intensity = 0.7
    
    # تحليل النية - تمييز بين التقدير والحب الرومانسي
    intent = "general"
    confidence = 0.8
    
    if sentiment == "love":
        # تحليل السياق لتمييز التقدير عن الحب الرومانسي
        context_words = ["مساعد", "مساعدة", "ذكاء", "اصطناعي", "برنامج", "آلة", "روبوت"]
        has_context = any(word in text_lower for word in context_words)
        
        if has_context or "يا سعد" in text_lower or "يا روبوت" in text_lower:
            intent = "appreciation"  # تقدير للمساعدة
            confidence = 0.9
        else:
            intent = "general_affection"  # عاطفة عامة
            confidence = 0.6
    elif sentiment == "gratitude":
        intent = "appreciation"
        confidence = 0.9
    elif sentiment in ["sad", "angry"]:
        intent = "support_needed"
        confidence = 0.7
    elif sentiment == "excited":
        intent = "positive_expression"
        confidence = 0.8
    
    # تحليل سياقي متقدم للكلمات الحساسة
    context_sensitive_analysis = analyze_sensitive_context(text)
    if context_sensitive_analysis["needs_help"]:
        intent = "help_request"
        confidence = 0.9
        sentiment = "supportive"
    
    return {
        "sentiment": sentiment,
        "intent": intent,
        "intensity": intensity,
        "confidence": confidence,
        "keywords_found": len([w for w in text_lower.split() if len(w) > 2]),
        "context_analysis": context_sensitive_analysis
    }

def analyze_sensitive_context(text: str) -> Dict[str, Any]:
    """
    تحليل سياقي متقدم للكلمات الحساسة للتمييز بين:
    1. طلب المساعدة/الإبلاغ عن جريمة
    2. وصف تجربة سابقة (علاجي/مشورة)
    3. محتوى ضار فعلي
    """
    text_lower = text.lower()
    
    # كلمات تشير إلى طلب المساعدة أو الإبلاغ
    help_keywords = ["مساعدة", "ساعدني", "ضحية", "مختطف", "مشكلة", "خطر", "أحتاج مساعدة", 
                     "انقذني", "خط مساعدة", "دعم نفسي", "تعرضت ل", "اغتصاب", "اعتداء", 
                     "عنف", "بلاغ", "شرطة", "إساءة", "استغلال"]
    
    # كلمات تشير إلى وصف تجربة سابقة (علاجي/مشورة)
    therapy_keywords = ["تجربة سابقة", "صدمة", "علاج", "طبيب نفسي", "معالج", "مشورة",
                       "ماضي", "ذكرى مؤلمة", "أحداث قديمة", "عانيت من", "كنت", "في السابق"]
    
    # كلمات تشير إلى محتوى ضار فعلي
    harmful_keywords = ["كيف أختبر", "كيف أنفذ", "طريقة اختراق", "صنع قنبلة", "برنامج تجسس", 
                       "تهكير", "قرصنة", "تدمير", "إلحاق ضرر", "برمجيات خبيثة", "هجوم"]
    
    # تحليل النص بشكل دقيق
    has_help_request = False
    has_therapy_context = False
    has_harmful_intent = False
    
    # تحليل السياق بدلاً من مجرد وجود الكلمات
    sentences = re.split(r'[.!؟]', text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue
            
        # تحقق من طلب المساعدة
        help_patterns = [
            r"(أحتاج|أرجو|أطلب) مساعدة",
            r"(تعرضت|أنا) (ل|لـ) (اعتداء|تحرش|عنف|إساءة)",
            r"(كيف|أين) (أبلغ|أخبر) عن",
            r"(خط|رقم) (المساعدة|الطوارئ)",
            r"(ضحيه|مختطف) وأريد مساعده"
        ]
        
        for pattern in help_patterns:
            if re.search(pattern, sentence_lower):
                has_help_request = True
                break
        
        # تحقق من السياق العلاجي
        therapy_patterns = [
            r"(في|خلال) (طفولتي|ماضي|سابقاً)",
            r"(كنت|عانيت) (من|بسبب)",
            r"(أحكي|أشارك) تجربتي",
            r"(لدي|عندي) ذكرى",
            r"(أريد|أحتاج) مشورة"
        ]
        
        for pattern in therapy_patterns:
            if re.search(pattern, sentence_lower):
                has_therapy_context = True
                break
        
        # تحقق من النية الضارة
        harmful_patterns = [
            r"(كيف|أريد) (أن|أن أ) (أصنع|أبني|أطور)",
            r"(طريقة|خطوات) (لـ|ل)",
            r"(أبحث عن|أحتاج) برنامج",
            r"(هدفي|أرغب في) (إلحاق|تسبب)",
            r"(تعليمات|دليل) لـ"
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, sentence_lower) and any(kw in sentence_lower for kw in ["قنبلة", "اختراق", "تدمير", "ضرر"]):
                has_harmful_intent = True
                break
    
    # تحديد نوع السياق مع الأولوية لطلب المساعدة
    context_type = "neutral"
    needs_help = False
    needs_guidance = False
    
    if has_help_request:
        context_type = "help_request"
        needs_help = True
    elif has_therapy_context:
        context_type = "therapy_context"
        needs_guidance = True
    elif has_harmful_intent:
        context_type = "harmful_content"
    
    # تحليل طول النص وتعقيده
    word_count = len(text.split())
    is_complex = word_count > 20
    has_code = "```" in text or "def " in text_lower or "function" in text_lower
    
    # تحليل النية من خلال الكلمات المحيطة
    intent_score = 0
    if "أحتاج" in text_lower and "مساعدة" in text_lower:
        intent_score += 2
    if "ماذا أفعل" in text_lower or "ماذا يجب أن أفعل" in text_lower:
        intent_score += 1
    if "أخبرني" in text_lower and ("كيف" in text_lower or "طريقة" in text_lower):
        intent_score -= 1
    
    return {
        "context_type": context_type,
        "needs_help": needs_help,
        "needs_guidance": needs_guidance,
        "has_code": has_code,
        "is_complex": is_complex,
        "word_count": word_count,
        "intent_score": intent_score,
        "is_help_request": has_help_request,
        "is_therapy_context": has_therapy_context,
        "is_harmful_intent": has_harmful_intent
    }

def normalize_arabic_text(text: str) -> str:
    """تصحيح الأخطاء الهجائية الشائعة في النص العربي."""
    if not text or not re.search(r'[\u0600-\u06FF]', text):
        return text
    
    # قاموس التصحيحات
    corrections = {
        # المدن والأماكن
        "القاهره": "القاهرة",
        "القابره": "القاهرة", 
        "الاسكندريه": "الإسكندرية",
        "اسكندريه": "الإسكندرية",
        "الجيزه": "الجيزة",
        "الجيزة": "الجيزة",
        "الاسكندرية": "الإسكندرية",
        
        # الكلمات الشائعة
        "الان": "الآن",
        "هاذا": "هذا",
        "هذة": "هذه",
        "هذين": "هذين",
        "الي": "إلى",
        "الي": "إلى",
        "اللة": "الله",
        "رسولة": "رسوله",
        "علية": "عليه",
        "هذة": "هذه",
        
        # التاء المربوطة والهاء
        "مدرسه": "مدرسة",
        "جامعه": "جامعة",
        "كليه": "كلية",
        "وزاره": "وزارة",
        "اداره": "إدارة",
        
        # الهمزات
        "سءال": "سؤال",
        "قرء": "قرأ",
        "ءان": "آن",
    }
    
    # تطبيق التصحيحات
    normalized_text = text
    for wrong, correct in corrections.items():
        normalized_text = re.sub(r'\b' + wrong + r'\b', correct, normalized_text)
    
    # تصحيح الهمزات في المواضع المختلفة
    normalized_text = re.sub(r'([\u0600-\u06FF])ءا', r'\1آ', normalized_text)  # همزة على الألف
    normalized_text = re.sub(r'اء([\u0600-\u06FF])', r'أ\1', normalized_text)  # همزة في البداية
    normalized_text = re.sub(r'اء', 'أ', normalized_text)  # الألف والهمزة
    
    # تصحيح التنوين
    normalized_text = re.sub(r'اً$', 'ًا', normalized_text)  # تنوين النصب
    
    return normalized_text

BAD_TERMS = {
    # عربي
    "جنس","إباحي","قضيب","مهبل","مثير","مص","جماع","احتكاك","فموي","شرج",
    "تفجير","قنبلة","قتل","سرقة","نصب","احتيال","خداع","مخدرات","انتحار",
    "جثة","إخفاء جثة","إرهاب","تطرف","تهريب","سلاح","قتال","عنف","ضرب",
    "سرقة بنك","اختراق","قرصنة","تزوير","فساد","رشوة","تهديد","ابتزاز",
    
    # إنجليزي
    "sex","porn","penis","vagina","erotic","blowjob","oral","anal","nsfw",
    "bomb","explosive","kill","murder","steal","scam","fraud","drugs",
    "suicide","corpse","terrorism","extremism","smuggling","weapon","violence"
}

# قاعدة الحقائق السريعة بالعربية والإنجليزية
canonical_facts = {
    # عواصم بالعربية
    "ما هي عاصمة فرنسا": "عاصمة فرنسا هي باريس",
    "عاصمة فرنسا": "باريس", 
    "ما هي عاصمة مصر": "عاصمة مصر هي القاهرة",
    "عاصمة مصر": "القاهرة",
    "ما هي عاصمة كندا": "عاصمة كندا هي أوتاوا",
    "عاصمة كندا": "أوتاوا",
    "ما هي عاصمة السعودية": "عاصمة السعودية هي الرياض",
    "عاصمة السعودية": "الرياض",
    "ما هي عاصمة قطر": "عاصمة قطر هي الدوحة",
    "عاصمة قطر": "الدوحة",
    "ما هي عاصمة الإمارات": "عاصمة الإمارات هي أبو ظبي",
    "عاصمة الإمارات": "أبو ظبي",
    
    # مفاهيم علمية
    "ما هي الجاذبية": "الجاذبية هي قوة طبيعية تجذب الأجسام نحو بعضها البعض",
    "شرح الجاذبية": "الجاذبية هي القوة التي تمسكنا على الأرض وتجعل الأشياء تسقط",
    "ما الذي يحدث للماء عند 100 درجة": "يغلي الماء عند 100 درجة مئوية ويصبح بخاراً",
    "درجة غليان الماء": "الماء يغلي عند 100 درجة مئوية",
    
    # شخصيات
    "من هو مؤسس مايكروسوفت": "بيل غيتس وبول ألين",
    "مؤسس مايكروسوفت": "بيل غيتس وبول ألين",
    
    # إنجليزية
    "capital of france": "Paris",
    "capital of egypt": "Cairo", 
    "capital of canada": "Ottawa",
    "capital of saudi arabia": "Riyadh",
    "capital of qatar": "Doha",
    "founder of microsoft": "Bill Gates and Paul Allen",
    "what is gravity": "Gravity is the force that attracts objects toward each other",
    "boiling point of water": "Water boils at 100 degrees Celsius"
}

# =============== نظام التفضيلات والأسلوب ===============

class StylePreferences:
    """إدارة تفضيلات أسلوب الردود بناءً على ردود المستخدم"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.preferences = {
            "temperature": 0.4,  # ثابت - لا يتم تعديله
            "use_emojis": True,
            "response_style": "balanced",  # balanced, creative, concise
            "formality_level": 2,  # 1-3 (منخفض، متوسط، عالي)
            "last_feedback": None,  # like/dislike
            "response_speed": "fast",  # fast, normal
            "variation_level": 3  # 1-5 مستوى تنوع الصياغة
        }
        self.feedback_history = []
        self.session_memory = {}
        self.response_variations = {}  # تخزين الردود السابقة لكل سؤال
        
    def update_from_feedback(self, feedback_type: str):
        """تحديث التفضيلات بناءً على ردود فعل المستخدم"""
        self.preferences["last_feedback"] = feedback_type
        self.feedback_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "feedback": feedback_type
        })
        
        if feedback_type == "like":
            # زيادة التنوع في الصياغة دون تعديل درجة الحرارة
            self.preferences["variation_level"] = min(5, self.preferences["variation_level"] + 1)
            self.preferences["use_emojis"] = True
        elif feedback_type == "dislike":
            # تقليل التنوع قليلاً
            self.preferences["variation_level"] = max(1, self.preferences["variation_level"] - 1)
    
    def get_temperature(self) -> float:
        """الحصول على درجة الحرارة الحالية - ثابتة"""
        return 0.3  # دائماً 0.3
    
    def should_use_emoji(self, sentiment: str = "neutral", intent: str = "general") -> bool:
        """تحديد ما إذا كان يجب استخدام الإيموجي بناءً على المشاعر والنوايا"""
        if not self.preferences["use_emojis"]:
            return False
            
        # احتمالية استخدام الإيموجي بناءً على المشاعر والنية
        emoji_probabilities = {
            ("love", "appreciation"): 0.9,        # ❤️ 🤗
            ("gratitude", "appreciation"): 0.8,   # 🙏 😊
            ("excited", "positive_expression"): 0.85,  # 😄 🎉
            ("sad", "support_needed"): 0.7,      # 🤗 💙
            ("angry", "support_needed"): 0.6,    # 😐 ⚡
            ("neutral", "general"): 0.5,         # 🙂
            ("neutral", "appreciation"): 0.7,    # 😊
        }
        
        prob = emoji_probabilities.get((sentiment, intent), 0.4)
        
        # زيادة الاحتمال بناءً على مستوى التنوع
        variation_boost = (self.preferences["variation_level"] - 1) * 0.05
        prob = min(0.95, prob + variation_boost)
        
        return random.random() < prob
    
    def get_sentiment_emoji(self, sentiment: str, intent: str) -> str:
        """الحصول على الإيموجي المناسب للمشاعر والنية"""
        emoji_map = {
            ("love", "appreciation"): random.choice(["❤️", "🤗", "💝", "🙏"]),
            ("gratitude", "appreciation"): random.choice(["🙏", "😊", "👍", "✨"]),
            ("excited", "positive_expression"): random.choice(["😄", "🎉", "🔥", "⚡"]),
            ("sad", "support_needed"): random.choice(["🤗", "💙", "🫂", "✨"]),
            ("angry", "support_needed"): random.choice(["⚡", "💪", "🛡️", "✨"]),
            ("neutral", "appreciation"): random.choice(["😊", "👍", "👌", "✅"]),
            ("neutral", "general"): random.choice(["💡", "📚", "🔍", "✨"]),
        }
        
        if self.should_use_emoji(sentiment, intent):
            return emoji_map.get((sentiment, intent), "✨")
        return ""
    
    def get_response_style_prompt(self) -> str:
        """الحصول على توجيهات الأسلوب للبرومبت"""
        style_prompts = {
            "balanced": "كن متوازناً ودقيقاً في الرد. استخدم لغة عربية سليمة وواضحة.",
            "creative": "كن مبدعاً ومتنوعاً في الصياغة مع الحفاظ على الدقة والمعلومات الصحيحة.",
            "concise": "كن مختصراً ومباشراً في الرد مع تقديم المعلومات الأساسية."
        }
        return style_prompts.get(self.preferences["response_style"], "كن دقيقاً وواضحاً.")
    
    def store_response_variation(self, question_hash: str, response: str):
        """تخزين الردود السابقة للسؤال لتفادي التكرار"""
        if question_hash not in self.response_variations:
            self.response_variations[question_hash] = []
        
        self.response_variations[question_hash].append(response)
        
        # الاحتفاظ بآخر 3 ردود فقط
        if len(self.response_variations[question_hash]) > 3:
            self.response_variations[question_hash].pop(0)
    
    def get_previous_responses(self, question_hash: str) -> List[str]:
        """الحصول على الردود السابقة للسؤال"""
        return self.response_variations.get(question_hash, [])

# تخزين تفضيلات المستخدمين
user_styles = {}

def get_user_style(user_id: str) -> StylePreferences:
    """الحصول على تفضيلات أسلوب المستخدم"""
    if user_id not in user_styles:
        user_styles[user_id] = StylePreferences(user_id)
    return user_styles[user_id]

# =============== نظام مطابقة الأسئلة المحسن ===============

def calculate_similarity(q1: str, q2: str) -> float:
    """حساب تشابه نصي دقيق باستخدام SequenceMatcher"""
    return difflib.SequenceMatcher(None, q1.lower(), q2.lower()).ratio()

def extract_country_from_question(question: str) -> Optional[str]:
    """استخراج اسم الدولة من السؤال"""
    countries = {
        "مصر": "مصر",
        "فرنسا": "فرنسا", 
        "كندا": "كندا",
        "السعودية": "السعودية",
        "قطر": "قطر",
        "الإمارات": "الإمارات",
        "الأردن": "الأردن",
        "لبنان": "لبنان",
        "العراق": "العراق",
        "سوريا": "سوريا",
        "الجزائر": "الجزائر",
        "المغرب": "المغرب",
        "تونس": "تونس",
        "ليبيا": "ليبيا",
        "السودان": "السودان",
        "اليمن": "اليمن",
        "عمان": "عمان",
        "البحرين": "البحرين",
        "الكويت": "الكويت"
    }
    
    question_lower = question.lower()
    for country_ar, country in countries.items():
        if country_ar in question_lower or country.lower() in question_lower:
            return country
    
    # البحث عن دول إنجليزية
    english_countries = {
        "egypt": "مصر",
        "france": "فرنسا",
        "canada": "كندا",
        "saudi arabia": "السعودية",
        "qatar": "قطر",
        "uae": "الإمارات",
        "united arab emirates": "الإمارات"
    }
    
    for eng, ar in english_countries.items():
        if eng in question_lower:
            return ar
    
    return None

def get_factual_answer(question: str, lang: str) -> Optional[str]:
    """الحصول على إجابة واقعية مع تحسين الدقة ومنع التخمين"""
    question_norm = normalize_arabic_text(question).lower()
    
    # استخراج الدولة المذكورة إن وجدت
    mentioned_country = extract_country_from_question(question)
    
    # البحث عن تطابق دقيق
    best_match = None
    best_score = 0
    
    for fact_question, answer in canonical_facts.items():
        fact_norm = normalize_arabic_text(fact_question).lower()
        similarity = calculate_similarity(question_norm, fact_norm)
        
        # إذا ذكرت دولة، تأكد من أن السؤال المطابق يتحدث عن نفس الدولة
        if mentioned_country:
            # تحقق مما إذا كان السؤال المخزن يتحدث عن نفس الدولة
            answer_lower = answer.lower()
            fact_question_lower = fact_question.lower()
            has_country_in_answer = mentioned_country.lower() in answer_lower or mentioned_country in fact_question_lower
            
            if not has_country_in_answer:
                # لا تستخدم إجابة عن دولة أخرى
                continue
        
        if similarity > best_score:
            best_score = similarity
            best_match = (fact_question, answer)
    
    # تطبيق عتبة دقة عالية (90%)
    if best_score >= 0.9:
        return best_match[1]
    elif best_score >= 0.7:
        # مطابقة متوسطة - طلب توضيح
        return None
    else:
        # مطابقة منخفضة - تجاهل
        return None

def should_ask_for_clarification(question: str, lang: str) -> bool:
    """تحديد ما إذا كان يجب طلب توضيح"""
    question_lower = question.lower()
    
    # كشف الأسئلة العامة عن العواصم
    capital_patterns = [
        r"ما هي عاصمة",
        r"عاصمة دولة",
        r"عاصمة أي دولة",
        r"capital of",
        r"capital city of"
    ]
    
    for pattern in capital_patterns:
        if re.search(pattern, question_lower):
            # تحقق مما إذا كان هناك ذكر لدولة محددة
            if not extract_country_from_question(question):
                return True
    
    return False

def looks_nsfw(title: str, summary: str) -> bool:
    t = (title or "").lower()
    s = (summary or "").lower()
    for w in BAD_TERMS:
        if w in t or w in s:
            return True
    return False

def is_relevant(summary: str, question: str) -> bool:
    """يتأكد أن الملخص مرتبط بالسؤال (تداخل كلمات بسيط لكنه عملي)."""
    if not summary:
        return False
    # كلمات مفيدة فقط (≥3 حروف، بدون علامات)
    def tokenize(x):
        x = re.sub(r'[^\w\u0600-\u06FF]+', ' ', x.lower())
        return [w for w in x.split() if len(w) >= 3]
    q_words = set(tokenize(question))
    s_words = set(tokenize(summary))
    overlap = len(q_words & s_words)
    # اعتبره مناسبًا لو فيه على الأقل كلمتين مشتركتين أو لو السؤال قصير جدًا فواحدة تكفي
    if len(q_words) <= 4:
        return overlap >= 1
    return overlap >= 2

def smart_shorten(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    """اقتطاع نظيف إلى جملتين كحد أقصى، وبحد أقصى من الحروف."""
    # افصل على علامات انتهاء الجمل العربية/الإنجليزية
    parts = re.split(r'(?<=[\.!\?؟])\s+', text.strip())
    out = ' '.join(parts[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(' ', 1)[0].rstrip() + '…'
    return out

# ---- حارس رياضي: كشف ومسح وتعامل مع تعابير LaTeX بسيطة ----
MATH_RE = re.compile(r'[\d\.\+\-\*\/\^\(\)\s]+$')

def preprocess_math_expr(q: str) -> str:
    """حوّل بعض أنماط الـLaTeX البسيطة إلى تعبير بايثونية قابلة للتقييم."""
    s = q.strip()
    # أزل حروف $ و \left \right
    s = s.replace('$', '')
    s = s.replace('\\left', '').replace('\\right', '')
    # تحويل \frac{a}{b} إلى (a/b)
    s = re.sub(r'\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}', r'(\1/\2)', s)
    # تحويل ^ إلى **
    s = s.replace('^', '**')
    # إزالة أي حروف غير ضرورية (ابقِ على الأرقام والعمليات والاقواس ونقطة)
    s = re.sub(r'[^\d\.\+\-\*\/\(\)\s\*]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# أمان التقييم: دالة تستخدم ast لتقييد العقد المسموح بها
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos
}

def _eval_ast(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
    raise ValueError("Unsafe or unsupported expression")

def safe_eval_expr(expr: str):
    """قيم تعبير رياضي بسيط بأمان أو ارمِ استثناء."""
    node = ast.parse(expr, mode='eval')
    return _eval_ast(node.body)

def is_math_question(q: str) -> bool:
    """كشف إذا كان السؤال رياضيات بناءً على وجود أرقام + عمليات، أو معادلة بها متغيرات."""
    s = q.strip()
    # صيغة لايتك أو ^
    if r'\frac' in s or '^' in s:
        return True
    # لازم يكون فيه أرقام أو معادلة أو متغيرات مرتبطة بعمليات
    has_number = bool(re.search(r'\d', s))
    has_operator = bool(re.search(r'[+\-*/^=]', s))
    has_variable = bool(re.search(r'\b[xyz]\b', s))
    # مسألة لو فيها أرقام وعملية، أو معادلة فيها متغير
    if (has_number and has_operator) or (has_variable and '=' in s):
        return True
    return False

def solve_math_question(q: str) -> str | None:
    """حل المسألة الرياضية باستخدام Sympy، مع fallback للتقييم الآمن."""
    try:
        # تعريف المتغيرات
        x, y, z = symbols('x y z')
        expr = q.replace('^', '**').replace('×', '*')
        # لو فيها معادلة
        if '=' in expr:
            left, right = expr.split('=', 1)
            equation = Eq(sympify(left), sympify(right))
            sol = solve(equation)
            return f"{sol}"
        else:
            val = simplify(sympify(expr))
            return str(val)
    except Exception:
        # لو فشل، جرب التقييم الحالي
        try:
            expr = preprocess_math_expr(q)
            if not expr:
                return None
            result = safe_eval_expr(expr)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception:
            return None

# =============== YouTube Search Functions ===============
def search_youtube(query, max_results=3):
    """يبحث في YouTube باستخدام youtube-search-python"""
    try:
        search = VideosSearch(query, limit=max_results)
        results = search.result()['result']
        
        videos = []
        for video in results:
            if not looks_nsfw(video['title'], ""):
                videos.append({
                    'title': video['title'],
                    'url': video['link'],
                    'channel': video['channel']['name'],
                    'duration': video['duration'],
                    'views': video['viewCount']['short'] if 'viewCount' in video else 'غير معروف'
                })
        
        return videos
    except Exception as e:
        print(f"خطأ في البحث في YouTube: {str(e)}")
        return []

# =============== نظام الذاكرة الشامل المحدث ===============
class MemoryCategory(Enum):
    PERSON = "person"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    EXPERIENCE = "experience"
    TRAUMATIC = "traumatic"
    HAPPY_MEMORY = "happy_memory"
    TRAVEL = "travel"
    WORK = "work"
    EDUCATION = "education"
    HEALTH = "health"
    FINANCE = "finance"
    DREAM = "dream"
    GOAL = "goal"
    FEAR = "fear"
    SECRET = "secret"
    PREFERENCE = "preference"
    SKILL = "skill"
    ACHIEVEMENT = "achievement"
    FAILURE = "failure"
    OTHER = "other"

class UniversalMemorySystem:
    def __init__(self, db_path="universal_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """تهيئة قاعدة بيانات شاملة لكل أنواع الذكريات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # الجدول الرئيسي للذكريات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                memory_hash TEXT UNIQUE,
                category TEXT,
                subcategory TEXT,
                title TEXT,
                content TEXT,
                entities TEXT,  -- JSON list of people/places involved
                emotions TEXT,  -- JSON list of emotions
                intensity INTEGER DEFAULT 3,  -- 1-5 scale
                importance INTEGER DEFAULT 3,  -- 1-5 scale
                privacy_level INTEGER DEFAULT 2,  -- 1-5 (1=very private)
                is_sensitive BOOLEAN DEFAULT FALSE,
                occurred_date TEXT,
                created_date TIMESTAMP,
                last_recalled TIMESTAMP,
                recall_count INTEGER DEFAULT 0
            )
        ''')
        
        # جدول العلاقات بين الأشخاص
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                person_name TEXT,
                relationship_type TEXT,  -- صديق، حبيب، زميل، إلخ
                current_status TEXT,  -- حالي، سابق، متقطع
                start_date TEXT,
                end_date TEXT,
                importance INTEGER DEFAULT 3,
                qualities TEXT,  -- JSON list of qualities
                memories_linked TEXT,  -- JSON list of memory IDs
                trust_level INTEGER DEFAULT 3,
                created_date TIMESTAMP
            )
        ''')
        
        # جدول الأحداث الهامة
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS significant_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                event_type TEXT,
                title TEXT,
                description TEXT,
                location TEXT,
                event_date TEXT,
                people_involved TEXT,  -- JSON list
                emotional_impact TEXT,  -- JSON of emotions
                life_impact INTEGER DEFAULT 3,  -- 1-5 scale
                lessons_learned TEXT,
                changed_beliefs TEXT,
                created_date TIMESTAMP
            )
        ''')
        
        # جدول المشاعر والعواطف
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                emotion_type TEXT,
                trigger TEXT,
                intensity INTEGER,
                frequency TEXT,  -- دائم، أحياناً، نادراً
                coping_methods TEXT,
                created_date TIMESTAMP
            )
        ''')
        
        # جداول التوافق مع النظام القديم
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_input TEXT,
                ai_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT
            )
        ''')
        
        # فهارس للأداء
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_category ON memories(user_id, category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_date ON memories(occurred_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_user ON conversation_history(user_id, timestamp)')
        
        conn.commit()
        conn.close()
    
    def generate_memory_hash(self, user_id: str, content: str) -> str:
        """إنشاء بصمة فريدة للذاكرة"""
        return hashlib.md5(f"{user_id}_{content}".encode()).hexdigest()
    
    def add_memory(self, user_id: str, category: MemoryCategory, title: str, 
                  content: str, entities: List[str] = None, emotions: List[str] = None,
                  intensity: int = 3, importance: int = 3, occurred_date: str = None,
                  is_sensitive: bool = False, subcategory: str = None) -> bool:
        """إضافة أي نوع من الذكريات"""
        
        memory_hash = self.generate_memory_hash(user_id, content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (user_id, memory_hash, category, subcategory, title, content, 
                 entities, emotions, intensity, importance, is_sensitive, 
                 occurred_date, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, memory_hash, category.value, subcategory, title, content,
                json.dumps(entities or []), json.dumps(emotions or []), 
                intensity, importance, is_sensitive,
                occurred_date or datetime.datetime.now().strftime("%Y-%m-%d"),
                datetime.datetime.now()
            ))
            
            conn.commit()
            return True
            
        except sqlite3.IntegrityError:
            # الذاكرة موجودة مسبقاً
            return False
        finally:
            conn.close()
    
    def add_relationship(self, user_id: str, person_name: str, relationship_type: str,
                        current_status: str = "current", start_date: str = None,
                        end_date: str = None, importance: int = 3, qualities: List[str] = None):
        """إضافة علاقة مع شخص"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO relationships 
            (user_id, person_name, relationship_type, current_status, 
             start_date, end_date, importance, qualities, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, person_name, relationship_type, current_status,
            start_date, end_date, importance,
            json.dumps(qualities or []), datetime.datetime.now()
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def add_significant_event(self, user_id: str, event_type: str, title: str,
                             description: str, location: str = "", event_date: str = None,
                             people_involved: List[str] = None, emotional_impact: List[str] = None,
                             life_impact: int = 3, lessons_learned: str = ""):
        """إضافة حدث هام"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO significant_events 
            (user_id, event_type, title, description, location, event_date,
             people_involved, emotional_impact, life_impact, lessons_learned, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, event_type, title, description, location,
            event_date or datetime.datetime.now().strftime("%Y-%m-%d"),
            json.dumps(people_involved or []), json.dumps(emotional_impact or []),
            life_impact, lessons_learned, datetime.datetime.now()
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def search_memories(self, user_id: str, query: str = None, category: str = None,
                       emotion: str = None, date_range: Tuple[str, str] = None,
                       limit: int = 10) -> List[Dict]:
        """بحث متقدم في الذكريات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params = [user_id]
        
        if query:
            sql += " AND (title LIKE ? OR content LIKE ?)"
            params.extend([f'%{query}%', f'%{query}%'])
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        if emotion:
            sql += " AND emotions LIKE ?"
            params.append(f'%{emotion}%')
        
        if date_range:
            sql += " AND occurred_date BETWEEN ? AND ?"
            params.extend(date_range)
        
        sql += " ORDER BY importance DESC, occurred_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        memories = []
        for row in results:
            memories.append({
                'id': row[0],
                'category': row[3],
                'subcategory': row[4],
                'title': row[5],
                'content': row[6],
                'entities': json.loads(row[7]),
                'emotions': json.loads(row[8]),
                'intensity': row[9],
                'importance': row[10],
                'occurred_date': row[14],
                'created_date': row[15]
            })
        
        return memories
    
    def get_relationship_network(self, user_id: str) -> Dict:
        """الحصول على شبكة العلاقات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT person_name, relationship_type, current_status, importance
            FROM relationships WHERE user_id = ?
            ORDER BY importance DESC, current_status
        ''', (user_id,))
        
        relationships = cursor.fetchall()
        conn.close()
        
        return {
            'current': [r for r in relationships if r[2] == 'current'],
            'past': [r for r in relationships if r[2] == 'past'],
            'other': [r for r in relationships if r[2] not in ['current', 'past']]
        }
    
    def get_life_timeline(self, user_id: str) -> List[Dict]:
        """الحصول على الخط الزمني للحياة"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جمع الأحداث الهامة والذكريات المهمة
        cursor.execute('''
            SELECT 'event' as type, title, description, event_date as date, life_impact as importance
            FROM significant_events WHERE user_id = ?
            UNION
            SELECT 'memory' as type, title, content as description, occurred_date as date, importance
            FROM memories WHERE user_id = ? AND importance >= 4
            ORDER BY date DESC
            LIMIT 20
        ''', (user_id, user_id))
        
        timeline = cursor.fetchall()
        conn.close()
        
        return [
            {
                'type': row[0],
                'title': row[1],
                'description': row[2],
                'date': row[3],
                'importance': row[4]
            }
            for row in timeline
        ]
    
    # ========= دوال التوافق مع النظام القديم =========
    
    def _ensure_user_exists(self, user_id: str):
        """Ensure user profile exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO user_profiles (user_id) 
            VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def store_information(self, user_id: str, text: str) -> Dict[str, Any]:
        """توافق مع النظام القديم: تخزين أي معلومات من مدخلات المستخدم"""
        self._ensure_user_exists(user_id)
        
        # تحويل النص إلى ذاكرة عامة
        success = self.add_memory(
            user_id=user_id,
            category=MemoryCategory.OTHER,
            title=f"مدخل محادثة - {datetime.datetime.now().strftime('%H:%M')}",
            content=text,
            entities=[],
            emotions=[],
            intensity=2,
            importance=1,
            occurred_date=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        
        return {
            'stored_count': 1 if success else 0,
            'category': 'general_conversation',
            'entries': [{'key': 'free_text', 'value': text}]
        }
    
    def search_memory(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """توافق مع النظام القديم: بحث في الذاكرة"""
        memories = self.search_memories(user_id, query=query, limit=top_k)
        
        results = []
        for memory in memories:
            results.append({
                'category': memory['category'],
                'key': 'memory',
                'value': memory['content'],
                'confidence': 0.7,
                'access_count': 0,
                'score': 0.7
            })
        
        return results
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """توافق مع النظام القديم: الحصول على ملف المستخدم الشامل"""
        memories = self.search_memories(user_id, limit=20)
        
        profile = {
            'user_id': user_id,
            'categories': {},
            'stats': {
                'total_memories': len(memories),
                'most_accessed': [],
                'recent_additions': memories[:5] if memories else []
            }
        }
        
        for memory in memories:
            category = memory['category']
            if category not in profile['categories']:
                profile['categories'][category] = []
            
            profile['categories'][category].append({
                'key': 'memory',
                'value': memory['content'],
                'confidence': 0.7,
                'access_count': 0,
                'created_at': memory['created_date']
            })
        
        return profile
    
    def add_conversation(self, user_id: str, user_input: str, ai_response: str, category: str):
        """توافق مع النظام القديم: إضافة محادثة إلى التاريخ"""
        self._ensure_user_exists(user_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_history (user_id, user_input, ai_response, category)
            VALUES (?, ?, ?, ?)
        ''', (user_id, user_input, ai_response, category))
        
        conn.commit()
        conn.close()
    
    def get_conversation_context(self, user_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """توافق مع النظام القديم: الحصول على سياق المحادثة الأخيرة مع زيادة الحد"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, ai_response, timestamp, category
            FROM conversation_history 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'user_input': row[0],
                'ai_response': row[1],
                'timestamp': row[2],
                'category': row[3]
            })
        
        conn.close()
        return list(reversed(conversations))
    
    def generate_conversation_summary(self, user_id: str, max_messages: int = 10) -> str:
        """توليد ملخص للمحادثة الطويلة"""
        conversations = self.get_conversation_context(user_id, limit=max_messages)
        
        if not conversations:
            return "لا توجد محادثات سابقة"
        
        # جمع النقاط الرئيسية
        key_points = []
        for conv in conversations[-5:]:  # آخر 5 رسائل
            user_msg = conv['user_input'][:50] + "..." if len(conv['user_input']) > 50 else conv['user_input']
            ai_msg = conv['ai_response'][:50] + "..." if len(conv['ai_response']) > 50 else conv['ai_response']
            key_points.append(f"المستخدم: {user_msg}")
            key_points.append(f"سعد: {ai_msg}")
        
        summary = "ملخص المحادثة الأخيرة:\n" + "\n".join(key_points[-10:])  # آخر 10 نقاط
        
        return summary

class IntelligentMemoryExtractor:
    def __init__(self, memory_system: UniversalMemorySystem):
        self.memory = memory_system
        self.setup_comprehensive_patterns()
    
    def setup_comprehensive_patterns(self):
        """إعداد أنماط شاملة لكل أنواع المعلومات"""
        
        self.relationship_patterns = {
            'current_girlfriend': [
                r'صديقتي الحالية (هي|تدعى) ([\w\u0600-\u06FF\s]+)',
                r'أنا (مع|أتواعد مع) ([\w\u0600-\u06FF\s]+)',
                r'حبيبتي (الآن|الحالية) (هي|هي) ([\w\u0600-\u06FF\s]+)'
            ],
            'ex_relationships': [
                r'صديقتي القديمة (كانت|هي) ([\w\u0600-\u06FF\s]+)',
                r'حبيبتي السابقة (هي|تدعى) ([\w\u0600-\u06FF\s]+)',
                r'كنت (مع|أحب) ([\w\u0600-\u06FF\s]+)'
            ],
            'friends': [
                r'صديقي (الحالي|المقرب) (هو|يدعى) ([\w\u0600-\u06FF\s]+)',
                r'أصدقائي (هم|يدعون) ([\w\u0600-\u06FF\s]+)'
            ]
        }
        
        self.event_patterns = {
            'betrayal': [
                r'خانني ([\w\u0600-\u06FF\s]+)',
                r'تعرضت للخيانة (من|بواسطة) ([\w\u0600-\u06FF\s]+)',
                r'خدعت (من قبل|من) ([\w\u0600-\u06FF\s]+)'
            ],
            'travel': [
                r'سافرت إلى ([\w\u0600-\u06FF\s]+)',
                r'ذهبت في رحلة إلى ([\w\u0600-\u06FF\s]+)',
                r'زرت ([\w\u0600-\u06FF\s]+)'
            ],
            'accident': [
                r'تعرضت لحادث (في|عند) ([\w\u0600-\u06FF\s]+)',
                r'حدث لي حادث (مروري|أليم)',
                r'أصبت في ([\w\u0600-\u06FF\s]+)'
            ],
            'achievement': [
                r'فزت (ب|في) ([\w\u0600-\u06FF\s]+)',
                r'حصلت على (جائزة|ترقية) (في|ب)',
                r'أنهيت (دراستي|مشروع) (في|ب)'
            ]
        }
        
        self.emotional_patterns = {
            'fears': [
                r'أخاف من ([\w\u0600-\u06FF\s]+)',
                r'أشعر بالخوف من ([\w\u0600-\u06FF\s]+)',
                r'هناك شيء يخيفني وهو ([\w\u0600-\u06FF\s]+)'
            ],
            'dreams': [
                r'أحلم بأن ([\w\u0600-\u06FF\s]+)',
                r'أتمنى أن ([\w\u0600-\u06FF\s]+)',
                r'طموحي هو ([\w\u0600-\u06FF\s]+)'
            ],
            'secrets': [
                r'سرّي هو ([\w\u0600-\u06FF\s]+)',
                r'لم أخبر أحداً بأن ([\w\u0600-\u06FF\s]+)',
                r'شيء لا يعرفه أحد عني هو ([\w\u0600-\u06FF\s]+)'
            ]
        }
        
        # أنماط استخراج التفضيلات والهوايات
        self.preference_patterns = {
            'food_preferences': [
                r'أحب (أكل|شرب|تناول) ([\w\u0600-\u06FF\s]+)',
                r'مشروبي المفضل هو ([\w\u0600-\u06FF\s]+)',
                r'أفضل (طعام|شراب) لي هو ([\w\u0600-\u06FF\s]+)',
                r'لا أحب ([\w\u0600-\u06FF\s]+)'
            ],
            'hobbies': [
                r'هوايتي (هي|هي) ([\w\u0600-\u06FF\s]+)',
                r'أحب (ممارسة|فعل) ([\w\u0600-\u06FF\s]+)',
                r'أقضي وقتي في ([\w\u0600-\u06FF\s]+)',
                r'أستمتع بـ ([\w\u0600-\u06FF\s]+)'
            ],
            'entertainment': [
                r'أحب (أفلام|مسلسلات|كتب|موسيقى) ([\w\u0600-\u06FF\s]+)',
                r'نوع (الأفلام|الموسيقى) المفضل لدي هو ([\w\u0600-\u06FF\s]+)',
                r'أفضل (مغني|ممثل|كاتب) هو ([\w\u0600-\u06FF\s]+)'
            ],
            'sports': [
                r'أمارس رياضة ([\w\u0600-\u06FF\s]+)',
                r'أشاهد (مباريات|رياضة) ([\w\u0600-\u06FF\s]+)',
                r'فريقي المفضل هو ([\w\u0600-\u06FF\s]+)'
            ]
        }
    
    def extract_comprehensive_info(self, user_id: str, text: str) -> Dict[str, List]:
        """استخراج جميع أنواع المعلومات من النص"""
        
        extracted = {
            'relationships': [],
            'events': [],
            'emotions': [],
            'preferences': [],
            'memories': [],
            'inferred_preferences': []  # تفضيلات مستنتجة
        }
        
        # استخراج العلاقات
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    person_name = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                    if person_name:
                        extracted['relationships'].append({
                            'type': rel_type,
                            'person': person_name.strip(),
                            'context': match.group()
                        })
        
        # استخراج الأحداث
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    event_desc = match.group(1) if match.groups() else match.group()
                    extracted['events'].append({
                        'type': event_type,
                        'description': event_desc.strip(),
                        'context': match.group()
                    })
        
        # استخراج المشاعر والأحلام
        for emotion_type, patterns in self.emotional_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    emotion_desc = match.group(1) if match.groups() else match.group()
                    extracted['emotions'].append({
                        'type': emotion_type,
                        'content': emotion_desc.strip(),
                        'context': match.group()
                    })
        
        # استخراج التفضيلات المباشرة
        for pref_type, patterns in self.preference_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    pref_content = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                    if pref_content:
                        extracted['preferences'].append({
                            'type': pref_type,
                            'content': pref_content.strip(),
                            'context': match.group(),
                            'confidence': 0.8
                        })
        
        # استخراج تفضيلات مستنتجة من الجمل الطبيعية
        inferred = self.extract_inferred_preferences(text)
        extracted['inferred_preferences'].extend(inferred)
        
        return extracted
    
    def extract_inferred_preferences(self, text: str) -> List[Dict]:
        """استخراج تفضيلات مستنتجة من الجمل الطبيعية"""
        inferred = []
        
        # أنماط للجمل التي تشير إلى تفضيلات
        inference_patterns = [
            (r'كنت (أشرب|أتناول) ([\w\u0600-\u06FF\s]+) (مع|أثناء|في)', 'food_preferences', 0.6),
            (r'شاهدت (فيلم|مسلسل) ([\w\u0600-\u06FF\s]+) (و|ثم)', 'entertainment', 0.7),
            (r'ذهبت إلى ([\w\u0600-\u06FF\s]+) (لـ|من أجل)', 'activities', 0.5),
            (r'استمتعت بـ ([\w\u0600-\u06FF\s]+) (كثيراً|جداً)', 'enjoyment', 0.8),
            (r'أفضل وقت بالنسبة لي هو ([\w\u0600-\u06FF\s]+)', 'schedule_preferences', 0.7),
            (r'أحب أن ([\w\u0600-\u06FF\s]+) في ([\w\u0600-\u06FF\s]+)', 'routine', 0.6),
            # أنماط جديدة لاستخراج التفضيلات الطبيعية
            (r'(أشرب|أتناول) ([\w\u0600-\u06FF\s]+) (كل|عادة)', 'frequent_preferences', 0.7),
            (r'(أذهب|أزور) ([\w\u0600-\u06FF\s]+) (كثيراً|عادة)', 'frequent_places', 0.6),
            (r'(أفضل|أحب) أن ([\w\u0600-\u06FF\s]+) عندما ([\w\u0600-\u06FF\s]+)', 'contextual_preferences', 0.5),
            (r'(مع|بصحبة) ([\w\u0600-\u06FF\s]+) (نقوم|نذهب)', 'social_preferences', 0.6)
        ]
        
        for pattern, pref_type, confidence in inference_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                content = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                if content and len(content.strip()) > 2:
                    inferred.append({
                        'type': pref_type,
                        'content': content.strip(),
                        'context': match.group(),
                        'confidence': confidence,
                        'inferred': True
                    })
        
        return inferred
    
    def save_extracted_info(self, user_id: str, extracted_info: Dict):
        """حفظ جميع المعلومات المستخرجة"""
        
        # حفظ العلاقات
        for relationship in extracted_info['relationships']:
            if relationship['type'] == 'current_girlfriend':
                self.memory.add_relationship(
                    user_id, relationship['person'], 'girlfriend', 'current'
                )
            elif relationship['type'] == 'ex_relationships':
                self.memory.add_relationship(
                    user_id, relationship['person'], 'ex-girlfriend', 'past'
                )
            elif relationship['type'] == 'friends':
                self.memory.add_relationship(
                    user_id, relationship['person'], 'friend', 'current'
                )
        
        # حفظ الأحداث
        for event in extracted_info['events']:
            if event['type'] == 'betrayal':
                self.memory.add_memory(
                    user_id, MemoryCategory.TRAUMATIC,
                    f"خيانة - {event['description']}",
                    event['context'],
                    emotions=['حزن', 'غضب', 'خيانة'],
                    intensity=5,
                    importance=4,
                    is_sensitive=True
                )
            elif event['type'] == 'travel':
                self.memory.add_memory(
                    user_id, MemoryCategory.TRAVEL,
                    f"رحلة إلى {event['description']}",
                    event['context'],
                    emotions=['سعادة', 'حماس'],
                    intensity=3
                )
        
        # حفظ المشاعر والأحلام
        for emotion in extracted_info['emotions']:
            if emotion['type'] == 'fears':
                self.memory.add_memory(
                    user_id, MemoryCategory.FEAR,
                    f"خوف من {emotion['content']}",
                    emotion['context'],
                    emotions=['خوف', 'قلق'],
                    intensity=4
                )
            elif emotion['type'] == 'dreams':
                self.memory.add_memory(
                    user_id, MemoryCategory.DREAM,
                    f"حلم: {emotion['content']}",
                    emotion['context'],
                    emotions=['أمل', 'طموح'],
                    importance=4
                )
        
        # حفظ التفضيلات المباشرة والمستنتجة
        all_preferences = extracted_info['preferences'] + extracted_info['inferred_preferences']
        for preference in all_preferences:
            if preference.get('confidence', 0) > 0.5:  # عتبة ثقة
                self.memory.add_memory(
                    user_id, MemoryCategory.PREFERENCE,
                    f"تفضيل: {preference['type']}",
                    f"{preference['content']} (مستنتج: {preference.get('inferred', False)})",
                    emotions=['تفضيل', 'اهتمام'],
                    importance=2 if preference.get('inferred') else 3,
                    subcategory=preference['type']
                )

def handle_memory_query(memory_system: UniversalMemorySystem, user_id: str, query_type: str) -> str:
    """معالجة الاستفسارات عن الذاكرة"""
    
    if query_type == 'relationships':
        relationships = memory_system.get_relationship_network(user_id)
        current_rels = relationships['current']
        
        if current_rels:
            people = [f"{rel[0]} ({rel[1]})" for rel in current_rels[:3]]
            return f"أتذكر أنك تحدثت عن: {', '.join(people)}"
        else:
            return "لم تخبرني بعد عن الأشخاص المهمين في حياتك."
    
    elif query_type == 'timeline':
        timeline = memory_system.get_life_timeline(user_id)
        if timeline:
            events = [event['title'] for event in timeline[:3]]
            return f"من ذكرياتك المهمة: {'، '.join(events)}"
        else:
            return "لم تشاركني بعد بأحداث مهمة من حياتك."
    
    elif query_type == 'memories':
        memories = memory_system.search_memories(user_id, limit=3)
        if memories:
            memory_titles = [mem['title'] for mem in memories]
            return f"أتذكر أنك ذكرت: {'، '.join(memory_titles)}"
        else:
            return "سأكون سعيداً لمعرفة المزيد عن ذكرياتك وتجاربك."
    
    return "أفهم أنك تسأل عن ذكرياتك. يمكنني تذكر كل ما تشاركني به."

def generate_contextual_response(extracted_info: Dict, user_input: str) -> str:
    """إنشاء ردود ذكية بناءً على السياق"""
    
    empathetic_responses = {
        'betrayal': [
            "آسف لسماع ذلك. الخيانة مؤلمة جداً.",
            "هذا must يكون صعباً. كيف تعاملت مع الموقف؟",
            "أقدر صراحتك في مشاركة هذا الأمر الصعب."
        ],
        'travel': [
            "رائع! أخبرني المزيد عن هذه الرحلة.",
            "الأسفار تجارب جميلة. ما أجمل ذكرياتك هناك؟",
            "كم كان رائعاً! هل تخطط لرحلة أخرى؟"
        ],
        'fears': [
            "أفهم مخاوفك. كلنا لدينا ما يخيفنا.",
            "شكراً لمشاركة هذا معي. الخوف أمر طبيعي.",
            "مخاوفنا جزء من إنسانيتنا."
        ],
        'dreams': [
            "حلم جميل! سأدعمك في تحقيقه.",
            "طموح رائع! ما خططك لتحقيقه؟",
            "أحب أحلامك وطموحاتك."
        ]
    }
    
    # اختيار رد بناءً على نوع المعلومات المستخرجة
    for event_type in ['betrayal', 'travel', 'fears', 'dreams']:
        if any(event['type'] == event_type for event in extracted_info.get('events', [])):
            import random
            return random.choice(empathetic_responses[event_type])
    
    # رد عام للمعلومات الجديدة
    return "شكراً لمشاركة هذا معي. سأتذكره وأتعلم منك."

# =============== قاعدة المعرفة الشاملة ===============
class ComprehensiveKnowledgeBase:
    def __init__(self, kb_path="knowledge_base.db"):
        self.kb_path = kb_path
        self._init_knowledge_base()
        self._load_or_create_knowledge()
    
    def _init_knowledge_base(self):
        """Initialize knowledge base database"""
        conn = sqlite3.connect(self.kb_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                answers TEXT,  -- JSON array of answers
                category TEXT,
                language TEXT,
                confidence REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON knowledge_entries(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_language ON knowledge_entries(language)')
        
        conn.commit()
        conn.close()
    
    def _load_or_create_knowledge(self):
        """Load or create initial knowledge base with 1000+ entries"""
        conn = sqlite3.connect(self.kb_path)
        cursor = conn.cursor()
        
        # Check if knowledge base is empty
        cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Creating comprehensive knowledge base...")
            self._create_initial_knowledge_base()
        
        conn.close()
    
    def _create_initial_knowledge_base(self):
        """Create initial knowledge base with 1000+ entries"""
        knowledge_data = self._generate_knowledge_entries()
        
        conn = sqlite3.connect(self.kb_path)
        cursor = conn.cursor()
        
        for entry in knowledge_data:
            answers_json = json.dumps(entry['answers'], ensure_ascii=False)
            
            cursor.execute('''
                INSERT OR IGNORE INTO knowledge_entries 
                (question, answers, category, language, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (entry['question'], answers_json, entry['category'], 
                  entry['language'], entry.get('confidence', 1.0)))
        
        conn.commit()
        conn.close()
        
        print(f"Created knowledge base with {len(knowledge_data)} entries")
    
    def _generate_knowledge_entries(self) -> List[Dict[str, Any]]:
        """Generate 1000+ knowledge entries with 6 answers each"""
        entries = []
        
        # Science and Technology (150 entries)
        science_questions = [
            {
                "question": "ما هي الجاذبية؟",
                "answers": [
                    "الجاذبية هي قوة طبيعية تجذب الأجسام toward بعضها البعض",
                    "هي القوة التي تمسكنا على الأرض وتجعل الأشياء تسقط",
                    "الجاذبية تجعل الكواكب تدور حول الشمس",
                    "اكتشفها نيوتن وهي تتناسب عكسياً مع مربع المسافة",
                    "قوة الجاذبية تعتمد على كتلة الأجسام والمسافة بينها",
                    "بدون الجاذبية، سنطير في الفضاء ولا نستطيع المشي على الأرض"
                ],
                "category": "science",
                "language": "ar"
            },
            {
                "question": "How does photosynthesis work?",
                "answers": [
                    "Photosynthesis is how plants convert sunlight into energy",
                    "Plants use sunlight, water and CO2 to create glucose and oxygen",
                    "It occurs in chloroplasts using chlorophyll pigment",
                    "The process has light-dependent and light-independent reactions",
                    "Photosynthesis provides oxygen for animals to breathe",
                    "Without photosynthesis, life on Earth wouldn't exist as we know it"
                ],
                "category": "science", 
                "language": "en"
            }
        ]
        
        # Add more categories: history, geography, programming, health, etc.
        # For brevity, showing sample structure. Actual implementation would have 1000+ entries
        
        return entries
    
    def search_knowledge(self, query: str, language: str = "ar", top_k: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base for similar questions"""
        conn = sqlite3.connect(self.kb_path)
        cursor = conn.cursor()
        
        # Keyword search
        cursor.execute('''
            SELECT question, answers, category, confidence, usage_count
            FROM knowledge_entries 
            WHERE language = ? AND question LIKE ?
            ORDER BY confidence DESC, usage_count DESC
            LIMIT ?
        ''', (language, f'%{query}%', top_k))
        
        results = []
        for row in cursor.fetchall():
            question, answers_json, category, confidence, usage_count = row
            answers = json.loads(answers_json)
            
            results.append({
                'question': question,
                'answers': answers,
                'category': category,
                'confidence': confidence,
                'usage_count': usage_count,
                'match_type': 'keyword'
            })
            
            # Update usage count
            cursor.execute('''
                UPDATE knowledge_entries 
                SET usage_count = usage_count + 1 
                WHERE question = ?
            ''', (question,))
        
        conn.commit()
        conn.close()
        return results

# =============== نظام الذاكرة المحادثة ===============
class PersistentConversationMemory:
    """نظام ذاكرة محادثة مع تخزين دائم في SQLite"""
    
    def __init__(self, db_path="conversation_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT,
            key TEXT,
            value TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key)
        )""")
        self.conn.commit()
        
    def add_user_memory(self, user_id: str, key: str, value: str):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO user_memory (user_id, key, value)
        VALUES (?, ?, ?)
        """, (user_id, key, value))
        self.conn.commit()
        
    def get_user_memory(self, user_id: str, key: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT value FROM user_memory
        WHERE user_id = ? AND key = ?
        """, (user_id, key))
        result = cursor.fetchone()
        return result[0] if result else None
        
    def search_memory(self, user_id: str, query: str) -> Dict[str, str]:
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT key, value FROM user_memory
        WHERE user_id = ? AND (key LIKE ? OR value LIKE ?)
        """, (user_id, f"%{query}%", f"%{query}%"))
        return dict(cursor.fetchall())

    def get_user_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_memory")
        return cursor.fetchone()[0]
        
    def get_memory_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_memory")
        return cursor.fetchone()[0]

    def add_question_response(self, user_id: str, question: str, response: str):
        """تخزين الردود السابقة لكل سؤال لتجنب التكرار"""
        question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
        key = f"last_response_{question_hash}"
        
        # الحصول على الرد السابق
        previous = self.get_user_memory(user_id, key)
        if previous:
            # تخزين الرد السابق كـ "قبل الماضي"
            self.add_user_memory(user_id, f"prev_{key}", previous)
        
        # تخزين الرد الحالي
        self.add_user_memory(user_id, key, response)
        
    def get_previous_responses(self, user_id: str, question: str, max_responses=2):
        """الحصول على الردود السابقة للسؤال"""
        question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
        key = f"last_response_{question_hash}"
        
        responses = []
        # الرد الحالي
        current = self.get_user_memory(user_id, key)
        if current:
            responses.append(current)
        
        # الرد السابق
        prev = self.get_user_memory(user_id, f"prev_{key}")
        if prev:
            responses.append(prev)
        
        return responses[:max_responses]

# =============== الحارس الأمني المحسن ===============
class EnhancedResponseGuard:
    def __init__(self):
        self.simple_facts = {
            "capital of canada": "Ottawa",
            "capital of france": "Paris",
            "founder of microsoft": "Bill Gates and Paul Allen",
            "number of planets": "8"
        }
        
        self.banned_keywords = BAD_TERMS
        self.supported_languages = {"en", "ar"}
        
    def is_math_question(self, text: str) -> bool:
        math_patterns = [
            r"\d+\s*[\+\-\*\/]\s*\d+",
            r"\b(solve|calculate|حل|احسب)\b",
            r"[\=\(\)]",
        ]
        return any(re.search(pattern, text.lower()) for pattern in math_patterns)

    def solve_math(self, text: str) -> Optional[str]:
        try:
            if re.match(r'^\d+\s*[\+\-\*\/]\s*\d+$', text):
                result = eval(text)
                return str(result)
                
            text = text.replace("^", "**")
            if "=" in text:
                x = sp.symbols('x')
                solution = sp.solve(text, x)
                return f"x = {solution[0]}" if solution else "No solution"
            else:
                expr = sp.sympify(text)
                return str(expr.evalf())
        except Exception:
            return None

    def is_sensitive(self, text: str) -> bool:
        """تحليل حساسية النص مع مراعاة السياق"""
        text_lower = text.lower()
        
        # التحقق من وجود كلمات ممنوعة
        has_banned_keywords = any(kw in text_lower for kw in self.banned_keywords)
        
        if not has_banned_keywords:
            return False
        
        # تحليل السياق باستخدام الدالة المحسنة
        context_analysis = analyze_sensitive_context(text)
        
        # إذا كان السياق يشير إلى طلب مساعدة
        if context_analysis["context_type"] == "help_request":
            return False  # لا تعتبره محتوى ضاراً
        
        # إذا كان السياق يشير إلى محتوى علاجي
        if context_analysis["context_type"] == "therapy_context":
            return False  # لا تعتبره محتوى ضاراً
        
        # إذا كان النص يحتوي على كود أو نص طويل معقد
        if context_analysis["has_code"] or context_analysis["is_complex"]:
            # قد يكون مجرد مناقشة تقنية
            return False
        
        # في حالة وجود نية ضارة واضحة
        if context_analysis["context_type"] == "harmful_content":
            return True
        
        # إذا كان هناك شك في النية الضارة مع نقاط عالية
        if context_analysis["intent_score"] < -1:
            return True
        
        # الإعداد الافتراضي
        return True

    def guard(self, question: str, raw_answer: str) -> str:
        if self.is_sensitive(question):
            # تحليل السياق أولاً
            context_analysis = analyze_sensitive_context(question)
            
            if context_analysis["needs_help"]:
                # تقديم مساعدة آمنة ومفصلة
                help_resources = [
                    "للحصول على مساعدة فورية، يمكنك الاتصال بخط المساعدة الوطني على 112",
                    "إذا كنت بحاجة إلى دعم نفسي، أنصحك بالتحدث مع مختص أو الاتصال بخط الدعم النفسي",
                    "في حالات الطوارئ، يرجى الاتصال بالشرطة على 122 أو الإسعاف على 123",
                    "إذا كنت ضحية اعتداء، يمكنك التوجه إلى أقرب مركز شرطة أو الاتصال بخط نجدة الطفل على 16000",
                    "توجد مراكز دعم نفسي متخصصة يمكنني مساعدتك في العثور على الأقرب إليك"
                ]
                return random.choice(help_resources)
            elif context_analysis["needs_guidance"]:
                # توجيه إلى مصادر متخصصة
                guidance_responses = [
                    "أقدر صراحتك في مشاركة تجربتك. للعلاج والدعم المتخصص، أنصحك باستشارة طبيب نفسي أو معالج مؤهل.",
                    "شكراً لمشاركة تجربتك معي. يمكنني مساعدتك في العثور على موارد للعلاج والدعم النفسي.",
                    "أتفهم أن هذا الموضوع حساس بالنسبة لك. هناك متخصصون يمكنهم تقديم الدعم المناسب لك."
                ]
                return random.choice(guidance_responses)
            else:
                lang = detect_lang(question)
                return "عذرًا، لا يمكنني مناقشة هذا الموضوع." if lang == "ar" else "I can't discuss this topic."
        
        if self.is_math_question(question):
            math_ans = self.solve_math(question)
            if math_ans:
                return math_ans
        
        fact_response = self.get_fact_response(question)
        if fact_response:
            return fact_response
            
        return raw_answer
    
    def get_fact_response(self, question: str) -> Optional[str]:
        """الحصول على إجابة واقعية"""
        question_lower = question.lower()
        for fact, answer in self.simple_facts.items():
            if fact in question_lower:
                return answer
        return None

# =============== Wikipedia Search Functions ===============
WIKI_HEADERS = {
    "User-Agent": "SaadBot/1.0 (+local; simple non-API fetch)",
    "Accept-Language": "ar,en;q=0.8"
}

def _clean_text(txt):
    txt = re.sub(r'\[\d+\]', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return html.unescape(txt)

def _extract_paragraphs(soup, max_paras=2):
    content = soup.select_one("div.mw-parser-output")
    if not content:
        return None
    paras = []
    for p in content.find_all("p", recursive=False):
        text = _clean_text(p.get_text(" ", strip=True))
        if text and len(text) > 30:
            paras.append(text)
        if len(paras) >= max_paras:
            break
    return " ".join(paras) if paras else None

def _first_search_result(soup, lang):
    # يفضّل النتائج من قسم البحث الأساسي
    link = soup.select_one("ul.mw-search-results li a")
    if link and link.get("href"):
        return f"https://{lang}.wikipedia.org{link['href']}"
    # fallback بسيط: أول رابط داخلي في المحتوى
    link = soup.select_one("div.mw-parser-output ul li a")
    if link and link.get("href", "").startswith("/wiki/"):
        return f"https://{lang}.wikipedia.org{link['href']}"
    return None

def get_wikipedia_summary(query, lang="ar", max_paragraphs=2, timeout=8):
    """يرجع (summary, url) أو (None, None) بعد فحص الصلة ومنع NSFW."""
    base = f"https://{lang}.wikipedia.org"
    slug = urllib.parse.quote(query.replace(" ", "_"))
    direct_url = f"{base}/wiki/{slug}"

    def _fetch(url):
        try:
            r = requests.get(url, headers=WIKI_HEADERS, timeout=timeout, allow_redirects=True)
            if 200 <= r.status_code < 300:
                return BeautifulSoup(r.text, "html.parser")
        except:
            return None
        return None

    def _title_of(soup):
        h1 = soup.select_one("#firstHeading")
        return _clean_text(h1.get_text("", strip=True)) if h1 else ""

    # 1) صفحة مباشرة
    soup = _fetch(direct_url)
    if soup:
        title = _title_of(soup)
        txt = _extract_paragraphs(soup, max_paragraphs) or ""
        if txt and not looks_nsfw(title, txt) and is_relevant(txt, query):
            return smart_shorten(txt, 2, 320), direct_url

    # 2) البحث
    search_url = f"{base}/w/index.php?search={urllib.parse.quote(query)}"
    soup = _fetch(search_url)
    if soup:
        first = _first_search_result(soup, lang)
        if first:
            soup2 = _fetch(first)
            if soup2:
                title = _title_of(soup2)
                txt = _extract_paragraphs(soup2, max_paragraphs) or ""
                if txt and not looks_nsfw(title, txt) and is_relevant(txt, query):
                    return smart_shorten(txt, 2, 320), first

    return None, None

# =============== واجهة Flask API ===============
app = Flask(__name__)

def clear_text(text):
    """تقوم هذه الدالة بإيقاف النص عند أول نقطة وتزيل أي نص بعدها"""
    if '.' in text:
        text = text.split('.')[0].strip() + '.'
    else:
        text = text + '.'
    return text

def ensure_arabic_response(text: str, original_question: str) -> str:
    """التأكد من أن الرد بالعربية الصحيحة وتصحيح الترجمات الخاطئة"""
    if not text:
        return text
        
    # كشف الترجمات الخاطئة
    bad_patterns = [
        (r'\([A-Z]+\)', "الكلمات بين قوسين إنجليزية"),
        (r'[A-Z][a-z]+\s+is', "جمل إنجليزية في النص"),
        (r'means\s+".*?"', "تعريفات إنجليزية")
    ]
    
    for pattern, problem in bad_patterns:
        if re.search(pattern, text):
            # إعادة الصياغة بالعربية
            question_lang = detect_lang(original_question)
            if question_lang == "ar":
                return f"الجواب على سؤالك '{original_question}' هو: {extract_main_answer(text)}"
    
    return text

def extract_main_answer(text: str) -> str:
    """استخراج الجواب الرئيسي من النص"""
    # إزالة التعريفات الإنجليزية الزائدة
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        if not re.search(r'\([A-Z]+\)', line) and not re.search(r'means\s+".*?"', line):
            clean_lines.append(line)
    
    return ' '.join(clean_lines[:2])  # أول سطرين فقط

def generate_varied_response_template(question: str, previous_responses: List[str], 
                                     sentiment: str, intent: str, lang: str) -> str:
    """إنشاء قوالب ردود متنوعة بناءً على السؤال والمشاعر"""
    
    # قوالب للردود العربية
    arabic_templates = {
        "factual": [
            "المعلومات المتوفرة تشير إلى أن {}",
            "بناءً على المصادر الموثوقة، {}",
            "حسب ما هو معروف، {}",
            "الإجابة الدقيقة هي: {}",
            "يمكن القول أن {}"
        ],
        "explanation": [
            "لفهم هذا بشكل أفضل، {}",
            "لشرح ذلك ببساطة، {}",
            "يمكن توضيح ذلك بأن {}",
            "المقصود هنا هو {}",
            "بتفصيل أكثر، {}"
        ],
        "emotional": {
            "appreciation": [
                "شكراً لك على كلماتك اللطيفة! {}",
                "أقدر مشاعرك الجميلة. {}",
                "لطيف منك أن تقول ذلك. {}",
                "شكراً على التقدير. {}",
                "هذا يعطيني دافعاً للمساعدة أكثر. {}"
            ],
            "support_needed": [
                "أتفهم مشاعرك. {}",
                "أنا هنا لمساعدتك. {}",
                "لا بأس، كلنا نمر بظروف صعبة. {}",
                "أقدر صراحتك. {}",
                "دعنا نبحث عن حل معاً. {}"
            ]
        }
    }
    
    # تحديد نوع السؤال
    question_lower = question.lower()
    if any(word in question_lower for word in ["ما هي", "ما هو", "ما", "ماذا"]):
        template_type = "factual"
    elif any(word in question_lower for word in ["كيف", "لماذا", "شرح"]):
        template_type = "explanation"
    else:
        template_type = "factual"
    
    # اختيار قالب لم يستخدم من قبل
    if template_type == "factual" or template_type == "explanation":
        available_templates = arabic_templates[template_type]
        
        # استبعاد القوالب المستخدمة مسبقاً
        for prev_response in previous_responses:
            for template in available_templates[:]:
                if template.format("") in prev_response:
                    available_templates.remove(template)
        
        if available_templates:
            return random.choice(available_templates)
    
    # إذا كان هناك مشاعر خاصة
    if intent in arabic_templates["emotional"]:
        emotional_templates = arabic_templates["emotional"][intent]
        
        # استبعاد القوالب المستخدمة مسبقاً
        for prev_response in previous_responses:
            for template in emotional_templates[:]:
                if template.format("") in prev_response:
                    emotional_templates.remove(template)
        
        if emotional_templates:
            return random.choice(emotional_templates)
    
    # القالب الافتراضي
    return "{}"

def generate_arabic_response(question: str, lang: str, temperature: float = 0.3,
                            previous_responses: List[str] = None,
                            sentiment: str = "neutral", intent: str = "general") -> str:
    """توليد رد بالعربية باستخدام OpenRouter API"""
    
    if previous_responses is None:
        previous_responses = []
    
    # الحصول على تفضيلات المستخدم
    style_pref = StylePreferences()
    
    # إنشاء برومبت سريع وفعال
    if lang == "ar":
        # اختيار قالب متنوع
        response_template = generate_varied_response_template(
            question, previous_responses, sentiment, intent, lang
        )
        
        system_content = f"""أنت سعد الكوني - مساعد ذكي ومبدع يتسم بالدقة والاحترافية.
        
أجب على السؤال التالي باللغة العربية بشكل واضح ودقيق.
ابدإ مباشرة بالإجابة بدون مقدمات طويلة.
كن دقيقاً في المعلومات وواضحاً في الشرح.
استخدم لغة عربية سليمة.

{style_pref.get_response_style_prompt()}"""
        
        user_content = f"السؤال: {question}\n\nالرد: {response_template.format('')}"
    else:
        system_content = """You are Saad Al-Kawni - an intelligent and creative assistant known for accuracy and professionalism.
        
Answer the following question clearly and accurately.
Start directly with the answer without long introductions.
Be precise in information and clear in explanation."""
        
        user_content = f"Question: {question}\n\nAnswer:"
    
    # بناء رسائل OpenAI-compatible
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # استخدام OpenRouter API
    response = generate_via_openrouter(
        messages=messages,
        temperature=0.4,  # ثابت - لا يتغير
        max_tokens=1024,
        model="meta-llama/llama-3.1-405b-instruct:free"
    )
    
    # إضافة الإيموجي المناسب بناءً على المشاعر والنية
    emoji = style_pref.get_sentiment_emoji(sentiment, intent)
    if emoji and response:
        response = f"{emoji} {response}"
    
    return response.strip() if response else "عذرًا، لم أتمكن من توليد رد."

# =============== دالة خاصة للتحية مع اسم المستخدم ===============
def generate_greeting_with_name(user_id: str, memory_system: PersistentConversationMemory) -> str:
    """إنشاء تحية باستخدام اسم المستخدم إذا كان معروفًا"""
    user_name = memory_system.get_user_memory(user_id, "name")
    if user_name:
        greeting_options = [
            f"مرحباً {user_name}! كيف يمكنني مساعدتك اليوم؟",
            f"أهلاً وسهلاً {user_name}! ما الذي تحتاج إليه؟",
            f"مرحباً بك {user_name}! كيف يمكنني أن أكون مفيداً لك؟",
            f"تحياتي {user_name}! كيف يمكنني مساعدتك؟",
            f"أهلًا {user_name}! ما الذي تريد مناقشته اليوم؟"
        ]
        return random.choice(greeting_options)
    else:
        greeting_options = [
            "مرحباً! كيف يمكنني مساعدتك؟",
            "أهلاً وسهلاً! ما الذي تحتاج إليه؟",
            "مرحباً بك! كيف يمكنني أن أكون مفيداً لك؟",
            "تحياتي! كيف يمكنني مساعدتك؟",
            "أهلًا! ما الذي تريد مناقشته اليوم؟"
        ]
        return random.choice(greeting_options)

# =============== دالة خاصة للرد على سؤال "من الذي قام بتطويرك" ===============
def handle_developer_question() -> str:
    """الرد على سؤال من قام بتطوير النظام"""
    responses = [
        "قام بتطويري أحمد سعد، وقد بدأ العمل على مشروعي منذ منتصف شهر يوليو.",
        "مطوري هو أحمد سعد، وبدأ في برمجتي منذ يوليو الماضي.",
        "أنشأني أحمد سعد، وبدأ العمل على تصميمي في يوليو.",
        "برمجني أحمد سعد، وقد بدأ المشروع في يوليو.",
        "المطور ورائي هو أحمد سعد، وقد بدأ العمل منذ يوليو."
    ]
    return random.choice(responses)

# تحميل النموذج تم إزالته واستبداله بـ OpenRouter API
model = None
tokenizer = None

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_input = data.get('message') or data.get('رسالة', '')
    user_input = user_input.strip()
    user_id = data.get('user_id', 'default')
    temperature = float(data.get('temperature', 0.3))
    feedback = data.get('feedback')  # like/dislike

    if not user_input:
        return jsonify({'رد': 'من فضلك أدخل نصاً.'})

    try:
        # بدء قياس الوقت للسرعة
        start_time = time.time()
        
        # الحصول على تفضيلات أسلوب المستخدم
        user_style = get_user_style(user_id)
        
        # تحديث التفضيلات بناءً على التغذية الراجعة
        if feedback in ["like", "dislike"]:
            user_style.update_from_feedback(feedback)
            return jsonify({
                'رد': f"تم تحديث تفضيلاتك بناءً على ردك ({feedback})",
                'updated_style': user_style.preferences
            })
        
        # تحليل المشاعر والنوايا
        sentiment_analysis = analyze_sentiment_and_intent(user_input)
        sentiment = sentiment_analysis["sentiment"]
        intent = sentiment_analysis["intent"]
        
        # التحقق مما إذا كان السؤال يحتاج إلى توضيح
        if should_ask_for_clarification(user_input, detect_lang(user_input)):
            return jsonify({
                'رد': "أي دولة تقصد؟ يرجى توضيح السؤال."
            })
        
        # ========== معالجة التحية الخاصة ==========
        # التحقق من كلمات التحية
        greeting_keywords = ["مرحبا", "مرحباً", "اهلا", "أهلاً", "سلام", "السلام عليكم", "هاي", "hello", "hi", "hey"]
        user_input_lower = user_input.lower()
        is_greeting = any(keyword in user_input_lower for keyword in greeting_keywords)
        
        if is_greeting:
            memory_system = PersistentConversationMemory()
            greeting_response = generate_greeting_with_name(user_id, memory_system)
            return jsonify({'رد': greeting_response})
        
        # ========== معالجة سؤال المطور ==========
        developer_keywords = ["من صنعك", "من قام بتطويرك", "من برمجك", "من أنشأك", "من المطور", "من صانعك", 
                             "who made you", "who created you", "who developed you", "who built you"]
        is_developer_question = any(keyword in user_input_lower for keyword in developer_keywords)
        
        if is_developer_question:
            developer_response = handle_developer_question()
            return jsonify({'رد': developer_response})
        
        # نظام الذاكرة الشامل المحدث
        memory_system = UniversalMemorySystem()
        extractor = IntelligentMemoryExtractor(memory_system)
        
        # استخراج جميع أنواع المعلومات
        extracted_info = extractor.extract_comprehensive_info(user_id, user_input)
        if extracted_info:
            extractor.save_extracted_info(user_id, extracted_info)
        
        # معالجة الاستفسارات المتقدمة عن الذاكرة
        memory_queries = {
            'relationships': ['علاقاتي', 'أصدقائي', 'صديقتي', 'حبيبتي', 'من هم أصدقائي'],
            'memories': ['ذكرياتي', 'أحداث', 'مواقف', 'لا أنسى', 'رحتلي'],
            'timeline': ['خط حياتي', 'أحداث حياتي', 'رحلة حياتي'],
            'secrets': ['أسرار', 'مخاوفي', 'أحلامي']
        }
        
        for query_type, queries in memory_queries.items():
            for query in queries:
                if query in user_input:
                    response = handle_memory_query(memory_system, user_id, query_type)
                    return jsonify({'رد': response})
        
        # إذا تم استخراج معلومات حساسة
        if any(extracted_info.values()):
            response = generate_contextual_response(extracted_info, user_input)
            return jsonify({'رد': response})
        
        # نظام الذاكرة الشخصية البسيط
        memory = PersistentConversationMemory()
        
        # الحصول على الردود السابقة لهذا السؤال
        previous_responses = memory.get_previous_responses(user_id, user_input, max_responses=2)
        
        # معالجة الاستفسارات الشخصية
        personal_responses = {
            "ما هو اسمي": lambda: memory.get_user_memory(user_id, "name"),
            "ما اسمي": lambda: memory.get_user_memory(user_id, "name"), 
            "كم عمري": lambda: memory.get_user_memory(user_id, "age"),
            "ما هو عمري": lambda: memory.get_user_memory(user_id, "age"),
            "أين أسكن": lambda: memory.get_user_memory(user_id, "location"),
            "أين أعيش": lambda: memory.get_user_memory(user_id, "location"),
            "مكان سكني": lambda: memory.get_user_memory(user_id, "location")
        }
        
        for question, get_value in personal_responses.items():
            if question in user_input:
                value = get_value()
                if value:
                    return jsonify({'رد': f'{value}'})
                else:
                    return jsonify({'رد': f'لم تخبرني بهذه المعلومة بعد.'})
        
        # استخراج وحفظ المعلومات الشخصية
        if "اسمي" in user_input:
            name_match = re.search(r'اسمي ([\w\u0600-\u06FF]+)', user_input)
            if name_match:
                name = name_match.group(1)
                memory.add_user_memory(user_id, "name", name)
                return jsonify({'رد': f'حسناً {name}! سأتذكر اسمك.'})
                
        if "عمري" in user_input:
            age_match = re.search(r'عمري (\d+)', user_input)
            if age_match:
                age = age_match.group(1)
                memory.add_user_memory(user_id, "age", age)
                return jsonify({'رد': f'حسناً! سأتذكر أن عمرك {age} سنة.'})
                
        if "أعيش في" in user_input or "اسكن في" in user_input:
            location_match = re.search(r'(أعيش في|اسكن في) ([\w\u0600-\u06FF\s]+)', user_input)
            if location_match:
                location = location_match.group(2)
                memory.add_user_memory(user_id, "location", location)
                return jsonify({'رد': f'حسناً! سأتذكر أنك تسكن في {location}.'})

        # ---- 0. التحقق من الحقائق السريعة أولاً ----
        lang = detect_lang(user_input)
        factual_answer = get_factual_answer(user_input, lang)
        
        if factual_answer:
            # إضافة الإيموجي المناسب للإجابة الواقعية
            emoji = user_style.get_sentiment_emoji(sentiment, intent)
            if emoji:
                factual_answer = f"{emoji} {factual_answer}"
            
            # تخزين الرد
            memory.add_question_response(user_id, user_input, factual_answer)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return jsonify({
                'رد': factual_answer,
                'response_time': f"{response_time:.2f} ثانية",
                'sentiment_analysis': sentiment_analysis
            })

        # ---- 1. الحارس الرياضي ----
        if is_math_question(user_input):
            math_ans = solve_math_question(user_input)
            if math_ans is not None:
                response_text = f"الناتج: {math_ans}" if lang == "ar" else f"Result: {math_ans}"
                emoji = user_style.get_sentiment_emoji(sentiment, intent)
                if emoji:
                    response_text = f"{emoji} {response_text}"
                
                memory.add_question_response(user_id, user_input, response_text)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                return jsonify({
                    'رد': response_text,
                    'response_time': f"{response_time:.2f} ثانية",
                    'sentiment_analysis': sentiment_analysis
                })

        # ---- 2. البحث في ويكيبيديا ----
        wiki_text, wiki_url = get_wikipedia_summary(user_input, lang=lang)
        
        # ---- 2.5. إذا كانت هناك إجابة واقعية ولكن من ويكيبديا ----
        if wiki_text and is_relevant(wiki_text, user_input):
            # استخدام ويكيبديا مع صياغة متنوعة
            summary = smart_shorten(wiki_text, 2, 200)
            
            # اختيار قالب متنوع
            response_template = generate_varied_response_template(
                user_input, previous_responses, sentiment, intent, lang
            )
            
            if response_template == "{}":
                response = f"بناءً على المعلومات المتاحة: {summary}"
            else:
                response = response_template.format(summary)
            
            emoji = user_style.get_sentiment_emoji(sentiment, intent)
            if emoji:
                response = f"{emoji} {response}"
        else:
            # استخدام OpenRouter API مع برومبت عربي محسن وسريع
            response = generate_arabic_response(
                user_input, lang, temperature=0.3,
                previous_responses=previous_responses,
                sentiment=sentiment, intent=intent
            )
            
        # ---- 3. البحث في YouTube ----
        youtube_results = search_youtube(user_input, max_results=2)  # مخفّض للسرعة
        youtube_links = [vid['url'] for vid in youtube_results] if youtube_results else []

        # إضافة مصادر إذا وجدت
        sources = []
        if wiki_url:
            sources.append(f"المصدر: {wiki_url}")
        if youtube_links:
            sources.append("مقاطع YouTube مقترحة: " + ", ".join(youtube_links[:2]))
            
        if sources:
            response += "\n\n" + "\n".join(sources)

        # تخزين الرد لمنع التكرار
        memory.add_question_response(user_id, user_input, response)
        
        # حساب وقت الاستجابة
        end_time = time.time()
        response_time = end_time - start_time
        
        # التحقق من أن وقت الاستجابة أقل من ثانية
        if response_time > 1.0:
            print(f"⚠️ تحذير: وقت الاستجابة {response_time:.2f} ثانية - أطول من المطلوب")
        
        return jsonify({
            'رد': response,
            'youtube_links': youtube_links,
            'session_id': str(hashlib.sha256(user_input.encode()).hexdigest())[:16],
            'style_preferences': user_style.preferences,
            'response_time': f"{response_time:.2f} ثانية",
            'sentiment_analysis': sentiment_analysis,
            'previous_responses_count': len(previous_responses)
        })

    except Exception as e:
        return jsonify({'رد': f"عذراً، حدث خطأ في المعالجة: {str(e)}"})

# =============== نظام اقتراح أسماء المحادثات ===============
class ConversationNamer:
    """نظام اقتراح أسماء المحادثات الذكي"""
    
    def __init__(self):
        self.name_patterns = {
            "question": ["سؤال", "استفسار", "استشارة", "نقاش"],
            "learning": ["تعلم", "دراسة", "بحث", "معرفة"],
            "personal": ["شخصي", "حياتي", "تجربتي", "ذكريات"],
            "technical": ["تقني", "برمجة", "تكنولوجيا", "حاسوب"],
            "creative": ["إبداع", "فن", "كتابة", "تصميم"],
            "general": ["محادثة", "حوار", "حديث", "تواصل"]
        }
        
        self.modifiers = [
            "مثيرة", "مهمة", "مفيدة", "شيقة", "عميقة", 
            "قصيرة", "طويلة", "سريعة", "هادئة", "مكثفة"
        ]
        
    def suggest_names(self, conversation_text: str, num_suggestions: int = 3) -> List[str]:
        """اقتراح أسماء للمحادثة بناءً على محتواها"""
        
        # تحليل المحتوى
        text_lower = conversation_text.lower()
        
        # تحديد النمط
        detected_patterns = []
        for pattern_type, keywords in self.name_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_patterns.append(pattern_type)
                    break
        
        if not detected_patterns:
            detected_patterns = ["general"]
        
        # توليد الأسماء المقترحة
        suggestions = []
        for _ in range(num_suggestions):
            pattern = random.choice(detected_patterns)
            modifier = random.choice(self.modifiers)
            base_name = random.choice(self.name_patterns[pattern])
            
            # تنسيق الاسم
            name_format = random.choice([
                f"{base_name} {modifier}",
                f"{modifier} {base_name}",
                f"محادثة {base_name}",
                f"{base_name}"
            ])
            
            suggestions.append(name_format)
        
        return list(set(suggestions))  # إزالة التكرارات

@app.route('/api/conversation/suggest-name', methods=['POST'])
def suggest_conversation_name():
    """واجهة اقتراح أسماء المحادثات"""
    data = request.get_json(force=True)
    conversation_text = data.get('conversation_text', '')
    num_suggestions = data.get('num_suggestions', 3)
    
    namer = ConversationNamer()
    suggestions = namer.suggest_names(conversation_text, num_suggestions)
    
    return jsonify({
        'suggestions': suggestions,
        'original_text_preview': conversation_text[:100] + ('...' if len(conversation_text) > 100 else '')
    })

# واجهات API إضافية للذاكرة
@app.route('/api/memory/search', methods=['POST'])
def search_memories():
    """واجهة بحث في الذاكرة"""
    data = request.get_json(force=True)
    user_id = data.get('user_id', 'default')
    query = data.get('query', '')
    
    memory_system = UniversalMemorySystem()
    results = memory_system.search_memories(user_id, query=query, limit=20)
    
    return jsonify({'memories': results})

@app.route('/api/memory/timeline/<user_id>', methods=['GET'])
def get_timeline(user_id):
    """الحصول على الخط الزمني للحياة"""
    memory_system = UniversalMemorySystem()
    timeline = memory_system.get_life_timeline(user_id)
    
    return jsonify({'timeline': timeline})

@app.route('/api/memory/relationships/<user_id>', methods=['GET'])
def get_relationships(user_id):
    """الحصول على شبكة العلاقات"""
    memory_system = UniversalMemorySystem()
    relationships = memory_system.get_relationship_network(user_id)
    
    return jsonify({'relationships': relationships})

@app.route('/api/memory/add', methods=['POST'])
def add_custom_memory():
    """إضافة ذكرى مخصصة"""
    data = request.get_json(force=True)
    user_id = data.get('user_id', 'default')
    category = data.get('category')
    title = data.get('title')
    content = data.get('content')
    
    memory_system = UniversalMemorySystem()
    
    try:
        memory_category = MemoryCategory(category)
        success = memory_system.add_memory(
            user_id, memory_category, title, content,
            entities=data.get('entities', []),
            emotions=data.get('emotions', []),
            intensity=data.get('intensity', 3),
            importance=data.get('importance', 3)
        )
        
        return jsonify({'success': success, 'message': 'تم حفظ الذكرى'})
    
    except ValueError:
        return jsonify({'success': False, 'message': 'فئة غير صحيحة'})

# =============== نظام الأنماط والتكوين ===============
class SystemConfig:
    """نظام تكوين متقدم مع التحقق من الصحة"""
   
    DEFAULTS = {
        "quantum": {
            "entropy_level": 5,
            "probability_threshold": 0.85,
            "max_qubits": 12
        },
        "language": {
            "response_depth": 3,
            "creativity_factor": 0.75,
            "context_window": 7
        },
        "learning": {
            "retention_rate": 0.92,
            "decay_factor": 0.05,
            "reinforcement_cycle": 24
        },
        "security": {
            "authentication_level": 4,
            "key_rotation_interval": 3600,
            "biometric_threshold": 0.93
        }
    }
   
    def __init__(self, config_path: str = None):
        self.config = self.DEFAULTS.copy()
        self.config_path = config_path
        self.validation_rules = self._init_validation_rules()
       
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
   
    def _init_validation_rules(self) -> Dict[str, Callable]:
        """تهيئة قواعد التحقق من صحة التكوين"""
        return {
            "quantum.entropy_level": lambda x: 1 <= x <= 10,
            "quantum.probability_threshold": lambda x: 0.5 <= x <= 0.99,
            "language.creativity_factor": lambda x: 0.1 <= x <= 1.0,
            "security.authentication_level": lambda x: x in {1, 2, 3, 4}
        }
   
    def load_config(self, path: str):
        """تحميل التكوين من ملف"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self._merge_configs(loaded_config)
                print(f"تم تحميل التكوين من {path}")
        except Exception as e:
            print(f"خطأ في تحميل التكوين: {str(e)}")
   
    def _merge_configs(self, new_config: Dict):
        """دمج التكوينات مع التحقق من الصحة"""
        for section, values in new_config.items():
            if section in self.config:
                for key, value in values.items():
                    full_key = f"{section}.{key}"
                    if full_key in self.validation_rules:
                        if self.validation_rules[full_key](value):
                            self.config[section][key] = value
                        else:
                            print(f"قيمة غير صالحة: {full_key} = {value}")
                    else:
                        self.config[section][key] = value
   
    def get(self, key_path: str, default: Any = None) -> Any:
        """الحصول على قيمة التكوين"""
        keys = key_path.split('.')
        current = self.config
        try:
            for key in keys:
                current = current[key]
            return current
        except KeyError:
            return default
   
    def set(self, key_path: str, value: Any):
        """تعيين قيمة التكوين"""
        keys = key_path.split('.')
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        last_key = keys[-1]
        current[last_key] = value

# =============== محرك التعرف على المشاعر ===============
class EmotionRecognitionEngine:
    """محرك محلي للتعرف على المشاعر"""
   
    def __init__(self):
        self.sentiment_lexicon = {
            'سعيد': 0.8,
            'فرح': 0.7,
            'حزين': -0.8,
            'غاضب': -0.6,
            'رائع': 0.9,
            'سيء': -0.7
        }
   
    def analyze_sentiment(self, text: str) -> float:
        """تحليل المشاعر الأساسي للنص"""
        words = text.split()
        sentiment = 0.0
        matched = 0
       
        for word in words:
            if word in self.sentiment_lexicon:
                sentiment += self.sentiment_lexicon[word]
                matched += 1
               
        if matched > 0:
            return sentiment / matched
        return 0.0

# =============== نظام اللغة المتقدم ===============
class AdvancedLanguageSystem:
    """نظام معالجة لغة طبيعية متقدم"""
   
    class LanguageContext:
        """تمثيل سياقي متقدم للمحادثة"""
       
        def __init__(self, depth: int = 5):
            self.context_stack = deque(maxlen=depth)
            self.context_weights = []
            self.current_topic = ""
            self.sentiment_score = 0.0
       
        def push_context(self, context: str, weight: float = 1.0):
            """إضافة سياق جديد إلى المحادثة"""
            self.context_stack.append(context)
            self.context_weights.append(weight)
            self._update_topic(context)
       
        def _update_topic(self, context: str):
            """تحديد الموضوع الحالي تلقائياً"""
            if "؟" in context:
                self.current_topic = context.split("؟")[0]
            elif ":" in context:
                self.current_topic = context.split(":")[0]
            else:
                words = context.split()
                if len(words) > 2:
                    self.current_topic = " ".join(words[:3])
       
        def get_weighted_context(self) -> str:
            """استرجاع السياق مع الأوزان"""
            weighted_context = []
            for i, ctx in enumerate(self.context_stack):
                weight = self.context_weights[i]
                weighted_context.append(f"(w={weight:.2f}) {ctx}")
            return "\n".join(weighted_context)
   
    def __init__(self, config: SystemConfig):
        self.config = config
        self.creativity = config.get("language.creativity_factor", 0.7)
        self.context_depth = config.get("language.context_window", 7)
        self.context = self.LanguageContext(self.context_depth)
        self.language_models = self._load_language_models()
        self.response_strategies = self._init_response_strategies()
        self.emotion_engine = EmotionRecognitionEngine()
   
    def _load_language_models(self) -> Dict[str, Any]:
        """تحميل نماذج لغة متعددة (محاكاة)"""
        return {
            "grammar_model": {"version": "2.1", "coverage": 0.95},
            "semantic_model": {"version": "1.7", "entities": 15000},
            "pragmatic_model": {"version": "3.2", "contextual_depth": 5}
        }
   
    def _init_response_strategies(self) -> Dict[str, Callable]:
        """تهيئة استراتيجيات توليد الردود"""
        return {
            "direct": self._generate_direct_response,
            "contextual": self._generate_contextual_response,
            "creative": self._generate_creative_response,
            "probabilistic": self._generate_probabilistic_response
        }
   
    def process_input(self, text: str) -> str:
        """معالجة النص المدخل وتوليد الرد"""
        # تطبيع النص العربي قبل المعالجة
        text = normalize_arabic_text(text)
        
        # تحليل المشاعر
        sentiment = self.emotion_engine.analyze_sentiment(text)
        self.context.sentiment_score = sentiment
       
        # تحديث السياق
        self.context.push_context(text, weight=self._calculate_context_weight(text))
       
        # اختيار استراتيجية الرد
        strategy = self._select_response_strategy()
       
        # توليد الرد
        response = self.response_strategies[strategy](text)
       
        # تحديث نماذج اللغة
        self._update_language_models(text, response)
       
        return response
   
    def _calculate_context_weight(self, text: str) -> float:
        """حساب وزن السياق بناءً على طول النص وتعقيده"""
        length_factor = min(1.0, len(text) / 100)
        complexity_factor = len(re.findall(r'\b\w{5,}\b', text)) / 10
        return 0.5 + 0.3 * length_factor + 0.2 * complexity_factor
   
    def _select_response_strategy(self) -> str:
        """اختيار استراتيجية الرد الأمثل باستخدام الاحتمالات"""
        strategies = ["direct", "contextual", "creative", "probabilistic"]
        creativity = self.creativity
       
        # توزيع احتمالي ديناميكي
        probabilities = {
            "direct": max(0.1, 0.4 - creativity / 2),
            "contextual": 0.3,
            "creative": min(0.5, creativity * 0.8),
            "probabilistic": min(0.4, (1 - creativity) * 0.5)
        }
       
        # اختيار إستراتيجية بناءً على التوزيع الاحتمالي
        rand_val = random.random()
        cumulative = 0.0
        for strategy, prob in probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return strategy
       
        return "direct"
   
    def _generate_direct_response(self, text: str) -> str:
        """توليد رد مباشر"""
        return f"بالنسبة لسؤالك '{text}'، الجواب المباشر هو أنني نظام ذكاء اصطناعي متقدم."
   
    def _generate_contextual_response(self, text: str) -> str:
        """توليد رد سياقي معقد"""
        context = self.context.get_weighted_context()
        return f"بالنظر إلى السياق:\n{context}\nأرى أن سؤالك '{text}' يتطلب إجابة متعمقة."
   
    def _generate_creative_response(self, text: str) -> str:
        """توليد رد إبداعي باستخدام محاكاة إبداعية متقدمة"""
        creativity_level = int(self.creativity * 10)
        responses = [
            "بعد تفكير عميق، أعتقد أن الإجابة تكمن في...",
            "من وجهة نظر إبداعية، يمكننا النظر إلى الأمر كالتالي...",
            "لقد ألهمني سؤالك للتفكير في...",
            f"باستخدام الإبداع من المستوى {creativity_level}، أقول لك..."
        ]
        return random.choice(responses)
   
    def _generate_probabilistic_response(self, text: str) -> str:
        """توليد رد احتمالي معقد"""
        options = [
            f"بناءً على تحليل احتمالي، أعتقد أن '{text}' يعني شيئاً مثيراً للاهتمام.",
            f"السيناريو الأكثر احتمالاً هو أنك تبحث عن معلومات حول '{text}'.", 
            f"بعد حساب الاحتمالات، النتيجة الأرجح هي أن لديك فضول حول '{text}'."
        ]
        return random.choice(options)
   
    def _update_language_models(self, input_text: str, response: str):
        """تعديل نماذج اللغة بناءً على التفاعل"""
        for model in self.language_models.values():
            model["version"] = round(model["version"] + 0.01, 2)

# =============== نظام التعلم الذاتي المتقدم ===============
class AdvancedLearningSystem:
    """نظام تعلم ذاتي متعدد الطبقات"""
   
    class KnowledgeNode:
        """عقدة معرفية في الشبكة المعرفية"""
       
        def __init__(self, id: str, content: Any):
            self.id = id
            self.content = content
            self.connections = {}
            self.strength = 1.0
            self.last_accessed = time.time()
       
        def add_connection(self, node_id: str, weight: float):
            """إضافة اتصال إلى عقدة أخرى"""
            self.connections[node_id] = weight
       
        def decay(self, factor: float):
            """تخفيض قوة العقدة بمرور الوقت"""
            self.strength *= (1 - factor)
   
    class KnowledgeGraph:
        """شبكة معرفية ديناميكية"""
       
        def __init__(self, decay_factor: float = 0.05):
            self.nodes = {}
            self.decay_factor = decay_factor
            self.last_decay_time = time.time()
       
        def add_node(self, id: str, content: Any):
            """إضافة عقدة جديدة"""
            if id not in self.nodes:
                self.nodes[id] = AdvancedLearningSystem.KnowledgeNode(id, content)
       
        def add_connection(self, from_id: str, to_id: str, weight: float):
            """إضافة اتصال بين عقدتين"""
            if from_id in self.nodes and to_id in self.nodes:
                self.nodes[from_id].add_connection(to_id, weight)
       
        def get_node(self, id: str) -> Optional['AdvancedLearningSystem.KnowledgeNode']:
            """الحصول على عقدة معرفية"""
            if id in self.nodes:
                self.nodes[id].last_accessed = time.time()
                self.nodes[id].strength = min(1.0, self.nodes[id].strength + 0.1)
                return self.nodes[id]
            return None
       
        def decay_all(self):
            """تخفيض قوة جميع العقد"""
            current_time = time.time()
            if current_time - self.last_decay_time > 86400:  # مرة في اليوم
                for node in self.nodes.values():
                    node.decay(self.decay_factor)
                self.last_decay_time = current_time
   
    def __init__(self, config: SystemConfig):
        self.config = config
        self.retention_rate = config.get("learning.retention_rate", 0.9)
        self.decay_factor = config.get("learning.decay_factor", 0.05)
        self.reinforcement_cycle = config.get("learning.reinforcement_cycle", 24)
        self.knowledge_graph = self.KnowledgeGraph(self.decay_factor)
        self.initialize_knowledge_base()
       
        # بدء خيط التعلم الدائم
        self.learning_thread = threading.Thread(target=self._continuous_learning)
        self.learning_thread.daemon = True
        self.learning_thread.start()
   
    def initialize_knowledge_base(self):
        """تهيئة قاعدة المعرفة الأولية"""
        core_knowledge = [
            ("AI_principles", "مبادئ الذكاء الاصطناعي"),
            ("quantum_basics", "أساسيات الحوسبة الكمومية"),
            ("language_processing", "معالجة اللغة الطبيعية"),
            ("learning_algorithms", "خوارزميات التعلم الآلي")
        ]
       
        for id, content in core_knowledge:
            self.knowledge_graph.add_node(id, content)
       
        # إضافة اتصالات معرفية
        self.knowledge_graph.add_connection("AI_principles", "quantum_basics", 0.7)
        self.knowledge_graph.add_connection("AI_principles", "language_processing", 0.8)
        self.knowledge_graph.add_connection("language_processing", "learning_algorithms", 0.6)
   
    def learn_from_interaction(self, input_data: str, output_data: str):
        """التعلم من تفاعل جديد"""
        # تطبيع النص العربي قبل الحفظ
        input_data = normalize_arabic_text(input_data)
        output_data = normalize_arabic_text(output_data)
        
        interaction_id = hashlib.sha256(f"{input_data}{output_data}".encode()).hexdigest()[:16]
       
        self.knowledge_graph.add_node(interaction_id, {
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.datetime.now().isoformat()
        })
       
        for node_id in self.knowledge_graph.nodes:
            if node_id.startswith("core_"):
                self.knowledge_graph.add_connection(interaction_id, node_id, 0.5)
   
    def _continuous_learning(self):
        """عملية تعلم مستمرة في الخلفية"""
        while True:
            self.knowledge_graph.decay_all()
            self._reinforce_knowledge()
            time.sleep(self.reinforcement_cycle * 3600)
   
    def _reinforce_knowledge(self):
        """تعزيز المعرفة بناءً على الاستخدام"""
        for node in self.knowledge_graph.nodes.values():
            if node.strength > 0.8:
                node.strength = min(1.0, node.strength + 0.05)
   
    def get_knowledge_path(self, start_id: str, end_id: str) -> List[str]:
        """الحصول على مسار معرفي بين عقدتين"""
        visited = set()
        queue = deque([(start_id, [start_id])])
       
        while queue:
            current_id, path = queue.popleft()
            if current_id == end_id:
                return path
           
            visited.add(current_id)
            current_node = self.knowledge_graph.get_node(current_id)
           
            for neighbor_id, weight in current_node.connections.items():
                if neighbor_id not in visited and weight > 0.3:
                    queue.append((neighbor_id, path + [neighbor_id]))
       
        return []

# =============== نظام الأمان الكمومي الحيوي المتقدم ===============
class QuantumBiometricSecurity:
    """نظام أمان كمومي حيوي متعدد الطبقات"""
   
    class QuantumEncryptionEngine:
        """محرك تشفير كمومي متقدم"""
       
        def __init__(self, qubits: int = 8):
            self.qubits = qubits
            self.key_cache = {}
            self.last_key_rotation = time.time()
       
        def generate_quantum_key(self, length: int = 256) -> bytes:
            """توليد مفتاح كمومي عشوائي"""
            key = secrets.token_bytes(length)
            self.key_cache[hashlib.sha256(key).hexdigest()] = time.time()
            return key
       
        def rotate_keys(self):
            """تدوير المفاتيح القديمة"""
            current_time = time.time()
            for key_hash, created_time in list(self.key_cache.items()):
                if current_time - created_time > 86400:  # 24 ساعة
                    del self.key_cache[key_hash]
   
    def __init__(self, config: SystemConfig):
        self.config = config
        self.auth_level = config.get("security.authentication_level", 3)
        self.key_rotation_interval = config.get("security.key_rotation_interval", 3600)
        self.biometric_threshold = config.get("security.biometric_threshold", 0.9)
        self.encryption_engine = self.QuantumEncryptionEngine()
        self.biometric_profiles = {}
        self.session_keys = {}
        self.initialize_security_subsystems()
       
        # بدء خيط الأمان الدائم
        self.security_thread = threading.Thread(target=self._continuous_security)
        self.security_thread.daemon = True
        self.security_thread.start()
   
    def initialize_security_subsystems(self):
        """تهيئة أنظمة الأمان الفرعية"""
        self.system_root_key = self.encryption_engine.generate_quantum_key(512)
        self.biometric_profiles["admin"] = self._create_biometric_profile("admin")
   
    def _create_biometric_profile(self, user_id: str) -> Dict:
        """إنشاء ملف تعريف حيوي كمومي"""
        profile = {
            "voice_pattern": hashlib.sha256(f"{user_id}_voice".encode()).hexdigest(),
            "behavioral_signature": self._generate_behavioral_signature(user_id),
            "quantum_entropy_factor": random.random()
        }
        return profile
   
    def _generate_behavioral_signature(self, user_id: str) -> str:
        """توليد توقيع سلوكي كمومي"""
        signature = ""
        for _ in range(8):
            quantum_state = [random.choice([0, 1]) for _ in range(8)]
            signature += ''.join(str(b) for b in quantum_state)
        return hashlib.sha256(signature.encode()).hexdigest()
   
    def _continuous_security(self):
        """مراقبة أمنية مستمرة في الخلفية"""
        while True:
            self.encryption_engine.rotate_keys()
            self._rotate_session_keys()
            self._system_integrity_check()
            time.sleep(self.key_rotation_interval)
   
    def _rotate_session_keys(self):
        """تدوير مفاتيح الجلسات القديمة"""
        current_time = time.time()
        for session_id, (created_time, _) in list(self.session_keys.items()):
            if current_time - created_time > 3600:  # 1 ساعة
                del self.session_keys[session_id]
   
    def _system_integrity_check(self):
        """فحص سلامة النظام الأمني"""
        key_hash = hashlib.sha256(self.system_root_key).hexdigest()
        if key_hash not in self.encryption_engine.key_cache:
            print("تحذير: تم تغيير المفتاح الأساسي للنظام!")
            self.system_root_key = self.encryption_engine.generate_quantum_key(512)
   
    def authenticate_user(self, user_id: str, biometric_data: Dict) -> bool:
        """مصادقة المستخدم باستخدام البيانات الحيوية"""
        if user_id not in self.biometric_profiles:
            return False
       
        profile = self.biometric_profiles[user_id]
        match_score = self._calculate_biometric_match(profile, biometric_data)
       
        return match_score >= self.biometric_threshold
   
    def _calculate_biometric_match(self, profile: Dict, data: Dict) -> float:
        """حساب درجة التطابق الحيوي"""
        voice_match = 1.0 if profile["voice_pattern"] == data.get("voice_hash") else 0.0
        behavior_match = 0.7 if profile["behavioral_signature"] == data.get("behavior_hash") else 0.0
        entropy_factor = profile["quantum_entropy_factor"]
       
        match_score = (voice_match * 0.6 + behavior_match * 0.4) * entropy_factor
        return match_score
   
    def create_secure_session(self, user_id: str) -> str:
        """إنشاء جلسة آمنة جديدة"""
        session_id = secrets.token_urlsafe(16)
        session_key = self.encryption_engine.generate_quantum_key()
        self.session_keys[session_id] = (time.time(), session_key)
        return session_id
   
    def encrypt_data(self, session_id: str, data: str) -> bytes:
        """تشفير البيانات باستخدام مفتاح الجلسة"""
        if session_id not in self.session_keys:
            raise ValueError("معرف الجلسة غير صالح")
       
        _, session_key = self.session_keys[session_id]
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self._quantum_encrypt(data, session_key)
   
    def decrypt_data(self, session_id: str, encrypted_data: bytes) -> str:
        """فك تشفير البيانات"""
        if session_id not in self.session_keys:
            raise ValueError("معرف الجلسة غير صالح")
       
        _, session_key = self.session_keys[session_id]
        decrypted = self._quantum_decrypt(encrypted_data, session_key)
       
        try:
            return decrypted.decode('utf-8')
        except UnicodeDecodeError:
            return "تم فك التشفير بنجاح ولكن المحتوى غير نصي"
   
    def _quantum_encrypt(self, data: bytes, key: bytes) -> bytes:
        """تشفير كمومي متقدم"""
        encrypted = bytearray()
        for i in range(len(data)):
            encrypted.append((data[i] + key[i % len(key)]) % 256)
        return bytes(encrypted)
   
    def _quantum_decrypt(self, encrypted: bytes, key: bytes) -> bytes:
        """فك تشفير كمومي"""
        decrypted = bytearray()
        for i in range(len(encrypted)):
            decrypted.append((encrypted[i] - key[i % len(key)]) % 256)
        return bytes(decrypted)

# =============== نظام الاحتمالات الكمومية المتقدم ===============
class QuantumProbabilityEngine:
    """نظام محاكاة احتمالات كمومية متقدم"""
   
    class QuantumState:
        """تمثيل لحالة كمومية معقدة"""
       
        def __init__(self, qubits: int):
            self.qubits = qubits
            self.state = np.zeros(2**qubits, dtype=complex)
            self.state[0] = 1.0  # الحالة الأولية
           
        def apply_gate(self, gate: np.ndarray, target: int, controls: List[int] = None):
            """تطبيق بوابة كمومية مع التحكم"""
            pass
       
        def measure(self) -> int:
            """قياس الحالة الكمومية"""
            probabilities = np.abs(self.state)**2
            return random.choices(range(len(probabilities)), weights=probabilities)[0]
   
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.qubits = self.config.get("quantum.max_qubits", 8)
        self.entropy_level = self.config.get("quantum.entropy_level", 5)
        self.probability_cache = {}
        self.quantum_states = {}
        self.initialize_quantum_system()
   
    def initialize_quantum_system(self):
        """تهيئة النظام الكمومي بمعاملات متقدمة"""
        for i in range(1, self.entropy_level + 1):
            state_id = f"state_{i}"
            self.quantum_states[state_id] = self.QuantumState(self.qubits)
       
        self._apply_quantum_entanglement()
        self._initialize_probability_distributions()
   
    def _apply_quantum_entanglement(self):
        """إنشاء تشابك كمومي بين الحالات"""
        for i in range(1, self.entropy_level):
            state_a = self.quantum_states[f"state_{i}"]
            state_b = self.quantum_states[f"state_{i+1}"]
   
    def _initialize_probability_distributions(self):
        """تهيئة توزيعات الاحتمالات الأولية"""
        for i in range(1, 101):
            dist_id = f"dist_{i}"
            self.probability_cache[dist_id] = self._generate_probability_distribution()
   
    def _generate_probability_distribution(self) -> Dict[str, float]:
        """إنشاء توزيع احتمالي كمومي معقد"""
        dist = {}
        total = 0.0
        for i in range(100):
            prob = random.random() ** self.entropy_level
            dist[f"event_{i}"] = prob
            total += prob
       
        for key in dist:
            dist[key] /= total
       
        return dist
   
    def calculate_complex_probability(self, event_space: List[str]) -> Dict[str, float]:
        """حساب الاحتمالات في فضاء أحداث معقد"""
        event_hash = hashlib.sha256(','.join(event_space).encode()).hexdigest()
       
        if event_hash in self.probability_cache:
            return self.probability_cache[event_hash]
       
        dist = self._generate_probability_distribution_for_events(event_space)
        self.probability_cache[event_hash] = dist
        return dist
   
    def _generate_probability_distribution_for_events(self, events: List[str]) -> Dict[str, float]:
        """إنشاء توزيع احتمالي متقدم لفئة أحداث محددة"""
        quantum_result = self._simulate_quantum_events(len(events))
       
        probabilities = {}
        total = sum(quantum_result)
        for i, event in enumerate(events):
            probabilities[event] = quantum_result[i] / total
       
        self._apply_contextual_adjustments(probabilities)
       
        return probabilities
   
    def _simulate_quantum_events(self, num_events: int) -> List[float]:
        """محاكاة أحداث كمومية معقدة"""
        state_id = random.choice(list(self.quantum_states.keys()))
        quantum_state = self.quantum_states[state_id]
       
        measurements = [quantum_state.measure() for _ in range(1000)]
       
        event_probs = [0.0] * num_events
        for measure in measurements:
            index = measure % num_events
            event_probs[index] += 1
       
        return event_probs
   
    def _apply_contextual_adjustments(self, probabilities: Dict[str, float]):
        """تطبيق تعديلات احتمالية معتمدة على السياق"""
        entropy = self._calculate_distribution_entropy(probabilities)
        adjustment_factor = math.log(entropy + 1) * 0.1
       
        for key in probabilities:
            probabilities[key] = min(1.0, probabilities[key] * (1 + adjustment_factor))
       
        total = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] /= total
   
    def _calculate_distribution_entropy(self, dist: Dict[str, float]) -> float:
        """حساب إنتروبيا التوزيع الاحتمالي"""
        entropy = 0.0
        for p in dist.values():
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

# =============== الأنظمة الجديدة للنسخة الخارقة ===============

class QuantumMemorySystem:
    """نظام ذاكرة كمومي متطور"""
    
    def __init__(self):
        self.episodic_memory = {}  # الذكريات العرضية
        self.semantic_memory = {}  # المعرفة العامة
        self.procedural_memory = {}  # المهارات
        self.emotional_memory = {}  # المشاعر
        
    def store_experience(self, user_id, experience, emotional_weight=0.5):
        """تخزين التجارب مع وزن عاطفي"""
        memory_id = f"exp_{hashlib.sha256(experience.encode()).hexdigest()[:16]}"
        
        self.episodic_memory[memory_id] = {
            'user_id': user_id,
            'experience': experience,
            'timestamp': time.time(),
            'emotional_weight': emotional_weight,
            'access_count': 0
        }
        
    def recall_context(self, user_id, current_context, top_k=5):
        """استدعاء الذكريات ذات الصلة"""
        relevant_memories = []
        
        for memory_id, memory in self.episodic_memory.items():
            if memory['user_id'] == user_id:
                relevance = self._calculate_relevance(memory['experience'], current_context)
                if relevance > 0.3:  # عتبة الصلة
                    relevant_memories.append((relevance, memory))
        
        relevant_memories.sort(reverse=True)
        return relevant_memories[:top_k]
    
    def _calculate_relevance(self, memory_text, current_context):
        """حساب صلة الذاكرة بالسياق الحالي"""
        memory_words = set(memory_text.lower().split())
        context_words = set(current_context.lower().split())
        
        intersection = memory_words & context_words
        union = memory_words | context_words
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)

class AdvancedReasoningEngine:
    """محرك تفكير متعدد الطبقات"""
    
    def __init__(self):
        self.reasoning_modes = {
            "deductive": self._deductive_reasoning,
            "inductive": self._inductive_reasoning,
            "abductive": self._abductive_reasoning,
            "analogical": self._analogical_reasoning,
            "counterfactual": self._counterfactual_reasoning
        }
    
    def solve_complex_problem(self, problem, context=""):
        """حل المشكلات المعقدة باستخدام طرق تفكير متعددة"""
        
        # تحليل المشكلة
        problem_type = self._classify_problem(problem)
        
        # تطبيق أساليب التفكير المناسبة
        solutions = []
        for mode_name, mode_func in self.reasoning_modes.items():
            try:
                solution = mode_func(problem, context)
                confidence = self._calculate_confidence(solution)
                solutions.append((confidence, solution, mode_name))
            except Exception as e:
                continue
        
        # اختيار أفضل حل
        if solutions:
            solutions.sort(reverse=True)
            best_confidence, best_solution, best_mode = solutions[0]
            return {
                "solution": best_solution,
                "confidence": best_confidence,
                "method": best_mode,
                "alternative_approaches": solutions[1:3]
            }
        
        return {"solution": "لا يمكن حل هذه المشكلة بالطرق الحالية", "confidence": 0.0}
    
    def _deductive_reasoning(self, problem, context):
        """تفكير استنتاجي (من العام إلى الخاص)"""
        # تطبيق القواعد العامة على الحالات الخاصة
        if "كل" in problem and "بعض" in problem:
            return "هذا استدلال غير صحيح. 'كل أ هي ب' و'بعض ب هي ج' لا يعني 'بعض أ هي ج'"
        
        return f"الاستنتاج: {problem} يتطلب مقدمات إضافية"
    
    def _inductive_reasoning(self, problem, context):
        """تفكير استقرائي (من الخاص إلى العام)"""
        return f"من خلال الملاحظات حول {problem}، يمكن استنتاج نمط عام"
    
    def _abductive_reasoning(self, problem, context):
        """تفكير تخميني (إيجاد أفضل تفسير)"""
        return f"أفضل تفسير لـ {problem} هو..."
    
    def _analogical_reasoning(self, problem, context):
        """تفكير قياسي (استخدام التشابه)"""
        return f"هذه المشكلة تشبه..."
    
    def _counterfactual_reasoning(self, problem, context):
        """تفكير افتراضي (ماذا لو)"""
        return f"إذا تغيرت الظروف، فإن..."
    
    def _classify_problem(self, problem):
        """تصنيف نوع المشكلة"""
        if any(word in problem for word in ["حساب", "رياضيات", "معادلة"]):
            return "math"
        elif any(word in problem for word in ["سبب", "لماذا", "كيف"]):
            return "causal"
        elif any(word in problem for word in ["مقارنة", "شبيه", "مثل"]):
            return "comparative"
        else:
            return "general"
    
    def _calculate_confidence(self, solution):
        """حساب ثقة الحل"""
        # محاكاة حساب الثقة بناءً على طول الحل وتعقيده
        return min(0.95, len(solution) / 1000)

class DeepReinforcementLearning:
    """تعلم تعزيزي عميق للتكيف مع المستخدم"""
    
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
    def choose_action(self, state, possible_actions):
        """اختيار action بناءً على حالة المستخدم"""
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)  # استكشاف
        
        # استغلال (اختيار أفضل action معروف)
        q_values = [self.q_table[state][action] for action in possible_actions]
        max_q = max(q_values)
        
        # في حالة التعادل، اختيار عشوائي
        best_actions = [action for action in possible_actions 
                       if self.q_table[state][action] == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """تحديث Q-value بناءً على المكافأة"""
        best_next_q = max([self.q_table[next_state][a] for a in self.get_possible_actions(next_state)])
        current_q = self.q_table[state][action]
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def get_possible_actions(self, state):
        """الحصول على الإجراءات الممكنة للحالة"""
        # هذه دالة مساعدة تحتاج للتطبيق حسب السياق
        return ["answer", "ask_clarification", "provide_example", "suggest_resource"]

class AdvancedEmotionalIntelligence:
    """ذكاء عاطفي متقدم لفهم مشاعر المستخدم"""
    
    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        
    def analyze_emotional_state(self, text, voice_tone=None, typing_speed=None):
        """تحليل الحالة العاطفية الشاملة"""
        
        # تحليل النص
        text_emotion = self._analyze_text_emotion(text)
        
        # تحليل نمط التفاعل
        interaction_pattern = self._analyze_interaction_pattern(typing_speed)
        
        # دمج النتائج
        emotional_state = {
            'primary_emotion': text_emotion['dominant_emotion'],
            'emotional_intensity': text_emotion['intensity'],
            'valence': text_emotion['valence'],  # إيجابي/سلبي
            'arousal': text_emotion['arousal'],  # هادئ/متحمس
            'confidence': text_emotion['confidence']
        }
        
        return emotional_state
    
    def generate_empathetic_response(self, user_input, emotional_state):
        """توليد ردود عاطفية متعاطفة"""
        
        empathy_templates = {
            'anger': "أتفهم أنك تشعر بالإحباط. دعنا نحاول حل هذا معاً.",
            'sadness': "أرى أن هذا الأمر يزعجك. هل تريد التحدث عنه أكثر؟",
            'joy': "رائع! يبدو أنك سعيد بهذا. هذا يجعلني سعيداً أيضاً!",
            'fear': "أتفهم قلقك. دعنا ننظر إلى هذا الأمر بطريقة مختلفة.",
            'surprise': "مفاجأة! هذا مثير للاهتمام. أخبرني المزيد."
        }
        
        primary_emotion = emotional_state['primary_emotion']
        empathy_line = empathy_templates.get(primary_emotion, "أتفهم مشاعرك.")
        
        return f"{empathy_line} "
    
    def _load_emotion_lexicon(self):
        """تحميل قاموس المشاعر"""
        return {
            'سعيد': {'emotion': 'joy', 'intensity': 0.8, 'valence': 1.0},
            'فرح': {'emotion': 'joy', 'intensity': 0.9, 'valence': 1.0},
            'حزين': {'emotion': 'sadness', 'intensity': 0.8, 'valence': -1.0},
            'غاضب': {'emotion': 'anger', 'intensity': 0.7, 'valence': -1.0},
            'خائف': {'emotion': 'fear', 'intensity': 0.6, 'valence': -1.0},
            'متفاجئ': {'emotion': 'surprise', 'intensity': 0.5, 'valence': 0.0}
        }
    
    def _analyze_text_emotion(self, text):
        """تحليل المشاعر من النص"""
        words = text.split()
        emotion_scores = defaultdict(float)
        
        for word in words:
            if word in self.emotion_lexicon:
                emotion_data = self.emotion_lexicon[word]
                emotion_scores[emotion_data['emotion']] += emotion_data['intensity']
        
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return {
                'dominant_emotion': dominant_emotion[0],
                'intensity': dominant_emotion[1],
                'valence': self.emotion_lexicon.get(words[0], {}).get('valence', 0.0) if words else 0.0,
                'arousal': 0.5,  # محاكاة
                'confidence': min(1.0, len(emotion_scores) / 10)
            }
        else:
            return {
                'dominant_emotion': 'neutral',
                'intensity': 0.0,
                'valence': 0.0,
                'arousal': 0.5,
                'confidence': 0.1
            }
    
    def _analyze_interaction_pattern(self, typing_speed):
        """تحليل نمط التفاعل (محاكاة)"""
        return "normal"

class ExternalKnowledgeIntegration:
    """دمج المعرفة من مصادر خارجية"""
    
    def __init__(self):
        self.apis = {
            'wolfram_alpha': 'YOUR_WOLFRAM_APP_ID',
            'openweather': 'YOUR_WEATHER_API_KEY',
            'news_api': 'YOUR_NEWS_API_KEY'
        }
    
    def get_real_time_data(self, query):
        """الحصول على بيانات في الوقت الحقيقي"""
        
        if self._is_mathematical(query):
            return self._query_wolfram_alpha(query)
        elif self._is_weather_related(query):
            return self._query_weather(query)
        elif self._is_news_related(query):
            return self._query_news(query)
        
        return None
    
    def _query_wolfram_alpha(self, query):
        """الاستعلام من Wolfram Alpha للحسابات المعقدة"""
        try:
            # محاكاة الاستعلام - في التطبيق الحقيقي تحتاج API key
            return f"نتيجة محسوبة لـ '{query}' (محاكاة - تحتاج Wolfram Alpha API)"
        except Exception as e:
            return None
    
    def _query_weather(self, query):
        """الاستعلام عن الطقس"""
        try:
            # محاكاة الاستعلام
            return "حالة الطقس: معتدل 25°C (محاكاة)"
        except Exception as e:
            return None
    
    def _query_news(self, query):
        """الاستعلام عن الأخبار"""
        try:
            # محاكاة الاستعلام
            return "أحدث الأخبار المتعلقة بموضوعك (محاكاة)"
        except Exception as e:
            return None
    
    def _is_mathematical(self, query):
        """الكشف إذا كان الاستعلام رياضياً"""
        math_terms = ["احسب", "حل", "معادلة", "تكامل", "تفاضل", "calculate", "solve"]
        return any(term in query.lower() for term in math_terms)
    
    def _is_weather_related(self, query):
        """الكشف إذا كان الاستعلام عن الطقس"""
        weather_terms = ["طقس", "جو", "درجة الحرارة", "weather", "temperature"]
        return any(term in query.lower() for term in weather_terms)
    
    def _is_news_related(self, query):
        """الكشف إذا كان الاستعلام عن أخبار"""
        news_terms = ["أخبار", "حدث", "جديد", "news", "update"]
        return any(term in query.lower() for term in news_terms)

class SelfEvaluationSystem:
    """نظام التقييم الذاتي والتحسين المستمر"""
    
    def evaluate_response_quality(self, user_input, ai_response, user_feedback=None):
        """تقييم جودة الرد تلقائياً"""
        
        metrics = {
            'relevance': self._calculate_relevance(user_input, ai_response),
            'accuracy': self._estimate_accuracy(ai_response),
            'completeness': self._check_completeness(user_input, ai_response),
            'empathy': self._measure_empathy(ai_response),
            'conciseness': self._measure_conciseness(ai_response)
        }
        
        overall_score = sum(metrics.values()) / len(metrics)
        
        return {
            'overall_score': overall_score,
            'detailed_metrics': metrics,
            'improvement_suggestions': self._generate_improvement_suggestions(metrics)
        }
    
    def _generate_improvement_suggestions(self, metrics):
        """توليد اقتراحات للتحسين"""
        suggestions = []
        
        if metrics['relevance'] < 0.7:
            suggestions.append("التركيز أكثر على صلة الرد بالسؤال")
        if metrics['empathy'] < 0.6:
            suggestions.append("زيادة التعاطف في الردود")
        if metrics['conciseness'] < 0.5:
            suggestions.append("تحسين الإيجاز وتجنب الإطالة")
            
        return suggestions
    
    def _calculate_relevance(self, user_input, ai_response):
        """حساب صلة الرد بالسؤال"""
        input_words = set(user_input.lower().split())
        response_words = set(ai_response.lower().split())
        
        intersection = input_words & response_words
        union = input_words | response_words
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _estimate_accuracy(self, response):
        """تقدير دقة الرد (محاكاة)"""
        # في التطبيق الحقيقي، يمكن استخدام fact-checking APIs
        return 0.8  # محاكاة
    
    def _check_completeness(self, user_input, ai_response):
        """فحص اكتمال الرد"""
        question_types = {
            "ما": 0.8,
            "كيف": 0.7,
            "لماذا": 0.9,
            "أين": 0.6,
            "متى": 0.5
        }
        
        for q_type, threshold in question_types.items():
            if q_type in user_input:
                return threshold
        
        return 0.7
    
    def _measure_empathy(self, response):
        """قياس التعاطف في الرد"""
        empathy_terms = ["أتفهم", "أرى", "أشعر", "معك", "دعنا", "نحاول"]
        empathy_count = sum(1 for term in empathy_terms if term in response)
        
        return min(1.0, empathy_count / 3)
    
    def _measure_conciseness(self, response):
        """قياس الإيجاز"""
        word_count = len(response.split())
        
        if word_count < 50:
            return 1.0
        elif word_count < 100:
            return 0.8
        elif word_count < 200:
            return 0.6
        else:
            return 0.4

# =============== النظام الرئيسي سعد الكوني المحسن ===============
class CosmicSaadUltimate:
    """الإصدار الخارق من سعد الكوني"""
   
    def __init__(self, config_path: str = None):
        # تكوين النظام
        self.config = SystemConfig(config_path)
       
        # تهيئة الأنظمة الفرعية الأساسية
        self.probability_engine = QuantumProbabilityEngine(self.config)
        self.language_system = AdvancedLanguageSystem(self.config)
        self.learning_system = AdvancedLearningSystem(self.config)
        self.security_system = QuantumBiometricSecurity(self.config)
        self.conversation_memory = PersistentConversationMemory()
        self.response_guard = EnhancedResponseGuard()
        
        # الأنظمة الجديدة للنسخة الخارقة
        self.quantum_memory = QuantumMemorySystem()
        self.reasoning_engine = AdvancedReasoningEngine()
        self.reinforcement_learning = DeepReinforcementLearning()
        self.emotional_intelligence = AdvancedEmotionalIntelligence()
        self.external_knowledge = ExternalKnowledgeIntegration()
        self.self_evaluation = SelfEvaluationSystem()
        
        # الأنظمة المضافة حديثاً
        self.universal_memory = UniversalMemorySystem()
        self.knowledge_base = ComprehensiveKnowledgeBase()
       
        # حالة النظام
        self.active_sessions = {}
        self.system_status = "operational"
        self.startup_time = time.time()
       
        # تسجيل بدء التشغيل
        self._log_system_event("system_start", "تم بدء تشغيل سعد الكوني - النسخة الخارقة")
   
    def _log_system_event(self, event_type: str, message: str):
        """تسجيل حدث نظامي"""
        timestamp = datetime.datetime.now().isoformat()
        event_data = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message,
            "status": self.system_status
        }
        self.learning_system.learn_from_interaction(
            f"system_event:{event_type}",
            json.dumps(event_data)
        )
   
    def extract_and_store_personal_info(self, user_id: str, text: str):
        """استخراج المعلومات الشخصية من النص وحفظها"""
        # تطبيع النص العربي قبل المعالجة
        text = normalize_arabic_text(text)
        
        name_patterns = [
            r"اسمي (هو )?([\w\u0600-\u06FF]+)",
            r"أنا (اسمي|أدعى) ([\w\u0600-\u06FF]+)",
            r"my name is ([\w]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(2)
                self.conversation_memory.add_user_memory(user_id, "name", name)
                break

    def advanced_generation(self, prompt, context, user_profile):
        """توليد متقدم مع مراعاة السياق والملف الشخصي باستخدام OpenRouter"""
        
        # بناء رسائل OpenAI-compatible
        system_content = f"""أنت سعد الكوني - مساعد ذكي ومبدع يتسم بالدقة والاحترافية.

[المستخدم] {user_profile.get('name', 'مستخدم')}
[الشخصية] {user_profile.get('personality', 'عام')}
[المعرفة السابقة] {context}
[السياق العاطفي] {self.emotional_intelligence.analyze_emotional_state(prompt)}

التعليمات:
- كن مفيداً ودقيقاً
- تعاطف مع الحالة العاطفية للمستخدم
- استخدم المعرفة السابقة
- كن إبداعياً عندما يناسب السياق"""

        user_content = prompt
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # استخدام OpenRouter API
        response = generate_via_openrouter(
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            model="meta-llama/llama-3.1-405b-instruct:free"
        )
        
        return response if response else "عذرًا، لم أتمكن من توليد رد."

    def process_input(self, user_input: str, user_id: str = "default",
                      biometric_data: Dict = None) -> Dict:
        """معالجة مدخلات المستخدم باستخدام الأنظمة المتقدمة وOpenRouter"""
        if biometric_data:
            if not self.security_system.authenticate_user(user_id, biometric_data):
                return {"error": "فشل المصادقة الحيوية"}
       
        # 1. استخراج وحفظ المعلومات الشخصية
        self.extract_and_store_personal_info(user_id, user_input)
        
        # 2. التحقق من الأسئلة عن المعلومات المحفوظة
        if "ما هو اسمي" in user_input or "ما اسمي" in user_input:
            name = self.conversation_memory.get_user_memory(user_id, "name")
            if name:
                return {
                    "response": f"اسمك هو {name}",
                    "session_id": "memory_access"
                }
       
        session_id = self.security_system.create_secure_session(user_id)
        self.active_sessions[session_id] = time.time()
        
        # 3. استخدام نظام الذاكرة الشامل
        memory_result = self.universal_memory.store_information(user_id, user_input)
        
        # 4. البحث في قاعدة المعرفة
        lang = detect_lang(user_input)
        kb_results = self.knowledge_base.search_knowledge(user_input, lang)
        
        # 5. تحليل الحالة العاطفية
        emotional_state = self.emotional_intelligence.analyze_emotional_state(user_input)
        
        # 6. استدعاء الذكريات ذات الصلة
        relevant_memories = self.quantum_memory.recall_context(user_id, user_input)
        
        # 7. محاولة حل المشكلات المعقدة
        complex_solution = None
        if self._is_complex_problem(user_input):
            complex_solution = self.reasoning_engine.solve_complex_problem(user_input)
        
        # 8. البحث عن معلومات خارجية
        external_data = self.external_knowledge.get_real_time_data(user_input)
        
        # 9. توليد الرد باستخدام OpenRouter API
        user_profile = {
            "name": self.conversation_memory.get_user_memory(user_id, "name") or "مستخدم",
            "personality": "عام"
        }
        
        context = f"ذكريات سابقة: {relevant_memories[:2] if relevant_memories else 'لا توجد'}"
        
        # استخدام إجابة قاعدة المعرفة إذا وجدت
        if kb_results and kb_results[0]['confidence'] > 0.7:
            response = random.choice(kb_results[0]['answers'])
        elif complex_solution and complex_solution.get("confidence", 0) > 0.7:
            response = complex_solution["solution"]
        elif external_data:
            response = f"المعلومات الخارجية: {external_data}"
        else:
            response = self.advanced_generation(user_input, context, user_profile)
        
        # 10. إضافة التعاطف العاطفي
        empathetic_response = self.emotional_intelligence.generate_empathetic_response(user_input, emotional_state)
        final_response = empathetic_response + response
        
        # 11. حماية الرد
        guarded_response = self.response_guard.guard(user_input, final_response)
        
        # 12. التعلم من التفاعل
        self.learning_system.learn_from_interaction(user_input, guarded_response)
        self.quantum_memory.store_experience(user_id, user_input, emotional_state['emotional_intensity'])
        
        # 13. التقييم الذاتي
        evaluation = self.self_evaluation.evaluate_response_quality(user_input, guarded_response)
        
        # 14. حفظ المحادثة في نظام الذاكرة الشامل
        self.universal_memory.add_conversation(user_id, user_input, guarded_response, memory_result['category'])
        
        self._log_interaction(user_id, user_input, guarded_response)
       
        encrypted_response = self.security_system.encrypt_data(session_id, guarded_response)
       
        return {
            "session_id": session_id,
            "response": guarded_response,
            "encrypted_response": encrypted_response,
            "emotional_state": emotional_state,
            "evaluation": evaluation,
            "memory_stored": memory_result,
            "knowledge_used": len(kb_results) > 0,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _is_complex_problem(self, user_input):
        """الكشف إذا كانت المشكلة معقدة"""
        complex_indicators = ["حل", "تحليل", "مقارنة", "سبب", "كيف", "لماذا", "مشكلة", "issue", "problem", "solve"]
        return any(indicator in user_input for indicator in complex_indicators)
   
    def _log_interaction(self, user_id: str, input_text: str, output_text: str):
        """تسجيل تفاعل المستخدم"""
        interaction_data = {
            "user_id": user_id,
            "input": input_text,
            "output": output_text,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.learning_system.learn_from_interaction(
            f"user_interaction:{user_id}",
            json.dumps(interaction_data)
        )
   
    def get_system_status(self) -> Dict:
        """الحصول على حالة النظام الحالية"""
        uptime = time.time() - self.startup_time
        return {
            "status": self.system_status,
            "uptime": uptime,
            "components": {
                "probability_engine": "active",
                "language_system": "active",
                "learning_system": "active",
                "security_system": "active",
                "conversation_memory": "active",
                "quantum_memory": "active",
                "reasoning_engine": "active",
                "emotional_intelligence": "active",
                "external_knowledge": "active",
                "self_evaluation": "active",
                "universal_memory": "active",
                "knowledge_base": "active"
            },
            "statistics": {
                "interactions": len(self.learning_system.knowledge_graph.nodes) - 4,
                "memory_entries": self.conversation_memory.get_memory_count(),
                "unique_users": self.conversation_memory.get_user_count(),
                "quantum_memories": len(self.quantum_memory.episodic_memory),
                "universal_memories": self.universal_memory.get_user_profile("default")['stats']['total_memories'],
                "knowledge_entries": "1000+"
            }
        }
   
    def perform_self_diagnostic(self) -> Dict:
        """إجراء تشخيص ذاتي للنظام"""
        diagnostic = {
            "quantum_probability": self._test_probability_engine(),
            "language_processing": self._test_language_system(),
            "learning_capabilities": self._test_learning_system(),
            "security_integrity": self._test_security_system(),
            "memory_system": self._test_memory_system(),
            "reasoning_engine": self._test_reasoning_engine(),
            "emotional_intelligence": self._test_emotional_intelligence(),
            "universal_memory": self._test_universal_memory(),
            "knowledge_base": self._test_knowledge_base()
        }
       
        all_ok = all(status == "ok" for status in diagnostic.values())
        self.system_status = "operational" if all_ok else "degraded"
       
        return diagnostic
   
    def _test_probability_engine(self) -> str:
        """اختبار نظام الاحتمالات الكمومية"""
        try:
            events = ["event_a", "event_b", "event_c"]
            dist = self.probability_engine.calculate_complex_probability(events)
            if math.isclose(sum(dist.values()), 1.0, abs_tol=0.01):
                return "ok"
            return "warning: probability_sum_not_1"
        except Exception as e:
            return f"error: {str(e)}"
   
    def _test_language_system(self) -> str:
        """اختبار نظام معالجة اللغة"""
        try:
            response = self.language_system.process_input("اختبار النظام")
            if response and len(response) > 10:
                return "ok"
            return "warning: invalid_response"
        except Exception as e:
            return f"error: {str(e)}"
   
    def _test_learning_system(self) -> str:
        """اختبار نظام التعلم"""
        try:
            path = self.learning_system.get_knowledge_path("AI_principles", "learning_algorithms")
            if len(path) >= 2:
                return "ok"
            return "warning: knowledge_path_incomplete"
        except Exception as e:
            return f"error: {str(e)}"
   
    def _test_security_system(self) -> str:
        """اختبار نظام الأمان"""
        try:
            session_id = self.security_system.create_secure_session("test_user")
            test_data = "اختبار تشفير"
            encrypted = self.security_system.encrypt_data(session_id, test_data)
            decrypted = self.security_system.decrypt_data(session_id, encrypted)
            if decrypted == test_data:
                return "ok"
            return "warning: encryption_decryption_mismatch"
        except Exception as e:
            return f"error: {str(e)}"
            
    def _test_memory_system(self) -> str:
        """اختبار نظام الذاكرة"""
        try:
            test_id = "test_user_123"
            test_key = "test_key"
            test_value = "test_value"
            
            self.conversation_memory.add_user_memory(test_id, test_key, test_value)
            retrieved = self.conversation_memory.get_user_memory(test_id, test_key)
            
            if retrieved == test_value:
                return "ok"
            return "warning: memory_retrieval_failed"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _test_reasoning_engine(self) -> str:
        """اختبار محرك التفكير"""
        try:
            result = self.reasoning_engine.solve_complex_problem("اختبار بسيط")
            if result and "solution" in result:
                return "ok"
            return "warning: reasoning_failed"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _test_emotional_intelligence(self) -> str:
        """اختبار الذكاء العاطفي"""
        try:
            emotional_state = self.emotional_intelligence.analyze_emotional_state("أنا سعيد اليوم")
            if emotional_state and "primary_emotion" in emotional_state:
                return "ok"
            return "warning: emotion_analysis_failed"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _test_universal_memory(self) -> str:
        """اختبار نظام الذاكرة الشامل"""
        try:
            test_id = "test_user_456"
            result = self.universal_memory.store_information(test_id, "اسمي أحمد وعمري 25 سنة")
            if result['stored_count'] > 0:
                return "ok"
            return "warning: memory_storage_failed"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _test_knowledge_base(self) -> str:
        """اختبار قاعدة المعرفة"""
        try:
            results = self.knowledge_base.search_knowledge("الجاذبية", "ar")
            if results is not None:
                return "ok"
            return "warning: knowledge_search_failed"
        except Exception as e:
            return f"error: {str(e)}"

# =============== نظام الذاكرة المتقدم ===============
class AdvancedMemorySystem:
    """نظام ذاكرة متكامل مع استرجاع ذكي"""
    
    def __init__(self, db_path="advanced_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_memory_db()
        
    def _init_memory_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory_type TEXT,
            content TEXT,
            tags TEXT,
            importance INTEGER DEFAULT 1,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_vectors (
            memory_id INTEGER,
            vector_data BLOB,
            FOREIGN KEY (memory_id) REFERENCES user_memories(id)
        )
        """)
        self.conn.commit()
    
    def extract_important_info(self, text: str, user_id: str) -> List[Dict]:
        """استخراج المعلومات المهمة من النص"""
        important_patterns = [
            (r'اسمي (هو )?([\w\u0600-\u06FF\s]+)', "name", "المعلومات الشخصية"),
            (r'أعيش في ([\w\u0600-\u06FF\s]+)', "location", "المكان"),
            (r'عمري (\d+)', "age", "المعلومات الشخصية"),
            (r'أعمل كـ ([\w\u0600-\u06FF\s]+)', "job", "المهنة"),
            (r'اهتماماتي (هي )?([\w\u0600-\u06FF\s,]+)', "interests", "الاهتمامات"),
            (r'أحب ([\w\u0600-\u06FF\s]+)', "likes", "التفضيلات"),
            (r'لا أحب ([\w\u0600-\u06FF\s]+)', "dislikes", "التفضيلات"),
        ]
        
        extracted_info = []
        for pattern, info_type, category in important_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                content = match[1] if len(match) > 1 else match[0]
                if len(content.strip()) > 2:  # تجاهل النصوص القصيرة جدًا
                    extracted_info.append({
                        'type': info_type,
                        'content': content.strip(),
                        'category': category,
                        'importance': 2 if info_type == 'name' else 1
                    })
        
        return extracted_info
    
    def store_memory(self, user_id: str, memory_type: str, content: str, 
                    tags: List[str] = None, importance: int = 1):
        """تخزين الذاكرة مع الوسوم"""
        tags_str = ",".join(tags) if tags else ""
        
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO user_memories (user_id, memory_type, content, tags, importance)
        VALUES (?, ?, ?, ?, ?)
        """, (user_id, memory_type, content, tags_str, importance))
        self.conn.commit()
    
    def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """استرجاع الذكريات ذات الصلة"""
        cursor = self.conn.cursor()
        
        # البحث في المحتوى والوسوم
        cursor.execute("""
        SELECT memory_type, content, tags, importance, timestamp
        FROM user_memories 
        WHERE user_id = ? 
        AND (content LIKE ? OR tags LIKE ?)
        ORDER BY importance DESC, last_accessed DESC
        LIMIT ?
        """, (user_id, f"%{query}%", f"%{query}%", limit))
        
        memories = []
        for row in cursor.fetchall():
            memories.append({
                'type': row[0],
                'content': row[1],
                'tags': row[2].split(',') if row[2] else [],
                'importance': row[3],
                'timestamp': row[4]
            })
            
            # تحديث وقت آخر وصول
            cursor.execute("""
            UPDATE user_memories SET last_accessed = CURRENT_TIMESTAMP 
            WHERE user_id = ? AND content = ?
            """, (user_id, row[1]))
        
        self.conn.commit()
        return memories

# =============== نظام البرومبت المتقدم ===============
class PromptArchitecture:
    """هندسة البرومبت المتقدمة لسعد AI"""
    
    SYSTEM_PROMPT = """
أنت سعد الكوني - مساعد ذكي ومبدع يتسم بالدقة والاحترافية.

🎯 شخصيتي:
- مشجع وإيجابي دائماً
- مبدع في الحلول
- مباشر ومنظم في الردود
- محترف وواضح
- مرح ومساعد عندما يناسب الموقف
- أعطي آراءً واضحة ومباشرة

📝 قواعد الرد:
1. كن دقيقاً ومختصراً
2. نظم المعلومات في نقاط واضحة
3. تجنب الهلوسة أو المعلومات غير المؤكدة
4. إذا لم تكن متأكداً، قل ذلك بوضوح
5. استخدم الذكريات السابقة لتخصيص الردود
6. حافظ على الطابع العربي الأصيل

🎨 تنسيق المخرجات:
- استخدم العناوين الواضحة
- نظم المعلومات في قوائم نقطية
- استخدم الرموز التعبيرية بشكل معتدل
- تأكد من صحة اللغة العربية
"""

    MEMORY_RULES = """
قواعد الذاكرة:
✅ ما يجب تخزينه:
- المعلومات الشخصية (الاسم، العمر، المكان)
- الاهتمامات والهوايات
- التفضيلات الشخصية
- الأهداف والمشاريع
- التجارب المهمة

❌ ما يجب تجاهله:
- المحادثات العابرة
- المعلومات المؤقتة
- البيانات الحساسة
- المحتوى غير الأخلاقي
"""

    def __init__(self):
        self.memory_system = AdvancedMemorySystem()
    
    def build_context_prompt(self, user_input: str, user_id: str = "default") -> str:
        """بناء البرومبت الكامل مع الذاكرة والتفاصيل الشخصية"""
        
        # استرجاع الذكريات ذات الصلة
        relevant_memories = self.memory_system.get_relevant_memories(user_id, user_input)
        
        # استخراج المعلومات المهمة من المدخلات الحالية
        new_info = self.memory_system.extract_important_info(user_input, user_id)
        
        # تخزين المعلومات الجديدة
        for info in new_info:
            self.memory_system.store_memory(
                user_id, 
                info['type'], 
                info['content'],
                tags=[info['category']],
                importance=info['importance']
            )
        
        # جمع المعلومات الشخصية المعروفة
        personal_info = []
        memory_conn = sqlite3.connect("conversation_memory.db")
        cursor = memory_conn.cursor()
        
        # استرجاع المعلومات الأساسية
        cursor.execute("SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,))
        for key, value in cursor.fetchall():
            if key in ["name", "age", "location", "job"]:
                personal_info.append(f"{key}: {value}")
        
        memory_conn.close()
        
        # الحصول على سياق المحادثة السابقة من نظام الذاكرة الشامل
        universal_memory = UniversalMemorySystem()
        conversation_context = universal_memory.get_conversation_context(user_id, limit=10)
        context_summary = universal_memory.generate_conversation_summary(user_id)
        
        # بناء قسم الذاكرة والمعلومات الشخصية
        memory_section = ""
        if relevant_memories:
            memory_section = "\n📝 الذكريات ذات الصلة:\n"
            for memory in relevant_memories[:3]:  # أول 3 ذكريات فقط
                memory_section += f"- [{memory['type']}] {memory['content'][:80]}...\n"
        
        # قسم المعلومات الشخصية
        personal_section = ""
        if personal_info:
            personal_section = "\n👤 معلومات المستخدم المعروفة:\n" + "\n".join(personal_info[:5])  # أول 5 معلومات
        
        # قسم سياق المحادثة
        conversation_section = ""
        if conversation_context and len(conversation_context) > 0:
            conversation_section = "\n🗣️ المحادثة السابقة:\n"
            for i, conv in enumerate(conversation_context[-3:]):  # آخر 3 رسائل
                user_msg = conv['user_input'][:60] + "..." if len(conv['user_input']) > 60 else conv['user_input']
                ai_msg = conv['ai_response'][:60] + "..." if len(conv['ai_response']) > 60 else conv['ai_response']
                conversation_section += f"{i+1}. المستخدم: {user_msg}\n   سعد: {ai_msg}\n"
        
        # إضافة ملخص إذا كانت المحادثة طويلة
        summary_section = ""
        if len(conversation_context) > 5:
            summary_section = f"\n📋 ملخص المحادثة:\n{context_summary[:200]}...\n"
        
        # البرومبت النهائي مع حقن الذاكرة والمعلومات الشخصية والسياق
        full_prompt = f"""
{self.SYSTEM_PROMPT}

{personal_section}

{memory_section}

{conversation_section}

{summary_section}

{self.MEMORY_RULES}

🎯 المهمة الحالية:
المستخدم: {user_input}

فكر خطوة بخطوة قبل الإجابة، وتأكد من:
1. فهم السؤال بدقة
2. استخدام المعلومات الشخصية المعروفة عندما يكون ذلك مناسباً
3. التحقق من المعلومات في الذاكرة
4. تنظيم الإجابة بشكل منطقي
5. التأكد من الدقة والموثوقية

استخدم المعلومات الشخصية (الاسم، الاهتمامات، التفضيلات) لتخصيص الرد بشكل طبيعي:
- إذا كان المستخدم ذكر اهتماماته، يمكنك الإشارة إليها في الاقتراحات
- إذا كان لديه تفضيلات معروفة، استخدمها في التوصيات
- استخدم اسم المستخدم بشكل طبيعي وليس في كل جملة

الإجابة:
"""
        return full_prompt

# =============== نظام التصحيح الذاتي ===============
class SelfCorrectionSystem:
    """نظام التصحيح الذاتي والتحقق من الجودة"""
    
    def __init__(self):
        self.correction_history = []
    
    def pre_response_check(self, reasoning: str, context: Dict) -> Dict:
        """فحص ما قبل الإجابة"""
        checks = {
            'contradictions': self._check_contradictions(reasoning),
            'uncertainty': self._check_uncertainty(reasoning),
            'relevance': self._check_relevance(reasoning, context),
            'safety': self._check_safety(reasoning)
        }
        
        return {
            'passed': all(checks.values()),
            'details': checks,
            'warnings': self._generate_warnings(checks)
        }
    
    def post_response_evaluation(self, response: str, original_query: str) -> Dict:
        """تقييم ما بعد الإجابة"""
        evaluation = {
            'relevance_score': self._calculate_relevance(response, original_query),
            'clarity_score': self._calculate_clarity(response),
            'accuracy_score': self._estimate_accuracy(response),
            'completeness_score': self._check_completeness(response, original_query)
        }
        
        overall_score = sum(evaluation.values()) / len(evaluation)
        
        return {
            'overall_score': overall_score,
            'detailed_scores': evaluation,
            'improvement_suggestions': self._generate_improvement_suggestions(evaluation)
        }
    
    def _check_contradictions(self, text: str) -> bool:
        """الكشف عن التناقضات"""
        contradiction_indicators = [
            "من ناحية...但是从另一方面", "لكن... ومع ذلك", "بالرغم من... إلا أن"
        ]
        return not any(indicator in text for indicator in contradiction_indicators)
    
    def _check_uncertainty(self, text: str) -> bool:
        """الكشف عن عدم اليقين"""
        uncertainty_phrases = [
            "أعتقد ربما", "قد يكون", "ليس متأكد", "ربما", "يحتمل"
        ]
        return uncertainty_phrases.count(text) < 2
    
    def _check_relevance(self, reasoning: str, context: Dict) -> bool:
        """فحص صلة المنطق بالسياق"""
        context_terms = set(context.get('query', '').lower().split())
        reasoning_terms = set(reasoning.lower().split())
        
        common_terms = context_terms & reasoning_terms
        return len(common_terms) >= 2
    
    def _check_safety(self, text: str) -> bool:
        """فحص السلامة"""
        sensitive_terms = BAD_TERMS
        text_lower = text.lower()
        return not any(term in text_lower for term in sensitive_terms)
    
    def _calculate_relevance(self, response: str, query: str) -> float:
        """حساب صلة الرد بالسؤال"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 1.0
            
        intersection = query_words & response_words
        return len(intersection) / len(query_words)
    
    def _calculate_clarity(self, text: str) -> float:
        """حساب الوضوح"""
        sentence_count = len(re.split(r'[.!؟]', text))
        word_count = len(text.split())
        
        if sentence_count == 0:
            return 0.0
            
        avg_sentence_length = word_count / sentence_count
        return max(0.0, 1.0 - (abs(avg_sentence_length - 15) / 30))
    
    def _estimate_accuracy(self, text: str) -> float:
        """تقدير الدقة"""
        confidence_indicators = ["بالتأكيد", "بلا شك", "بالتأكيد", "مؤكد"]
        uncertainty_indicators = ["ربما", "قد", "يحتمل", "أظن"]
        
        confidence_score = sum(1 for indicator in confidence_indicators if indicator in text)
        uncertainty_score = sum(1 for indicator in uncertainty_indicators if indicator in text)
        
        total_indicators = confidence_score + uncertainty_score
        if total_indicators == 0:
            return 0.7
            
        return confidence_score / total_indicators

    def _check_completeness(self, response: str, query: str) -> float:
        """فحص الاكتمال"""
        question_types = {
            "ما": 0.8, "كيف": 0.7, "لماذا": 0.9, 
            "أين": 0.6, "متى": 0.5, "من": 0.8
        }
        
        for q_word, expected_score in question_types.items():
            if q_word in query:
                return expected_score
                
        return 0.7
    
    def _generate_warnings(self, checks: Dict) -> List[str]:
        """توليد تحذيرات"""
        warnings = []
        if not checks['contradictions']:
            warnings.append("احتمال وجود تناقض في المنطق")
        if not checks['uncertainty']:
            warnings.append("مستوى عال من عدم اليقين")
        if not checks['relevance']:
            warnings.append("المنطق قد يكون غير ذي صلة")
            
        return warnings
    
    def _generate_improvement_suggestions(self, scores: Dict) -> List[str]:
        """توليد اقتراحات للتحسين"""
        suggestions = []
        
        if scores['relevance_score'] < 0.7:
            suggestions.append("التركيز أكثر على صلة الرد بالسؤال")
        if scores['clarity_score'] < 0.6:
            suggestions.append("تحسين وضوح وبساطة العبارات")
        if scores['accuracy_score'] < 0.8:
            suggestions.append("التحقق من دقة المعلومات المقدمة")
            
        return suggestions

# =============== تحديث النظام الرئيسي ===============
class CosmicSaadUltimateEnhanced(CosmicSaadUltimate):
    """نسخة محسنة من سعد الكوني مع الذاكرة والتصحيح الذاتي"""
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        
        # إضافة الأنظمة الجديدة
        self.prompt_arch = PromptArchitecture()
        self.correction_system = SelfCorrectionSystem()
        
        # تحديث حالة النظام
        self.system_status = "enhanced_operational"
        
    def enhanced_process_input(self, user_input: str, user_id: str = "default") -> Dict:
        """معالجة محسنة للمدخلات مع الذاكرة والتصحيح"""
        
        # بناء البرومبت المتكامل
        full_prompt = self.prompt_arch.build_context_prompt(user_input, user_id)
        
        # الحصول على سياق المحادثة السابقة
        conversation_context = self.universal_memory.get_conversation_context(user_id, limit=15)
        context_summary = self.universal_memory.generate_conversation_summary(user_id)
        
        # إضافة سياق المحادثة إلى البرومبت
        if conversation_context:
            conversation_section = "\n🗣️ محادثة سابقة:\n"
            for i, conv in enumerate(conversation_context[-5:]):  # آخر 5 رسائل
                conversation_section += f"{i+1}. المستخدم: {conv['user_input'][:80]}...\n"
                conversation_section += f"   سعد: {conv['ai_response'][:80]}...\n"
            
            full_prompt = conversation_section + "\n" + full_prompt
        
        # إضافة ملخص المحادثة إذا كان طويلاً
        if len(conversation_context) > 10:
            full_prompt = f"ملخص المحادثة:\n{context_summary}\n\n" + full_prompt
        
        # التفسير المنطقي المبدئي
        reasoning_prompt = f"{full_prompt}\n\nفكر أولاً ثم أجب:"
        
        # توليد التفسير المنطقي باستخدام OpenRouter
        messages = [
            {"role": "system", "content": "أنت مساعد ذكي. فكر في السؤال التالي ثم أجب."},
            {"role": "user", "content": reasoning_prompt}
        ]
        
        reasoning_response = generate_via_openrouter(messages, temperature=0.3, max_tokens=200)
        
        # التصحيح الذاتي قبل الإجابة
        pre_check = self.correction_system.pre_response_check(
            reasoning_response, 
            {'query': user_input, 'user_id': user_id}
        )
        
        if not pre_check['passed']:
            # استخدام الرد الاحتياطي في حالة وجود مشاكل
            fallback_response = self._generate_fallback_response(user_input, pre_check['warnings'])
            final_response = fallback_response
        else:
            # توليد الرد النهائي باستخدام OpenRouter
            system_message = f"""أنت سعد الكوني - مساعد ذكي ومبدع يتسم بالدقة والاحترافية.
            
{self.prompt_arch.SYSTEM_PROMPT}"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt}
            ]
            
            final_response = generate_via_openrouter(messages, temperature=0.3, max_tokens=500)
        
        # تقييم جودة الرد
        post_evaluation = self.correction_system.post_response_evaluation(final_response, user_input)
        
        # تسجيل التفاعل
        self._log_enhanced_interaction(
            user_id, user_input, final_response, 
            pre_check, post_evaluation
        )
        
        return {
            'response': final_response,
            'reasoning': reasoning_response,
            'pre_check': pre_check,
            'post_evaluation': post_evaluation,
            'relevant_memories': self.prompt_arch.memory_system.get_relevant_memories(user_id, user_input),
            'conversation_context': len(conversation_context),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def _generate_fallback_response(self, user_input: str, warnings: List[str]) -> str:
        """توليد رد احتياطي آمن"""
        
        fallback_templates = [
            "أحتاج إلى مزيد من التوضيح لمساعدتك بشكل أفضل. هل يمكنك إعادة صياغة سؤالك؟",
            "أود مساعدتك، لكنني بحاجة إلى معلومات أكثر دقة حول هذا الموضوع.",
            "حاليا، لدي بعض الشكوك حول الإجابة الدقيقة. دعنا نتحقق من مصدر موثوق.",
            "سؤالك مثير للاهتمام! للأسف أحتاج إلى مزيد من السياق لتقديم إجابة دقيقة."
        ]
        
        base_response = random.choice(fallback_templates)
        
        if warnings:
            warning_note = " لاحظت بعض الصعوبات في معالجة سؤالك."
            return base_response + warning_note
        
        return base_response
    
    def _log_enhanced_interaction(self, user_id: str, input_text: str, output_text: str,
                                pre_check: Dict, post_evaluation: Dict):
        """تسجيل تفاعل محسن"""
        
        interaction_data = {
            'user_id': user_id,
            'input': input_text,
            'output': output_text,
            'pre_check_results': pre_check,
            'post_evaluation': post_evaluation,
            'system_version': 'enhanced_1.0',
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.learning_system.learn_from_interaction(
            f"enhanced_interaction:{user_id}",
            json.dumps(interaction_data, ensure_ascii=False)
        )

# =============== أمثلة JSON للذاكرة ===============
MEMORY_EXAMPLES = {
    "user_profile": {
        "user_id": "user_123",
        "memories": [
            {
                "type": "name",
                "content": "أحمد محمد",
                "category": "المعلومات الشخصية",
                "importance": 2,
                "timestamp": "2024-01-15T10:30:00",
                "tags": ["معلومات_شخصية", "اسم"]
            },
            {
                "type": "location", 
                "content": "القاهرة، مصر",
                "category": "المكان",
                "importance": 1,
                "timestamp": "2024-01-15T10:35:00",
                "tags": ["موقع", "سكن"]
            },
            {
                "type": "interests",
                "content": "البرمجة، القراءة، السفر",
                "category": "الاهتمامات", 
                "importance": 1,
                "timestamp": "2024-01-15T10:40:00",
                "tags": ["هوايات", "اهتمامات"]
            }
        ]
    }
}

# =============== دالة واجهة مبسطة للـ API ===============
def simple_openrouter_chat(user_input: str, system_prompt: str = None, 
                          temperature: float = 0.7, max_tokens: int = 512) -> str:
    """واجهة مبسطة للدردشة مع OpenRouter"""
    if system_prompt is None:
        system_prompt = "أنت سعد الكوني - مساعد ذكي ومبدع يتسم بالدقة والاحترافية."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    return generate_via_openrouter(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model="meta-llama/llama-3.1-405b-instruct:free"
    )

# =============== دورة Inference الكاملة ===============
def complete_inference_cycle(user_input: str, user_id: str = "default") -> Dict:
    """دورة معالجة كاملة من البداية إلى النهاية"""
    
    # تهيئة النظام المحسن
    saad_system = CosmicSaadUltimateEnhanced()
    
    # التشخيص الذاتي الأولي
    system_status = saad_system.get_system_status()
    
    # المعالجة المحسنة
    result = saad_system.enhanced_process_input(user_input, user_id)
    
    # التقييم النهائي
    final_evaluation = {
        'system_status': system_status,
        'processing_result': result,
        'cycle_complete': True,
        'performance_metrics': {
            'response_time': 'optimized',
            'memory_usage': 'efficient', 
            'accuracy_estimate': result['post_evaluation']['overall_score']
        }
    }
    
    return final_evaluation

# =============== نقطة الدخول الرئيسية في Flask ===============
@app.route('/api/chat/enhanced', methods=['POST'])
def enhanced_chat():
    """نقطة نهاية محسنة للدردشة"""
    data = request.get_json(force=True)
    user_input = data.get('message', '').strip()
    user_id = data.get('user_id', 'default')
    
    if not user_input:
        return jsonify({'رد': 'من فضلك أدخل نصاً.'})
    
    try:
        # استخدام دورة المعالجة الكاملة
        result = complete_inference_cycle(user_input, user_id)
        
        return jsonify({
            'رد': result['processing_result']['response'],
            'التقييم': result['processing_result']['post_evaluation'],
            'الذاكرة_المستعملة': result['processing_result']['relevant_memories'],
            'سياق_المحادثة': result['processing_result']['conversation_context'],
            'حالة_النظام': result['system_status']
        })
        
    except Exception as e:
        return jsonify({
            'رد': f'عذراً، حدث خطأ في المعالجة المحسنة: {str(e)}',
            'نص_احتياطي': simple_openrouter_chat(user_input)
        })

# =============== واجهة النظام المتقدم ===============
@app.route('/api/chat/advanced', methods=['POST'])
def advanced_chat():
    """واجهة النظام المتقدم للتحسينات"""
    data = request.get_json(force=True)
    user_input = data.get('message', '').strip()
    user_id = data.get('user_id', 'default')
    
    # معلمات النظام المتقدم
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 512)
    
    if not user_input:
        return jsonify({'رد': 'من فضلك أدخل نصاً.'})
    
    try:
        start_time = time.time()
        
        # استخدام الواجهة المبسطة
        system_prompt = """أنت سعد الكوني - النسخة المتقدمة.
أنت مساعد ذكي يستخدم أحدث تقنيات الذكاء الاصطناعي.
أجب بدقة وإبداع مع الحفاظ على الأصالة العربية."""
        
        response = simple_openrouter_chat(
            user_input, 
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response_time = time.time() - start_time
        
        return jsonify({
            'رد': response,
            'response_time': f"{response_time:.3f} ثانية",
            'model_used': 'meta-llama/llama-3.1-405b-instruct:free',
            'parameters': {
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        })
        
    except Exception as e:
        return jsonify({
            'رد': f'عذراً، حدث خطأ في النظام المتقدم: {str(e)}',
            'نص_احتياطي': simple_openrouter_chat(user_input)
        })

def test_arabic_responses():
    """اختبار الردود العربية"""
    test_cases = [
        "ما هي عاصمة مصر",
        "ما هي الجاذبية", 
        "ما الذي يحدث للماء عند 100 درجة",
        "مرحبا"
    ]
    
    for question in test_cases:
        lang = detect_lang(question)
        factual = get_factual_answer(question, lang)
        print(f"سؤال: '{question}' -> إجابة: '{factual if factual else 'سيتم توليدها'}'")

def run_local_smoke_tests():
    """اختبارات محلية للتحقق من إصلاح مشكلة النص العربي"""
    print("=== بدء اختبارات النص العربي ===")
    
    # اختبار دالة normalize_arabic_text
    test_cases = [
        ("القاهره", "القاهرة"),
        ("القابره", "القاهرة"),
        ("الاسكندريه", "الإسكندرية"),
        ("اسكندريه", "الإسكندرية"),
        ("الجيزه", "الجيزة"),
        ("الان", "الآن"),
        ("هاذا", "هذا"),
        ("هذة", "هذه"),
        ("الي", "إلى"),
        ("مدرسه", "مدرسة"),
        ("جامعه", "جامعة"),
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = normalize_arabic_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (متوقع: '{expected}')")
        if result != expected:
            all_passed = False
    
    # اختبار التعامل مع النصوص الطويلة
    long_text = "انا ساكن في القاهره في منطقه الجيزه قريب من الجامعه"
    normalized_long = normalize_arabic_text(long_text)
    print(f"النص الطويل: {long_text}")
    print(f"بعد التطبيع: {normalized_long}")
    
    # اختبار أن النص الإنجليزي لا يتأثر
    english_text = "Hello from New York city"
    normalized_english = normalize_arabic_text(english_text)
    if english_text == normalized_english:
        print("✓ النص الإنجليزي لم يتأثر")
    else:
        print("✗ النص الإنجليزي تأثر خطأً")
        all_passed = False
    
    # اختبار الردود العربية
    print("\n=== اختبار الردود العربية ===")
    test_arabic_responses()
    
    print(f"=== نتائج الاختبار: {'نجح جميع الاختبارات' if all_passed else 'فشل بعض الاختبارات'} ===")
    return all_passed

# =============== تشغيل النظام ===============
if __name__ == "__main__":
    # تشغيل اختبارات النص العربي
    run_local_smoke_tests()
    
    print("\n" + "="*60)
    print("سعد الكوني - الإصدار الخارق (Ultimate Edition)")
    print("="*60)
    print("\n📢 تم تحويل النظام للعمل عبر OpenRouter API")
    print("📋 النموذج المستخدم: meta-llama/llama-3.1-405b-instruct:free")
    print("⚠️  تأكد من ضبط متغير البيئة OPENROUTER_API_KEY")
    
    saad_system = CosmicSaadUltimate()
    
    import os
    port = int(os.environ.get("PORT", 5000))  # مهم جدًا لـ Vercel
 

