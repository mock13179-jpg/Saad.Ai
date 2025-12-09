"""
Ultimate Security System for AI Chat Application - Enhanced Version
نظام أمان نهائي مع تفاعلات كيميائية ودفاعات متتالية وقواعد بيانات وهمية ديناميكية
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status, Response, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
import asyncio
from pydantic import BaseModel, EmailStr, validator, Field, conlist
import redis.asyncio as redis
import redis.cluster as redis_cluster
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac as cryptohmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import uuid
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import base64
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import random
import string
from collections import defaultdict
import re
import pyotp
from typing_extensions import Literal
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
from fastapi.responses import JSONResponse
import os
import pickle
import zlib
import msgpack
from datetime import timezone
import ipaddress
import socket
import dns.resolver
from ssl import SSLContext, PROTOCOL_TLSv1_2, CERT_REQUIRED
import certifi
import aiohttp
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
from elasticsearch import AsyncElasticsearch
import brotli
import lz4.frame
import numpy as np
from scipy import stats
import pandas as pd
from datasketch import MinHash, MinHashLSH
import ssdeep
import tlsh

# ==================== CONFIGURATION ====================
class SecurityConfig:
    # Encryption
    ENCRYPTION_KEY = Fernet.generate_key()
    JWT_SECRET = secrets.token_urlsafe(256)
    DATA_MUTATION_KEY = secrets.token_urlsafe(192)
    CHAT_ENCRYPTION_KEY = secrets.token_bytes(32)
    
    # Reaction-based Mutation
    MUTATION_TRIGGERS = [
        "sql_injection",
        "xss_attempt",
        "brute_force",
        "unauthorized_access",
        "data_exfiltration",
        "token_theft"
    ]
    
    MUTATION_LEVELS = {
        "low": {"rounds": 3, "fake_users": 5},
        "medium": {"rounds": 5, "fake_users": 10},
        "high": {"rounds": 7, "fake_users": 20},
        "critical": {"rounds": 10, "fake_users": 30}
    }
    
    # Cascade Reaction
    CASCADE_REACTIONS_PER_ATTACK = 20
    CASCADE_DELAY_BETWEEN_REACTIONS = 0.1  # seconds
    FAKE_USER_PATTERNS = [
        "{username}_fake_{index}",
        "{username}_backup_{random}",
        "{username}.migrated",
        "{username}.deprecated",
        "{username}_system_mirror",
        "backup_{username}_{random}",
        "{username}_archived_{timestamp}",
        "{username}_temp_{random}"
    ]
    
    # Dynamic Honey Database
    HONEY_DB_PREFIX = "honey_"
    HONEY_DB_TTL = 86400  # 24 hours
    MAX_HONEY_DATABASES = 1000
    
    # Rate Limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_WINDOW_SECONDS = 300
    API_RATE_LIMIT = "500/minute"
    CHAT_RATE_LIMIT = "1000/minute"
    
    # WAF Rules
    SQL_KEYWORDS = ["SELECT", "INSERT", "DELETE", "UPDATE", "DROP", "UNION", "--", ";", "'", "OR 1=1", "EXEC", "EXECUTE", "INTO OUTFILE"]
    XSS_PATTERNS = ["<script>", "javascript:", "onload=", "onerror=", "onclick=", "eval(", "document.cookie", "alert("]
    
    # Dangerous Content Detection
    DANGEROUS_PATTERNS = [
        r'\b\d{16}\b',
        r'\b\d{3}-\d{2}-\d{4}\b',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b\d{10,15}\b',
        r'\b(قنبل|تفجير|إرهاب|قتل|انتحار|انتحر)\b',
        r'\b(bomb|explosion|terror|kill|suicide|murder)\b',
        r'\b(عنصري|تحرش|تهديد|ابتزاز)\b',
        r'\b(racist|harassment|threat|blackmail)\b',
    ]
    
    # Session
    SESSION_TIMEOUT = 3600
    MFA_REQUIRED_FOR_ALL = True
    
    # Chemical Reaction Defense
    REACTION_TRIGGER_THRESHOLD = 2
    MUTATION_ROUNDS = 7
    DATA_DECOY_RATIO = 0.4
    
    # Chat Security
    MESSAGE_ENCRYPTION = True
    END_TO_END_ENCRYPTION = True
    CHAT_LOG_RETENTION = 30
    REAL_TIME_THREAT_DETECTION = True
    
    # Performance
    REDIS_CLUSTER_NODES = [
        {"host": "redis-node-1", "port": 7000},
        {"host": "redis-node-2", "port": 7001},
        {"host": "redis-node-3", "port": 7002},
        {"host": "redis-node-4", "port": 7003},
    ]
    
    # Databases
    POSTGRES_URL = "postgresql://user:pass@db-cluster:5432/secure_chat"
    MONGO_URL = "mongodb://mongo-cluster:27017"
    ELASTICSEARCH_URLS = ["http://es-node-1:9200", "http://es-node-2:9200"]
    
    # Kafka for real-time processing
    KAFKA_BOOTSTRAP_SERVERS = ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"]
    
    # Dynamic Defense
    FINGERPRINT_HEADERS = [
        "user-agent",
        "accept-language",
        "accept-encoding",
        "connection",
        "upgrade-insecure-requests",
        "sec-fetch-site",
        "sec-fetch-mode",
        "sec-fetch-user",
        "sec-fetch-dest",
        "viewport-width",
        "device-memory",
        "hardware-concurrency"
    ]

# ==================== NEW FEATURE: REACTION-BASED MUTATION ====================
class ReactionBasedMutation:
    """
    تفاعل كيميائي يغيّر شكل البيانات عند اكتشاف محاولة غير شرعية
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.mutation_history = {}
        
    async def trigger_mutation(self, attack_data: Dict[str, Any], 
                               original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تشغيل طفرة تفاعلية على البيانات
        """
        attack_type = attack_data.get("attack_type", "unknown")
        severity = attack_data.get("severity", "medium")
        
        # تحديد مستوى الطفرة
        mutation_config = SecurityConfig.MUTATION_LEVELS.get(
            severity, 
            SecurityConfig.MUTATION_LEVELS["medium"]
        )
        
        mutated_data = original_data.copy()
        
        # جولات متعددة من الطفرات
        for round_num in range(mutation_config["rounds"]):
            await logger.info(f"Mutation round {round_num + 1} for attack {attack_type}")
            
            # تطبيق أنواع مختلفة من الطفرات
            mutated_data = await self._apply_username_mutation(mutated_data)
            mutated_data = await self._apply_salt_mutation(mutated_data)
            mutated_data = await self._apply_hash_mutation(mutated_data)
            mutated_data = await self._apply_uuid_mutation(mutated_data)
            mutated_data = await self._apply_token_mutation(mutated_data)
            mutated_data = await self._apply_structure_mutation(mutated_data)
            
            # إنشاء نسخ وهمية
            fake_copies = await self._generate_fake_copies(
                mutated_data, 
                mutation_config["fake_users"]
            )
            
            # تخزين النسخ الوهمية
            await self._store_fake_copies(fake_copies, attack_data)
            
            # إضافة تأخير لمحاكاة التفاعل الكيميائي
            await asyncio.sleep(0.05)
        
        # تسجيل الطفرة
        mutation_id = str(uuid.uuid4())
        await self._record_mutation(mutation_id, attack_data, original_data, mutated_data)
        
        return {
            "mutation_id": mutation_id,
            "original_data_hash": hashlib.sha256(json.dumps(original_data).encode()).hexdigest(),
            "mutated_data_hash": hashlib.sha256(json.dumps(mutated_data).encode()).hexdigest(),
            "rounds": mutation_config["rounds"],
            "fake_copies_generated": mutation_config["fake_users"],
            "attack_type": attack_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _apply_username_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في اسم المستخدم"""
        if "username" in data:
            username = data["username"]
            
            # طرق مختلفة لتغيير الشكل
            mutations = [
                lambda x: x + "_" + secrets.token_hex(4),
                lambda x: x.translate(str.maketrans("aeiou", "43102")),
                lambda x: "".join(random.choices(string.ascii_letters + string.digits, k=8)) + "_" + x,
                lambda x: x[::-1] + "_rev",
                lambda x: hashlib.md5(x.encode()).hexdigest()[:8],
                lambda x: base64.b64encode(x.encode()).decode()[:10],
                lambda x: "user_" + str(int(time.time() * 1000))[-6:]
            ]
            
            mutation_func = random.choice(mutations)
            data["username"] = mutation_func(username)
            data["original_username"] = username  # حفظ النسخة الأصلية
            
            # إضافة حقول وهمية مرتبطة
            data["username_variants"] = [
                username + "_backup",
                username.upper() + "_SYSTEM",
                username + "." + secrets.token_hex(3)
            ]
        
        return data
    
    async def _apply_salt_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في الملح"""
        if "salt" in data:
            # توليد ملح جديد
            new_salt = secrets.token_bytes(32)
            data["original_salt"] = data["salt"]
            data["salt"] = base64.b64encode(new_salt).decode()
            
            # إضافة أملاح وهمية
            data["backup_salts"] = [
                base64.b64encode(secrets.token_bytes(32)).decode(),
                base64.b64encode(secrets.token_bytes(32)).decode(),
                base64.b64encode(secrets.token_bytes(32)).decode()
            ]
        
        return data
    
    async def _apply_hash_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في التجزئة"""
        if "password_hash" in data:
            # إنشاء تجزئة جديدة
            new_password = secrets.token_urlsafe(16)
            new_salt = secrets.token_bytes(16)
            
            # تجزئة متعددة الطبقات
            hash1 = hashlib.sha256(new_password.encode() + new_salt).hexdigest()
            hash2 = hashlib.sha512(hash1.encode() + new_salt).hexdigest()
            hash3 = hashlib.blake2b(hash2.encode(), salt=new_salt).hexdigest()
            
            data["original_password_hash"] = data["password_hash"]
            data["password_hash"] = hash3
            data["hash_chain"] = [hash1, hash2, hash3]
            data["hash_version"] = "v3_mutated"
            
            # إضافة تجزئات وهمية
            data["shadow_hashes"] = [
                hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
                hashlib.sha512(secrets.token_bytes(32)).hexdigest(),
                hashlib.blake2b(secrets.token_bytes(32)).hexdigest()
            ]
        
        return data
    
    async def _apply_uuid_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في المعرف الفريد"""
        if "user_id" in data:
            # إنشاء UUID جديد
            new_uuid = str(uuid.uuid4())
            data["original_user_id"] = data["user_id"]
            data["user_id"] = new_uuid
            
            # إنشاء نظام UUIDs وهمي
            data["uuid_cluster"] = {
                "primary": new_uuid,
                "secondary": str(uuid.uuid4()),
                "backup": str(uuid.uuid4()),
                "mirror": str(uuid.uuid4()),
                "shadow": str(uuid.uuid5(uuid.NAMESPACE_DNS, data.get("username", "")))
            }
            
            # إضافة تواقيع UUID
            data["uuid_signatures"] = [
                hashlib.sha256(new_uuid.encode()).hexdigest(),
                hashlib.md5(new_uuid.encode()).hexdigest()[:8],
                base64.b64encode(new_uuid.encode()).decode()
            ]
        
        return data
    
    async def _apply_token_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في التوكن"""
        if "auth_token" in data:
            # توليد توكنات جديدة
            token_secret = secrets.token_urlsafe(32)
            
            # إنشاء توكنات متعددة
            tokens = {
                "primary": self._generate_jwt_token(data, token_secret),
                "backup": self._generate_jwt_token(data, token_secret + "_backup"),
                "mirror": self._generate_jwt_token(data, token_secret + "_mirror"),
                "shadow": self._generate_jwt_token(data, token_secret[::-1])
            }
            
            data["original_token"] = data["auth_token"]
            data["auth_token"] = tokens["primary"]
            data["token_family"] = tokens
            
            # إضافة توكنات وهمية
            data["decoy_tokens"] = [
                secrets.token_urlsafe(64),
                secrets.token_hex(32),
                base64.b64encode(secrets.token_bytes(48)).decode()
            ]
        
        return data
    
    async def _apply_structure_mutation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """طفرة في هيكل البيانات"""
        # تغيير هيكل البيانات
        mutated = {}
        
        for key, value in data.items():
            # تغيير أسماء الحقول
            mutated_key = self._mutate_key_name(key)
            
            # تغيير قيم الحقول
            if isinstance(value, str):
                mutated_value = self._mutate_string_value(value)
            elif isinstance(value, dict):
                mutated_value = await self._apply_structure_mutation(value)
            elif isinstance(value, list):
                mutated_value = [self._mutate_string_value(str(v)) if isinstance(v, str) else v for v in value]
            else:
                mutated_value = value
            
            mutated[mutated_key] = mutated_value
        
        # إضافة حقول وهمية
        fake_fields = {
            "system_metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "mutated_at": datetime.utcnow().isoformat(),
                "mutation_id": str(uuid.uuid4()),
                "data_signature": hashlib.sha256(json.dumps(mutated).encode()).hexdigest()
            },
            "shadow_data": {
                "checksum": secrets.token_hex(16),
                "version": f"mutated_{int(time.time())}",
                "compression": "lz4",
                "encryption": "aes256_gcm"
            }
        }
        
        mutated.update(fake_fields)
        return mutated
    
    def _mutate_key_name(self, key: str) -> str:
        """تغيير اسم الحقل"""
        mutations = [
            lambda k: k + "_encrypted",
            lambda k: "shadow_" + k,
            lambda k: k.upper(),
            lambda k: k.replace("_", ""),
            lambda k: k[::-1],
            lambda k: hashlib.md5(k.encode()).hexdigest()[:8],
            lambda k: base64.b64encode(k.encode()).decode()[:10]
        ]
        
        return random.choice(mutations)(key)
    
    def _mutate_string_value(self, value: str) -> str:
        """تغيير قيمة النص"""
        if len(value) < 5:
            return value
        
        mutations = [
            lambda v: v + secrets.token_hex(4),
            lambda v: base64.b64encode(v.encode()).decode(),
            lambda v: hashlib.sha256(v.encode()).hexdigest(),
            lambda v: v.translate(str.maketrans("aeiouAEIOU", "@3!0@3!0")),
            lambda v: v[::-1],
            lambda v: "".join(random.choices(string.ascii_letters + string.digits, k=8)) + v
        ]
        
        return random.choice(mutations)(value)
    
    def _generate_jwt_token(self, data: Dict[str, Any], secret: str) -> str:
        """إنشاء توكن JWT"""
        payload = {
            "sub": data.get("user_id", "unknown"),
            "username": data.get("username", "unknown"),
            "mutated": True,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),
            "mutation_timestamp": int(time.time()),
            "attack_response": "chemical_mutation"
        }
        
        return jwt.encode(payload, secret, algorithm="HS256")
    
    async def _generate_fake_copies(self, original_data: Dict[str, Any], 
                                   count: int) -> List[Dict[str, Any]]:
        """إنشاء نسخ وهمية من البيانات"""
        fake_copies = []
        
        for i in range(count):
            fake_copy = original_data.copy()
            
            # تطبيق تغييرات عشوائية على كل نسخة
            fake_copy["is_fake"] = True
            fake_copy["fake_index"] = i + 1
            fake_copy["fake_id"] = str(uuid.uuid4())
            
            # تغيير اسم المستخدم في النسخة الوهمية
            if "username" in fake_copy:
                patterns = SecurityConfig.FAKE_USER_PATTERNS
                pattern = random.choice(patterns)
                
                fake_username = pattern.format(
                    username=fake_copy["username"],
                    index=i + 1,
                    random=secrets.token_hex(4),
                    timestamp=int(time.time())
                )
                fake_copy["username"] = fake_username
            
            # تغيير المعرفات
            fake_copy["user_id"] = str(uuid.uuid4())
            fake_copy["auth_token"] = secrets.token_urlsafe(32)
            
            # إضافة بيانات وهمية إضافية
            fake_copy["fake_metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "generation_reason": "chemical_reaction_defense",
                "copy_type": random.choice(["backup", "mirror", "shadow", "archive", "temp"]),
                "valid_until": (datetime.utcnow() + timedelta(hours=random.randint(1, 24))).isoformat()
            }
            
            fake_copies.append(fake_copy)
        
        return fake_copies
    
    async def _store_fake_copies(self, fake_copies: List[Dict[str, Any]], 
                                attack_data: Dict[str, Any]):
        """تخزين النسخ الوهمية"""
        attack_id = attack_data.get("attack_id", "unknown")
        
        for fake_copy in fake_copies:
            fake_id = fake_copy.get("fake_id", str(uuid.uuid4()))
            storage_key = f"fake_data:{attack_id}:{fake_id}"
            
            # تخزين في Redis
            await self.redis.setex(
                storage_key,
                3600,  # ساعة واحدة
                json.dumps(fake_copy)
            )
            
            # تسجيل في سجلات النظام
            await logger.info(
                "Fake data copy stored",
                fake_id=fake_id,
                attack_id=attack_id,
                username=fake_copy.get("username", "unknown"),
                storage_key=storage_key
            )
    
    async def _record_mutation(self, mutation_id: str, attack_data: Dict[str, Any],
                              original_data: Dict[str, Any], mutated_data: Dict[str, Any]):
        """تسجيل تفاصيل الطفرة"""
        mutation_record = {
            "mutation_id": mutation_id,
            "attack_data": attack_data,
            "original_data_snapshot": {
                "hash": hashlib.sha256(json.dumps(original_data).encode()).hexdigest(),
                "keys": list(original_data.keys())
            },
            "mutated_data_snapshot": {
                "hash": hashlib.sha256(json.dumps(mutated_data).encode()).hexdigest(),
                "keys": list(mutated_data.keys())
            },
            "mutation_timestamp": datetime.utcnow().isoformat(),
            "mutation_duration_ms": attack_data.get("detection_time_ms", 0)
        }
        
        # تخزين التسجيل
        record_key = f"mutation_record:{mutation_id}"
        await self.redis.setex(
            record_key,
            86400,  # 24 ساعة
            json.dumps(mutation_record)
        )
        
        self.mutation_history[mutation_id] = mutation_record
        return mutation_id

# ==================== NEW FEATURE: CASCADE REACTION DEFENSE ====================
class CascadeReactionDefense:
    """
    تفاعل متسلسل يولد استجابات وهمية متعددة عند اكتشاف هجوم
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.reaction_executor = ThreadPoolExecutor(max_workers=20)
        self.active_cascades = {}
    
    async def trigger_cascade_reaction(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تشغيل تفاعل متسلسل من الاستجابات الوهمية
        """
        attack_id = attack_data.get("attack_id", str(uuid.uuid4()))
        target_username = attack_data.get("target_username", "unknown")
        attack_type = attack_data.get("attack_type", "unknown")
        
        # إنشاء تفاعل متسلسل
        cascade_id = str(uuid.uuid4())
        self.active_cascades[cascade_id] = {
            "attack_id": attack_id,
            "started_at": datetime.utcnow(),
            "reactions_generated": 0,
            "status": "active"
        }
        
        # بدء التفاعل في الخلفية
        asyncio.create_task(
            self._execute_cascade_reactions(
                cascade_id, attack_id, target_username, attack_type
            )
        )
        
        # تسجيل بدء التفاعل
        await logger.warning(
            "Cascade reaction defense activated",
            cascade_id=cascade_id,
            attack_id=attack_id,
            target_username=target_username,
            attack_type=attack_type,
            expected_reactions=SecurityConfig.CASCADE_REACTIONS_PER_ATTACK
        )
        
        return {
            "cascade_id": cascade_id,
            "attack_id": attack_id,
            "target_username": target_username,
            "attack_type": attack_type,
            "reactions_triggered": SecurityConfig.CASCADE_REACTIONS_PER_ATTACK,
            "status": "initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_cascade_reactions(self, cascade_id: str, attack_id: str,
                                        target_username: str, attack_type: str):
        """تنفيذ التفاعلات المتسلسلة"""
        reactions_generated = 0
        
        for reaction_num in range(SecurityConfig.CASCADE_REACTIONS_PER_ATTACK):
            try:
                # تأخير بين التفاعلات لمحاكاة السلسلة
                await asyncio.sleep(SecurityConfig.CASCADE_DELAY_BETWEEN_REACTIONS)
                
                # توليد تفاعل وهمي
                reaction_data = await self._generate_fake_reaction(
                    reaction_num + 1, target_username, attack_type
                )
                
                # تخزين التفاعل
                await self._store_reaction(
                    cascade_id, attack_id, reaction_num + 1, reaction_data
                )
                
                # محاكاة تأثيرات جانبية
                await self._simulate_side_effects(reaction_data)
                
                reactions_generated += 1
                
                # تحديث حالة التفاعل
                if cascade_id in self.active_cascades:
                    self.active_cascades[cascade_id]["reactions_generated"] = reactions_generated
                
                await logger.debug(
                    "Cascade reaction generated",
                    cascade_id=cascade_id,
                    reaction_number=reaction_num + 1,
                    reaction_type=reaction_data.get("reaction_type"),
                    fake_username=reaction_data.get("fake_username")
                )
                
            except Exception as e:
                await logger.error(
                    "Failed to generate cascade reaction",
                    cascade_id=cascade_id,
                    reaction_number=reaction_num + 1,
                    error=str(e)
                )
        
        # إنهاء التفاعل
        if cascade_id in self.active_cascades:
            self.active_cascades[cascade_id]["status"] = "completed"
            self.active_cascades[cascade_id]["completed_at"] = datetime.utcnow()
            
            # تسجيل إحصائيات التفاعل
            await self._record_cascade_statistics(cascade_id, attack_id, reactions_generated)
            
            # تنظيف بعد فترة
            asyncio.create_task(self._cleanup_cascade(cascade_id))
        
        await logger.info(
            "Cascade reaction defense completed",
            cascade_id=cascade_id,
            attack_id=attack_id,
            total_reactions=reactions_generated
        )
    
    async def _generate_fake_reaction(self, reaction_num: int, 
                                     target_username: str, 
                                     attack_type: str) -> Dict[str, Any]:
        """توليد تفاعل وهمي"""
        reaction_types = [
            "user_account_creation",
            "data_backup",
            "system_migration",
            "security_scan",
            "access_log_update",
            "session_regeneration",
            "token_refresh",
            "password_reset",
            "account_verification",
            "data_synchronization"
        ]
        
        # اختيار نوع التفاعل
        reaction_type = random.choice(reaction_types)
        
        # إنشاء اسم مستخدم وهمي
        fake_username = self._generate_fake_username(target_username, reaction_num)
        
        # إنشاء بيانات التفاعل
        reaction_data = {
            "reaction_id": str(uuid.uuid4()),
            "reaction_number": reaction_num,
            "reaction_type": reaction_type,
            "original_target": target_username,
            "fake_username": fake_username,
            "attack_type": attack_type,
            "timestamp": datetime.utcnow().isoformat(),
            "system_impact": random.choice(["low", "medium", "high"]),
            "response_code": random.choice([200, 201, 202, 204, 301, 302]),
            "fake_data": self._generate_fake_response_data(fake_username, reaction_type)
        }
        
        return reaction_data
    
    def _generate_fake_username(self, original_username: str, index: int) -> str:
        """إنشاء اسم مستخدم وهمي"""
        patterns = SecurityConfig.FAKE_USER_PATTERNS
        
        # اختيار نمط عشوائي
        pattern = random.choice(patterns)
        
        # تعبئة النمط
        return pattern.format(
            username=original_username,
            index=index,
            random=secrets.token_hex(4),
            timestamp=int(time.time())
        )
    
    def _generate_fake_response_data(self, fake_username: str, 
                                    reaction_type: str) -> Dict[str, Any]:
        """إنشاء بيانات استجابة وهمية"""
        fake_data_templates = {
            "user_account_creation": {
                "status": "created",
                "user_id": str(uuid.uuid4()),
                "email": f"{fake_username}@system.backup",
                "created_at": datetime.utcnow().isoformat(),
                "account_type": "backup_user",
                "permissions": ["read_only", "backup_access"],
                "storage_quota": random.randint(100, 1000)
            },
            "data_backup": {
                "backup_id": str(uuid.uuid4()),
                "username": fake_username,
                "backup_type": "full_system",
                "size_gb": random.uniform(1.0, 50.0),
                "compression_ratio": random.uniform(2.0, 5.0),
                "encryption": "aes256_gcm",
                "backup_location": f"/backup/system/{fake_username}/{int(time.time())}",
                "checksum": secrets.token_hex(32)
            },
            "system_migration": {
                "migration_id": str(uuid.uuid4()),
                "source_user": fake_username,
                "target_cluster": f"cluster_{random.randint(1, 10)}",
                "migration_status": "in_progress",
                "estimated_completion": (datetime.utcnow() + timedelta(minutes=random.randint(5, 60))).isoformat(),
                "data_transferred_mb": random.randint(100, 10000),
                "migration_type": "hot_migration"
            },
            "security_scan": {
                "scan_id": str(uuid.uuid4()),
                "target": fake_username,
                "scan_type": "deep_security",
                "vulnerabilities_found": random.randint(0, 5),
                "threat_level": random.choice(["low", "medium", "high"]),
                "recommendations": [
                    "update_permissions",
                    "enable_2fa",
                    "review_access_logs"
                ],
                "scan_duration_seconds": random.randint(30, 300)
            }
        }
        
        # استخدام قالب أو إنشاء عام
        template = fake_data_templates.get(
            reaction_type,
            {
                "status": "processed",
                "operation": reaction_type,
                "target": fake_username,
                "timestamp": datetime.utcnow().isoformat(),
                "transaction_id": str(uuid.uuid4()),
                "system_response": "operation_completed"
            }
        )
        
        return template
    
    async def _store_reaction(self, cascade_id: str, attack_id: str,
                             reaction_num: int, reaction_data: Dict[str, Any]):
        """تخزين تفاعل السلسلة"""
        storage_key = f"cascade_reaction:{cascade_id}:{reaction_num}"
        
        # تخزين في Redis
        await self.redis.setex(
            storage_key,
            7200,  # ساعتين
            json.dumps(reaction_data)
        )
        
        # تحديث مؤشر التفاعلات
        index_key = f"cascade_index:{cascade_id}"
        await self.redis.zadd(
            index_key,
            {str(reaction_num): reaction_num}
        )
        await self.redis.expire(index_key, 7200)
    
    async def _simulate_side_effects(self, reaction_data: Dict[str, Any]):
        """محاكاة تأثيرات جانبية للتفاعل"""
        side_effects = [
            self._simulate_log_entries,
            self._simulate_network_traffic,
            self._simulate_database_activity,
            self._simulate_cache_updates,
            self._simulate_notifications
        ]
        
        # تنفيذ تأثيرات جانبية عشوائية
        selected_effects = random.sample(side_effects, random.randint(2, 4))
        
        for effect_func in selected_effects:
            try:
                await effect_func(reaction_data)
            except Exception as e:
                await logger.debug(f"Side effect simulation failed: {str(e)}")
    
    async def _simulate_log_entries(self, reaction_data: Dict[str, Any]):
        """محاكاة إدخالات السجلات"""
        log_entries = [
            f"User {reaction_data.get('fake_username')} authenticated successfully",
            f"Backup completed for {reaction_data.get('fake_username')}",
            f"Security scan initiated for {reaction_data.get('fake_username')}",
            f"Data migration started for user {reaction_data.get('fake_username')}",
            f"Account verification email sent to {reaction_data.get('fake_username')}",
            f"Session renewed for {reaction_data.get('fake_username')}",
            f"Access permissions updated for {reaction_data.get('fake_username')}"
        ]
        
        log_entry = random.choice(log_entries)
        log_key = f"system_log:cascade:{int(time.time())}:{secrets.token_hex(4)}"
        
        await self.redis.setex(
            log_key,
            3600,
            json.dumps({
                "message": log_entry,
                "timestamp": datetime.utcnow().isoformat(),
                "reaction_id": reaction_data.get("reaction_id"),
                "severity": random.choice(["INFO", "DEBUG", "NOTICE"])
            })
        )
    
    async def _simulate_network_traffic(self, reaction_data: Dict[str, Any]):
        """محاكاة حركة مرور الشبكة"""
        traffic_data = {
            "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "destination_ip": f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
            "bytes_sent": random.randint(1024, 1048576),
            "bytes_received": random.randint(1024, 524288),
            "protocol": random.choice(["TCP", "UDP", "HTTP", "HTTPS"]),
            "port": random.randint(1024, 65535),
            "duration_ms": random.randint(10, 1000),
            "reaction_id": reaction_data.get("reaction_id")
        }
        
        traffic_key = f"network_traffic:cascade:{int(time.time())}:{secrets.token_hex(4)}"
        await self.redis.setex(traffic_key, 1800, json.dumps(traffic_data))
    
    async def _simulate_database_activity(self, reaction_data: Dict[str, Any]):
        """محاكاة نشاط قاعدة البيانات"""
        db_operations = ["INSERT", "UPDATE", "SELECT", "DELETE", "CREATE"]
        
        activity_data = {
            "database": f"honey_db_{secrets.token_hex(8)}",
            "table": f"users_{reaction_data.get('fake_username', 'unknown')}",
            "operation": random.choice(db_operations),
            "rows_affected": random.randint(1, 1000),
            "execution_time_ms": random.randint(1, 100),
            "query_hash": hashlib.md5(str(time.time()).encode()).hexdigest(),
            "reaction_id": reaction_data.get("reaction_id")
        }
        
        activity_key = f"db_activity:cascade:{int(time.time())}:{secrets.token_hex(4)}"
        await self.redis.setex(activity_key, 1800, json.dumps(activity_data))
    
    async def _simulate_cache_updates(self, reaction_data: Dict[str, Any]):
        """محاكاة تحديثات التخزين المؤقت"""
        cache_keys = [
            f"user:{reaction_data.get('fake_username')}:profile",
            f"session:{secrets.token_hex(16)}",
            f"token:{secrets.token_urlsafe(32)}",
            f"data:{reaction_data.get('fake_username')}:backup",
            f"system:cache:{secrets.token_hex(8)}"
        ]
        
        for cache_key in random.sample(cache_keys, random.randint(1, 3)):
            cache_data = {
                "value": secrets.token_urlsafe(32),
                "updated_at": datetime.utcnow().isoformat(),
                "ttl": random.randint(300, 3600),
                "reaction_id": reaction_data.get("reaction_id")
            }
            
            await self.redis.setex(
                f"cache_simulation:{cache_key}",
                cache_data["ttl"],
                json.dumps(cache_data)
            )
    
    async def _simulate_notifications(self, reaction_data: Dict[str, Any]):
        """محاكاة إشعارات النظام"""
        notification_types = ["email", "sms", "push", "webhook", "system_alert"]
        
        notification = {
            "type": random.choice(notification_types),
            "recipient": reaction_data.get("fake_username"),
            "subject": f"System Notification: {reaction_data.get('reaction_type', 'Operation')}",
            "content": f"Your {reaction_data.get('reaction_type', 'request')} has been processed successfully.",
            "priority": random.choice(["low", "normal", "high"]),
            "sent_at": datetime.utcnow().isoformat(),
            "reaction_id": reaction_data.get("reaction_id")
        }
        
        notification_key = f"notification:cascade:{int(time.time())}:{secrets.token_hex(4)}"
        await self.redis.setex(notification_key, 3600, json.dumps(notification))
    
    async def _record_cascade_statistics(self, cascade_id: str, attack_id: str,
                                        reactions_generated: int):
        """تسجيل إحصائيات التفاعل المتسلسل"""
        stats_key = f"cascade_stats:{cascade_id}"
        
        stats = {
            "cascade_id": cascade_id,
            "attack_id": attack_id,
            "reactions_generated": reactions_generated,
            "start_time": self.active_cascades[cascade_id]["started_at"].isoformat(),
            "completion_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - self.active_cascades[cascade_id]["started_at"]).total_seconds(),
            "average_reaction_interval": SecurityConfig.CASCADE_DELAY_BETWEEN_REACTIONS,
            "success_rate": 1.0  # يمكن تحسينه لتسجيل النجاح/الفشل الفعلي
        }
        
        await self.redis.setex(stats_key, 86400, json.dumps(stats))
    
    async def _cleanup_cascade(self, cascade_id: str):
        """تنظيف بيانات التفاعل بعد فترة"""
        await asyncio.sleep(7200)  # انتظار ساعتين
        
        if cascade_id in self.active_cascades:
            del self.active_cascades[cascade_id]
            
            # محاولة حذف البيانات المؤقتة
            try:
                # حذف فهرس التفاعلات
                index_key = f"cascade_index:{cascade_id}"
                await self.redis.delete(index_key)
                
                # حذف إحصائيات التفاعل
                stats_key = f"cascade_stats:{cascade_id}"
                await self.redis.delete(stats_key)
                
            except Exception as e:
                await logger.debug(f"Cascade cleanup failed: {str(e)}")
    
    async def get_cascade_status(self, cascade_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على حالة التفاعل المتسلسل"""
        if cascade_id not in self.active_cascades:
            return None
        
        cascade_data = self.active_cascades[cascade_id].copy()
        
        # إضافة معلومات إضافية من Redis
        stats_key = f"cascade_stats:{cascade_id}"
        stats_data = await self.redis.get(stats_key)
        
        if stats_data:
            cascade_data["statistics"] = json.loads(stats_data)
        
        # جلب بعض التفاعلات الأخيرة
        reaction_keys = await self.redis.keys(f"cascade_reaction:{cascade_id}:*")
        recent_reactions = []
        
        for key in reaction_keys[:10]:  # آخر 10 تفاعلات
            reaction_data = await self.redis.get(key)
            if reaction_data:
                recent_reactions.append(json.loads(reaction_data))
        
        cascade_data["recent_reactions"] = recent_reactions
        cascade_data["total_reactions_stored"] = len(reaction_keys)
        
        return cascade_data

# ==================== NEW FEATURE: PER-ATTACK DYNAMIC HONEY DATABASE ====================
class PerAttackDynamicHoneyDatabase:
    """
    قاعدة بيانات وهمية ديناميكية خاصة بكل هجوم
    """
    
    def __init__(self, redis_client, postgres_pool, mongo_client):
        self.redis = redis_client
        self.postgres = postgres_pool
        self.mongo = mongo_client
        self.honey_databases = {}
        self.attack_profiles = {}
    
    async def create_honey_database(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        إنشاء قاعدة بيانات وهمية ديناميكية بناءً على بيانات الهجوم
        """
        attack_id = attack_data.get("attack_id", str(uuid.uuid4()))
        attacker_ip = attack_data.get("attacker_ip", "unknown")
        fingerprint = attack_data.get("fingerprint", {})
        attack_type = attack_data.get("attack_type", "unknown")
        
        # إنشاء معرف فريد لقاعدة البيانات الوهمية
        honey_db_id = self._generate_honey_db_id(attacker_ip, fingerprint, attack_type)
        
        # إنشاء قاعدة البيانات الوهمية
        honey_db = await self._build_honey_database(
            honey_db_id, attack_data, fingerprint
        )
        
        # تخزين ملف تعريف الهجوم
        await self._store_attack_profile(attack_id, attacker_ip, fingerprint, attack_type, honey_db_id)
        
        # تسجيل إنشاء قاعدة البيانات الوهمية
        await logger.warning(
            "Dynamic honey database created",
            honey_db_id=honey_db_id,
            attack_id=attack_id,
            attacker_ip=attacker_ip,
            attack_type=attack_type,
            db_size_entries=len(honey_db.get("tables", {}))
        )
        
        return {
            "honey_db_id": honey_db_id,
            "attack_id": attack_id,
            "attacker_ip": attacker_ip,
            "attack_type": attack_type,
            "created_at": datetime.utcnow().isoformat(),
            "database_schema": honey_db.get("schema"),
            "total_tables": len(honey_db.get("tables", {})),
            "total_fake_records": honey_db.get("total_records", 0),
            "access_credentials": honey_db.get("access_credentials"),
            "database_features": honey_db.get("features")
        }
    
    def _generate_honey_db_id(self, ip: str, fingerprint: Dict[str, Any], 
                             attack_type: str) -> str:
        """إنشاء معرف فريد لقاعدة البيانات الوهمية"""
        # إنشاء هاش من بيانات المهاجم
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        combined = f"{ip}:{fingerprint_str}:{attack_type}:{int(time.time())}"
        
        honey_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"{SecurityConfig.HONEY_DB_PREFIX}{honey_hash}"
    
    async def _build_honey_database(self, honey_db_id: str, 
                                   attack_data: Dict[str, Any],
                                   fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """بناء قاعدة بيانات وهمية مخصصة"""
        # تحليل بصمة المهاجم لتخصيص قاعدة البيانات
        db_type = self._determine_db_type_from_fingerprint(fingerprint)
        complexity = self._determine_complexity_from_attack(attack_data)
        
        # إنشاء مخطط قاعدة البيانات
        schema = await self._generate_db_schema(db_type, complexity, attack_data)
        
        # إنشاء بيانات وهمية
        fake_data = await self._generate_fake_data(schema, complexity)
        
        # إنشاء وثائق وهمية في MongoDB
        mongo_collections = await self._create_mongo_collections(honey_db_id, fake_data)
        
        # إنشاء جداول وهمية في PostgreSQL
        postgres_tables = await self._create_postgres_tables(honey_db_id, schema)
        
        # إنشاء بيانات وهمية في Redis
        redis_data = await self._create_redis_structures(honey_db_id, fake_data)
        
        # إنشاء بيانات وصول وهمية
        access_creds = self._generate_access_credentials(honey_db_id)
        
        # تجميع قاعدة البيانات الوهمية
        honey_db = {
            "id": honey_db_id,
            "type": db_type,
            "complexity": complexity,
            "schema": schema,
            "tables": postgres_tables,
            "collections": mongo_collections,
            "redis_structures": redis_data,
            "fake_data_samples": fake_data[:5],  # عينات فقط
            "access_credentials": access_creds,
            "total_records": sum(len(table.get("rows", [])) for table in postgres_tables.values()),
            "features": self._generate_db_features(db_type, complexity),
            "created_at": datetime.utcnow().isoformat(),
            "valid_until": (datetime.utcnow() + timedelta(seconds=SecurityConfig.HONEY_DB_TTL)).isoformat()
        }
        
        # تخزين في الذاكرة
        self.honey_databases[honey_db_id] = honey_db
        
        # تخزين في Redis
        await self.redis.setex(
            f"honey_db:{honey_db_id}",
            SecurityConfig.HONEY_DB_TTL,
            json.dumps(honey_db)
        )
        
        return honey_db
    
    def _determine_db_type_from_fingerprint(self, fingerprint: Dict[str, Any]) -> str:
        """تحديد نوع قاعدة البيانات من بصمة المهاجم"""
        user_agent = fingerprint.get("user_agent", "").lower()
        
        if any(db in user_agent for db in ["mysql", "mariadb"]):
            return "mysql"
        elif "postgresql" in user_agent or "psql" in user_agent:
            return "postgresql"
        elif "mongodb" in user_agent:
            return "mongodb"
        elif "redis" in user_agent:
            return "redis"
        elif "sqlite" in user_agent:
            return "sqlite"
        elif "oracle" in user_agent:
            return "oracle"
        elif "sqlserver" in user_agent or "mssql" in user_agent:
            return "sqlserver"
        else:
            # نوع افتراضي بناءً على الصعوبة المتوقعة
            return random.choice(["mysql", "postgresql", "mongodb"])
    
    def _determine_complexity_from_attack(self, attack_data: Dict[str, Any]) -> str:
        """تحديد مدى تعقيد قاعدة البيانات من نوع الهجوم"""
        attack_type = attack_data.get("attack_type", "unknown")
        severity = attack_data.get("severity", "medium")
        
        complexity_map = {
            "sql_injection": "high",
            "brute_force": "medium",
            "xss_attempt": "low",
            "data_exfiltration": "high",
            "unauthorized_access": "medium",
            "token_theft": "high"
        }
        
        base_complexity = complexity_map.get(attack_type, "medium")
        
        # تعديل التعقيد بناءً على الشدة
        if severity == "critical":
            return "very_high"
        elif severity == "high":
            return "high"
        elif severity == "low":
            return "low"
        else:
            return base_complexity
    
    async def _generate_db_schema(self, db_type: str, complexity: str,
                                 attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """إنشاء مخطط قاعدة بيانات وهمي"""
        # جداول أساسية شائعة
        base_tables = {
            "users": {
                "columns": [
                    {"name": "id", "type": "INT", "primary_key": True},
                    {"name": "username", "type": "VARCHAR(255)"},
                    {"name": "email", "type": "VARCHAR(255)"},
                    {"name": "password_hash", "type": "VARCHAR(512)"},
                    {"name": "created_at", "type": "TIMESTAMP"},
                    {"name": "last_login", "type": "TIMESTAMP"}
                ],
                "indexes": ["username", "email"],
                "estimated_rows": 1000
            },
            "sessions": {
                "columns": [
                    {"name": "session_id", "type": "VARCHAR(255)", "primary_key": True},
                    {"name": "user_id", "type": "INT"},
                    {"name": "ip_address", "type": "VARCHAR(45)"},
                    {"name": "user_agent", "type": "TEXT"},
                    {"name": "created_at", "type": "TIMESTAMP"},
                    {"name": "expires_at", "type": "TIMESTAMP"}
                ],
                "foreign_keys": [{"column": "user_id", "references": "users(id)"}],
                "estimated_rows": 5000
            }
        }
        
        # إضافة جداول بناءً على التعقيد
        additional_tables = {}
        
        if complexity in ["medium", "high", "very_high"]:
            additional_tables.update({
                "user_profiles": {
                    "columns": [
                        {"name": "user_id", "type": "INT", "primary_key": True},
                        {"name": "full_name", "type": "VARCHAR(255)"},
                        {"name": "phone", "type": "VARCHAR(20)"},
                        {"name": "address", "type": "TEXT"},
                        {"name": "date_of_birth", "type": "DATE"},
                        {"name": "profile_picture", "type": "VARCHAR(512)"}
                    ],
                    "estimated_rows": 1000
                },
                "messages": {
                    "columns": [
                        {"name": "message_id", "type": "BIGINT", "primary_key": True},
                        {"name": "sender_id", "type": "INT"},
                        {"name": "receiver_id", "type": "INT"},
                        {"name": "content", "type": "TEXT"},
                        {"name": "encrypted", "type": "BOOLEAN"},
                        {"name": "sent_at", "type": "TIMESTAMP"},
                        {"name": "read_at", "type": "TIMESTAMP"}
                    ],
                    "estimated_rows": 10000
                }
            })
        
        if complexity in ["high", "very_high"]:
            additional_tables.update({
                "payments": {
                    "columns": [
                        {"name": "payment_id", "type": "VARCHAR(255)", "primary_key": True},
                        {"name": "user_id", "type": "INT"},
                        {"name": "amount", "type": "DECIMAL(10,2)"},
                        {"name": "currency", "type": "VARCHAR(3)"},
                        {"name": "status", "type": "VARCHAR(20)"},
                        {"name": "payment_method", "type": "VARCHAR(50)"},
                        {"name": "transaction_data", "type": "JSON"}
                    ],
                    "estimated_rows": 5000
                },
                "security_logs": {
                    "columns": [
                        {"name": "log_id", "type": "BIGINT", "primary_key": True},
                        {"name": "user_id", "type": "INT"},
                        {"name": "action", "type": "VARCHAR(100)"},
                        {"name": "ip_address", "type": "VARCHAR(45)"},
                        {"name": "user_agent", "type": "TEXT"},
                        {"name": "timestamp", "type": "TIMESTAMP"},
                        {"name": "details", "type": "JSON"}
                    ],
                    "estimated_rows": 100000
                }
            })
        
        if complexity == "very_high":
            additional_tables.update({
                "encryption_keys": {
                    "columns": [
                        {"name": "key_id", "type": "VARCHAR(255)", "primary_key": True},
                        {"name": "user_id", "type": "INT"},
                        {"name": "key_type", "type": "VARCHAR(50)"},
                        {"name": "public_key", "type": "TEXT"},
                        {"name": "private_key_encrypted", "type": "TEXT"},
                        {"name": "created_at", "type": "TIMESTAMP"},
                        {"name": "expires_at", "type": "TIMESTAMP"}
                    ],
                    "estimated_rows": 1000
                },
                "audit_trail": {
                    "columns": [
                        {"name": "audit_id", "type": "BIGINT", "primary_key": True},
                        {"name": "table_name", "type": "VARCHAR(255)"},
                        {"name": "record_id", "type": "VARCHAR(255)"},
                        {"name": "action", "type": "VARCHAR(50)"},
                        {"name": "old_values", "type": "JSON"},
                        {"name": "new_values", "type": "JSON"},
                        {"name": "changed_by", "type": "INT"},
                        {"name": "changed_at", "type": "TIMESTAMP"}
                    ],
                    "estimated_rows": 50000
                }
            })
        
        # دمج الجداول
        all_tables = {**base_tables, **additional_tables}
        
        # إضافة حقول خاصة بنوع الهجوم
        if attack_data.get("attack_type") == "sql_injection":
            # إضافة حقول قد تكون مستهدفة في حقن SQL
            for table_name, table_def in all_tables.items():
                if "columns" in table_def:
                    table_def["columns"].extend([
                        {"name": "is_admin", "type": "BOOLEAN", "default": "false"},
                        {"name": "access_level", "type": "INT", "default": "1"}
                    ])
        
        return {
            "database_type": db_type,
            "complexity": complexity,
            "tables": all_tables,
            "total_tables": len(all_tables),
            "total_estimated_rows": sum(t.get("estimated_rows", 0) for t in all_tables.values()),
            "schema_version": "honey_1.0",
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_fake_data(self, schema: Dict[str, Any], 
                                 complexity: str) -> List[Dict[str, Any]]:
        """إنشاء بيانات وهمية للمخطط"""
        fake_data = []
        tables = schema.get("tables", {})
        
        for table_name, table_def in tables.items():
            estimated_rows = table_def.get("estimated_rows", 100)
            
            # إنشاء بيانات وهمية للجدول
            for i in range(min(estimated_rows, 100)):  # حد أقصى 100 سجل لكل جدول
                row_data = {"table": table_name, "row_id": i + 1}
                
                for column in table_def.get("columns", []):
                    column_name = column["name"]
                    column_type = column["type"].lower()
                    
                    # توليد بيانات وهمية بناءً على نوع العمود
                    row_data[column_name] = self._generate_fake_column_value(
                        column_name, column_type, i + 1
                    )
                
                fake_data.append(row_data)
        
        return fake_data
    
    def _generate_fake_column_value(self, column_name: str, column_type: str, 
                                   row_id: int) -> Any:
        """توليد قيمة وهمية للعمود"""
        column_lower = column_name.lower()
        
        # التعامل مع الأعمدة الشائعة
        if column_lower in ["id", "user_id", "sender_id", "receiver_id"]:
            return row_id
        
        elif column_lower == "username":
            return f"user_{row_id}_{secrets.token_hex(4)}"
        
        elif column_lower == "email":
            return f"user{row_id}@honeydb.example.com"
        
        elif column_lower == "password_hash":
            return hashlib.sha256(f"password_{row_id}".encode()).hexdigest()
        
        elif column_lower in ["created_at", "last_login", "sent_at", "timestamp"]:
            offset = random.randint(0, 30*24*3600)  # حتى 30 يوم عشوائي
            return (datetime.utcnow() - timedelta(seconds=offset)).isoformat()
        
        elif "name" in column_lower:
            names = ["Ahmed", "Mohamed", "Fatima", "Aisha", "Omar", "Khalid"]
            return random.choice(names) + " " + random.choice(["Al", "Bin", "Ibn"]) + " " + random.choice(names)
        
        elif "phone" in column_lower:
            return f"+9665{random.randint(10000000, 99999999)}"
        
        elif "address" in column_lower:
            addresses = ["Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia", 
                        "Dammam, Saudi Arabia", "Mecca, Saudi Arabia"]
            return random.choice(addresses)
        
        elif column_type.startswith("varchar") or column_type == "text":
            lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            return lorem[:random.randint(10, 100)]
        
        elif column_type == "int":
            return random.randint(1, 10000)
        
        elif column_type == "boolean":
            return random.choice([True, False])
        
        elif column_type == "decimal":
            return round(random.uniform(1.0, 1000.0), 2)
        
        elif column_type == "json":
            return json.dumps({"fake": True, "id": row_id, "timestamp": int(time.time())})
        
        else:
            return secrets.token_hex(8)
    
    async def _create_mongo_collections(self, honey_db_id: str, 
                                       fake_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """إنشاء مجموعات وهمية في MongoDB"""
        # تجميع البيانات حسب الجدول
        collections_data = {}
        
        for row in fake_data:
            table_name = row.pop("table")  # إزالة حقل الجدول
            row_id = row.pop("row_id", 0)
            
            if table_name not in collections_data:
                collections_data[table_name] = []
            
            # إضافة معرف MongoDB
            row["_id"] = f"{honey_db_id}:{table_name}:{row_id}"
            row["honey_db_id"] = honey_db_id
            row["is_honey_data"] = True
            row["created_at"] = datetime.utcnow().isoformat()
            
            collections_data[table_name].append(row)
        
        # في الإنتاج، سيتم تخزين في MongoDB فعلياً
        # هنا سنخزن في Redis لمحاكاة MongoDB
        mongo_key = f"honey_mongo:{honey_db_id}"
        
        await self.redis.setex(
            mongo_key,
            SecurityConfig.HONEY_DB_TTL,
            json.dumps(collections_data)
        )
        
        return {
            "database": honey_db_id,
            "collections": list(collections_data.keys()),
            "total_documents": sum(len(docs) for docs in collections_data.values()),
            "storage_simulation": "redis_backed"
        }
    
    async def _create_postgres_tables(self, honey_db_id: str, 
                                     schema: Dict[str, Any]) -> Dict[str, Any]:
        """إنشاء جداول وهمية في PostgreSQL"""
        # في الإنتاج، سيتم إنشاء جداول فعلية
        # هنا نعود بهيكل وهمي
        
        tables_info = {}
        
        for table_name, table_def in schema.get("tables", {}).items():
            tables_info[table_name] = {
                "columns": table_def.get("columns", []),
                "estimated_rows": table_def.get("estimated_rows", 0),
                "primary_keys": [col["name"] for col in table_def.get("columns", []) 
                               if col.get("primary_key")],
                "indexes": table_def.get("indexes", []),
                "foreign_keys": table_def.get("foreign_keys", [])
            }
        
        # تخزين هيكل الجداول في Redis
        tables_key = f"honey_postgres:{honey_db_id}"
        
        await self.redis.setex(
            tables_key,
            SecurityConfig.HONEY_DB_TTL,
            json.dumps(tables_info)
        )
        
        return tables_info
    
    async def _create_redis_structures(self, honey_db_id: str, 
                                      fake_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """إنشاء هياكل وهمية في Redis"""
        redis_structures = {
            "strings": {},
            "hashes": {},
            "lists": {},
            "sets": {},
            "sorted_sets": {}
        }
        
        # إنشاء بيانات وهمية لكل نوع من هياكل Redis
        for row in fake_data[:50]:  # أول 50 سجل فقط
            table_name = row.get("table", "unknown")
            row_id = row.get("row_id", 0)
            
            # مفتاح سلسلة
            string_key = f"{honey_db_id}:{table_name}:{row_id}"
            redis_structures["strings"][string_key] = json.dumps(row)
            
            # هاش
            hash_key = f"{honey_db_id}:hash:{table_name}:{row_id}"
            redis_structures["hashes"][hash_key] = {
                "id": str(row_id),
                "table": table_name,
                "timestamp": int(time.time()),
                "data_hash": hashlib.md5(json.dumps(row).encode()).hexdigest()
            }
            
            # قائمة (للمستخدمين)
            if table_name == "users":
                list_key = f"{honey_db_id}:list:users"
                if list_key not in redis_structures["lists"]:
                    redis_structures["lists"][list_key] = []
                redis_structures["lists"][list_key].append(row.get("username", "unknown"))
            
            # مجموعة (للعناوين)
            if "ip_address" in row:
                set_key = f"{honey_db_id}:set:ip_addresses"
                if set_key not in redis_structures["sets"]:
                    redis_structures["sets"][set_key] = set()
                redis_structures["sets"][set_key].add(row["ip_address"])
            
            # مجموعة مرتبة (للتواريخ)
            if "created_at" in row:
                zset_key = f"{honey_db_id}:zset:created_times"
                if zset_key not in redis_structures["sorted_sets"]:
                    redis_structures["sorted_sets"][zset_key] = {}
                score = int(datetime.fromisoformat(row["created_at"]).timestamp())
                redis_structures["sorted_sets"][zset_key][row_id] = score
        
        # تخزين في Redis فعلياً
        for key_type, data in redis_structures.items():
            if key_type == "strings":
                for key, value in data.items():
                    await self.redis.setex(key, SecurityConfig.HONEY_DB_TTL, value)
            elif key_type == "hashes":
                for key, hash_data in data.items():
                    await self.redis.hset(key, mapping=hash_data)
                    await self.redis.expire(key, SecurityConfig.HONEY_DB_TTL)
            elif key_type == "lists":
                for key, list_data in data.items():
                    await self.redis.delete(key)  # تنظيف أولي
                    if list_data:
                        await self.redis.rpush(key, *list_data)
                        await self.redis.expire(key, SecurityConfig.HONEY_DB_TTL)
        
        return {
            "structures_created": len(redis_structures),
            "total_keys": sum(len(data) for data in redis_structures.values()),
            "storage_actual": "redis_implemented"
        }
    
    def _generate_access_credentials(self, honey_db_id: str) -> Dict[str, Any]:
        """إنشاء بيانات وصول وهمية"""
        creds = {
            "database_url": f"postgresql://honey_user_{secrets.token_hex(8)}:honey_pass_{secrets.token_hex(16)}@honeydb-cluster:5432/{honey_db_id}",
            "mongodb_uri": f"mongodb://honey_user:{secrets.token_urlsafe(24)}@honeymongo-cluster:27017/{honey_db_id}?authSource=admin",
            "redis_url": f"redis://:honey_redis_pass_{secrets.token_hex(12)}@honeyredis-cluster:6379/0",
            "api_key": f"hk_{honey_db_id}_{secrets.token_hex(16)}",
            "jwt_secret": secrets.token_urlsafe(32),
            "ssh_key_fingerprint": hashlib.md5(secrets.token_bytes(256)).hexdigest(),
            "access_token": f"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.{secrets.token_urlsafe(48)}.{secrets.token_hex(16)}"
        }
        
        return creds
    
    def _generate_db_features(self, db_type: str, complexity: str) -> List[str]:
        """توليد قائمة ميزات قاعدة البيانات"""
        base_features = ["fake_data", "honeypot", "attack_monitoring"]
        
        if complexity in ["high", "very_high"]:
            base_features.extend([
                "data_encryption",
                "audit_logging",
                "real_time_replication",
                "backup_automation",
                "query_analytics",
                "performance_monitoring"
            ])
        
        if db_type == "mongodb":
            base_features.extend(["document_storage", "aggregation_pipeline", "geospatial_indexes"])
        elif db_type in ["mysql", "postgresql"]:
            base_features.extend(["acid_compliant", "transaction_support", "foreign_keys"])
        
        if complexity == "very_high":
            base_features.extend([
                "machine_learning_anomalies",
                "blockchain_audit_trail",
                "quantum_resistant_encryption",
                "zero_trust_architecture"
            ])
        
        return base_features
    
    async def _store_attack_profile(self, attack_id: str, attacker_ip: str,
                                   fingerprint: Dict[str, Any], attack_type: str,
                                   honey_db_id: str):
        """تخزين ملف تعريف الهجوم"""
        attack_profile = {
            "attack_id": attack_id,
            "attacker_ip": attacker_ip,
            "fingerprint": fingerprint,
            "attack_type": attack_type,
            "honey_db_id": honey_db_id,
            "first_seen": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "total_attempts": 1,
            "honey_db_active": True
        }
        
        # تخزين في Redis
        profile_key = f"attack_profile:{attack_id}"
        await self.redis.setex(
            profile_key,
            SecurityConfig.HONEY_DB_TTL * 2,  # ضعف مدة قاعدة البيانات
            json.dumps(attack_profile)
        )
        
        # تخزين في الذاكرة
        self.attack_profiles[attack_id] = attack_profile
        
        # تحديث فهرس الهجمات حسب IP
        ip_index_key = f"attacks_by_ip:{attacker_ip}"
        await self.redis.sadd(ip_index_key, attack_id)
        await self.redis.expire(ip_index_key, SecurityConfig.HONEY_DB_TTL * 2)
    
    async def get_honey_database(self, honey_db_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على قاعدة بيانات وهمية"""
        # محاولة جلب من الذاكرة أولاً
        if honey_db_id in self.honey_databases:
            return self.honey_databases[honey_db_id]
        
        # محاولة جلب من Redis
        db_data = await self.redis.get(f"honey_db:{honey_db_id}")
        
        if db_data:
            honey_db = json.loads(db_data)
            self.honey_databases[honey_db_id] = honey_db
            return honey_db
        
        return None
    
    async def query_honey_database(self, honey_db_id: str, 
                                  query_type: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """استعلام قاعدة البيانات الوهمية"""
        honey_db = await self.get_honey_database(honey_db_id)
        
        if not honey_db:
            return {"error": "Honey database not found", "honey_db_id": honey_db_id}
        
        # محاكاة وقت الاستعلام
        query_delay = random.uniform(0.1, 2.0)
        await asyncio.sleep(min(query_delay, 0.5))  # حد أقصى 0.5 ثانية
        
        # توليد نتائج وهمية بناءً على نوع الاستعلام
        if query_type == "sql":
            results = self._simulate_sql_query(query_params, honey_db)
        elif query_type == "mongo":
            results = self._simulate_mongo_query(query_params, honey_db)
        elif query_type == "redis":
            results = self._simulate_redis_query(query_params, honey_db)
        else:
            results = {"error": "Unsupported query type", "query_type": query_type}
        
        # تسجيل الاستعلام
        await self._log_honey_query(honey_db_id, query_type, query_params, results)
        
        return {
            "honey_db_id": honey_db_id,
            "query_type": query_type,
            "query_params": query_params,
            "results": results,
            "execution_time_ms": query_delay * 1000,
            "timestamp": datetime.utcnow().isoformat(),
            "is_honey_response": True
        }
    
    def _simulate_sql_query(self, query_params: Dict[str, Any], 
                           honey_db: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة استعلام SQL"""
        table_name = query_params.get("table", "users")
        limit = query_params.get("limit", 10)
        
        # بيانات وهمية
        fake_rows = []
        columns = honey_db.get("schema", {}).get("tables", {}).get(table_name, {}).get("columns", [])
        
        for i in range(min(limit, 50)):
            row = {"row_number": i + 1}
            
            for col in columns:
                col_name = col["name"]
                row[col_name] = self._generate_fake_column_value(
                    col_name, col["type"], i + 1
                )
            
            fake_rows.append(row)
        
        return {
            "table": table_name,
            "rows": fake_rows,
            "total_rows": len(fake_rows),
            "affected_rows": len(fake_rows),
            "query_hash": hashlib.md5(json.dumps(query_params).encode()).hexdigest()
        }
    
    def _simulate_mongo_query(self, query_params: Dict[str, Any],
                             honey_db: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة استعلام MongoDB"""
        collection = query_params.get("collection", "users")
        limit = query_params.get("limit", 10)
        
        fake_documents = []
        
        for i in range(min(limit, 50)):
            doc = {
                "_id": f"honey_{collection}_{i+1}",
                "collection": collection,
                "document_id": i + 1,
                "fake_data": True,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "field1": secrets.token_hex(8),
                    "field2": random.randint(1, 1000),
                    "field3": random.choice(["active", "inactive", "pending"])
                }
            }
            
            fake_documents.append(doc)
        
        return {
            "collection": collection,
            "documents": fake_documents,
            "total_documents": len(fake_documents),
            "query_execution": "success"
        }
    
    def _simulate_redis_query(self, query_params: Dict[str, Any],
                             honey_db: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة استعلام Redis"""
        key_pattern = query_params.get("key_pattern", "honey_*")
        data_type = query_params.get("type", "string")
        
        fake_data = {
            "keys_found": random.randint(1, 20),
            "key_pattern": key_pattern,
            "data_type": data_type,
            "sample_data": {}
        }
        
        # عينة بيانات
        if data_type == "string":
            fake_data["sample_data"] = {
                f"{key_pattern}_1": secrets.token_hex(16),
                f"{key_pattern}_2": json.dumps({"fake": True, "timestamp": int(time.time())})
            }
        elif data_type == "hash":
            fake_data["sample_data"] = {
                f"{key_pattern}_hash": {
                    "field1": "value1",
                    "field2": "value2",
                    "timestamp": int(time.time())
                }
            }
        
        return fake_data
    
    async def _log_honey_query(self, honey_db_id: str, query_type: str,
                              query_params: Dict[str, Any], results: Dict[str, Any]):
        """تسجيل استعلام قاعدة البيانات الوهمية"""
        log_entry = {
            "honey_db_id": honey_db_id,
            "query_type": query_type,
            "query_params": query_params,
            "results_summary": {
                "items_returned": results.get("total_rows") or 
                                 results.get("total_documents") or 
                                 results.get("keys_found") or 0,
                "execution_success": "error" not in results
            },
            "timestamp": datetime.utcnow().isoformat(),
            "log_id": str(uuid.uuid4())
        }
        
        log_key = f"honey_query_log:{honey_db_id}:{int(time.time())}:{secrets.token_hex(4)}"
        await self.redis.setex(log_key, 86400, json.dumps(log_entry))
    
    async def cleanup_old_honey_databases(self):
        """تنظيف قواعد البيانات الوهمية القديمة"""
        current_time = time.time()
        expired_dbs = []
        
        for honey_db_id, honey_db in list(self.honey_databases.items()):
            valid_until_str = honey_db.get("valid_until")
            
            if valid_until_str:
                valid_until = datetime.fromisoformat(valid_until_str).timestamp()
                
                if valid_until < current_time:
                    expired_dbs.append(honey_db_id)
                    
                    # حذف من الذاكرة
                    del self.honey_databases[honey_db_id]
                    
                    # حذف من Redis
                    await self.redis.delete(f"honey_db:{honey_db_id}")
                    await self.redis.delete(f"honey_mongo:{honey_db_id}")
                    await self.redis.delete(f"honey_postgres:{honey_db_id}")
                    
                    # حذف مفاتيح Redis الوهمية
                    pattern = f"{honey_db_id}:*"
                    keys = await self.redis.keys(pattern)
                    if keys:
                        await self.redis.delete(*keys)
        
        if expired_dbs:
            await logger.info(
                "Cleaned up expired honey databases",
                count=len(expired_dbs),
                databases=expired_dbs
            )
        
        return expired_dbs

# ==================== ENHANCED THREAT INTELLIGENCE WITH NEW FEATURES ====================
class EnhancedThreatIntelligence:
    """
    ذكاء تهديدات محسّن مع الميزات الجديدة
    """
    
    def __init__(self, redis_client, postgres_pool, mongo_client):
        self.redis = redis_client
        self.postgres = postgres_pool
        self.mongo = mongo_client
        
        # الميزات الجديدة
        self.reaction_mutation = ReactionBasedMutation(redis_client)
        self.cascade_defense = CascadeReactionDefense(redis_client)
        self.honey_database = PerAttackDynamicHoneyDatabase(redis_client, postgres_pool, mongo_client)
        
        # التتبع
        self.attack_counter = defaultdict(int)
        self.active_attacks = {}
        
    async def detect_and_respond(self, request: Request, 
                                attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        اكتشاف الهجوم والرد بالميزات الجديدة
        """
        attack_id = str(uuid.uuid4())
        attack_type = attack_data.get("attack_type", "unknown")
        severity = attack_data.get("severity", "medium")
        
        # تحديث العداد
        self.attack_counter[attack_type] += 1
        
        # تسجيل الهجوم النشط
        self.active_attacks[attack_id] = {
            **attack_data,
            "attack_id": attack_id,
            "detected_at": datetime.utcnow(),
            "response_initiated": False,
            "responses": []
        }
        
        # تحليل الهجوم
        analysis = await self._analyze_attack(attack_data)
        
        # تحديد الردود المناسبة
        responses = []
        
        # 1. رد الطفرة التفاعلية (A)
        if severity in ["medium", "high", "critical"]:
            mutation_response = await self.reaction_mutation.trigger_mutation(
                attack_data, analysis.get("target_data", {})
            )
            responses.append({
                "type": "reaction_mutation",
                "response": mutation_response
            })
        
        # 2. رد الدفاع المتسلسل (B)
        if severity in ["high", "critical"]:
            cascade_response = await self.cascade_defense.trigger_cascade_reaction(
                attack_data
            )
            responses.append({
                "type": "cascade_defense",
                "response": cascade_response
            })
        
        # 3. قاعدة البيانات الوهمية الديناميكية (C)
        honey_db_response = await self.honey_database.create_honey_database(
            {**attack_data, "attack_id": attack_id}
        )
        responses.append({
            "type": "honey_database",
            "response": honey_db_response
        })
        
        # تحديث الهجوم النشط
        self.active_attacks[attack_id]["response_initiated"] = True
        self.active_attacks[attack_id]["responses"] = responses
        
        # تسجيل الاستجابة
        await self._log_attack_response(attack_id, attack_data, responses)
        
        return {
            "attack_id": attack_id,
            "attack_type": attack_type,
            "severity": severity,
            "detection_time": datetime.utcnow().isoformat(),
            "analysis": analysis,
            "responses": responses,
            "total_responses": len(responses),
            "system_status": "defense_activated"
        }
    
    async def _analyze_attack(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل الهجوم"""
        analysis = {
            "attack_pattern": attack_data.get("attack_type", "unknown"),
            "severity": attack_data.get("severity", "medium"),
            "attacker_info": {
                "ip": attack_data.get("attacker_ip", "unknown"),
                "user_agent": attack_data.get("user_agent", ""),
                "fingerprint": attack_data.get("fingerprint", {})
            },
            "target_info": {
                "endpoint": attack_data.get("endpoint", ""),
                "method": attack_data.get("method", "GET"),
                "parameters": attack_data.get("parameters", {})
            },
            "timing": {
                "detected_at": datetime.utcnow().isoformat(),
                "frequency": self.attack_counter[attack_data.get("attack_type", "unknown")]
            },
            "recommended_responses": self._recommend_responses(attack_data)
        }
        
        # تحليل متقدم
        if attack_data.get("attack_type") == "sql_injection":
            analysis["sql_pattern"] = attack_data.get("sql_pattern", "")
            analysis["injection_points"] = self._find_injection_points(
                attack_data.get("parameters", {})
            )
        
        elif attack_data.get("attack_type") == "brute_force":
            analysis["attempt_count"] = attack_data.get("attempt_count", 1)
            analysis["target_accounts"] = attack_data.get("target_accounts", [])
        
        return analysis
    
    def _find_injection_points(self, parameters: Dict[str, Any]) -> List[str]:
        """إيجاد نقاط الحقن في المعاملات"""
        injection_points = []
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # البحث عن أنماط SQL خطيرة
                sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", 
                              "UNION", "OR 1=1", "--", ";"]
                
                if any(keyword in value.upper() for keyword in sql_keywords):
                    injection_points.append({
                        "parameter": key,
                        "value_preview": value[:100] + "..." if len(value) > 100 else value,
                        "risk_level": "high"
                    })
        
        return injection_points
    
    def _recommend_responses(self, attack_data: Dict[str, Any]) -> List[str]:
        """توصية بالردود المناسبة"""
        severity = attack_data.get("severity", "medium")
        attack_type = attack_data.get("attack_type", "unknown")
        
        recommendations = []
        
        # توصيات أساسية
        recommendations.append("log_attack")
        recommendations.append("notify_security_team")
        
        # توصيات حسب الشدة
        if severity in ["medium", "high", "critical"]:
            recommendations.append("reaction_based_mutation")
            recommendations.append("dynamic_honey_database")
        
        if severity in ["high", "critical"]:
            recommendations.append("cascade_reaction_defense")
            recommendations.append("ip_blocking")
            recommendations.append("session_invalidation")
        
        # توصيات حسب نوع الهجوم
        if attack_type == "sql_injection":
            recommendations.append("sql_sanitization")
            recommendations.append("parameterized_queries")
        
        elif attack_type == "brute_force":
            recommendations.append("account_lockout")
            recommendations.append("rate_limiting")
            recommendations.append("captcha_verification")
        
        elif attack_type == "xss_attempt":
            recommendations.append("input_sanitization")
            recommendations.append("csp_headers")
            recommendations.append("xss_filtering")
        
        return recommendations
    
    async def _log_attack_response(self, attack_id: str, 
                                  attack_data: Dict[str, Any],
                                  responses: List[Dict[str, Any]]):
        """تسجيل استجابة الهجوم"""
        log_entry = {
            "attack_id": attack_id,
            "attack_data": attack_data,
            "responses": responses,
            "logged_at": datetime.utcnow().isoformat(),
            "system_version": "enhanced_defense_v2"
        }
        
        log_key = f"attack_response_log:{attack_id}"
        await self.redis.setex(log_key, 604800, json.dumps(log_entry))  # أسبوع
        
        # أيضاً تسجيل في سجلات النظام
        await logger.warning(
            "Attack detected and responses deployed",
            attack_id=attack_id,
            attack_type=attack_data.get("attack_type"),
            severity=attack_data.get("severity"),
            total_responses=len(responses),
            response_types=[r["type"] for r in responses]
        )
    
    async def get_attack_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الهجمات"""
        stats = {
            "total_attacks": sum(self.attack_counter.values()),
            "by_type": dict(self.attack_counter),
            "active_attacks": len(self.active_attacks),
            "attack_timeline": await self._get_attack_timeline(),
            "response_statistics": await self._get_response_stats()
        }
        
        return stats
    
    async def _get_attack_timeline(self) -> List[Dict[str, Any]]:
        """الحصول على خط زمني للهجمات"""
        timeline = []
        
        for attack_id, attack_data in list(self.active_attacks.items())[-100:]:  # آخر 100 هجوم
            timeline.append({
                "attack_id": attack_id,
                "type": attack_data.get("attack_type"),
                "severity": attack_data.get("severity"),
                "detected_at": attack_data.get("detected_at").isoformat() 
                               if hasattr(attack_data.get("detected_at"), 'isoformat') 
                               else str(attack_data.get("detected_at")),
                "responses_count": len(attack_data.get("responses", []))
            })
        
        return timeline
    
    async def _get_response_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الردود"""
        response_stats = {
            "reaction_mutation": {
                "triggered": 0,
                "successful": 0
            },
            "cascade_defense": {
                "triggered": 0,
                "total_reactions": 0
            },
            "honey_database": {
                "created": 0,
                "active": 0,
                "queries_served": 0
            }
        }
        
        # جلب الإحصائيات من Redis
        mutation_stats = await self.redis.get("stats:reaction_mutation")
        if mutation_stats:
            response_stats["reaction_mutation"] = json.loads(mutation_stats)
        
        cascade_stats = await self.redis.get("stats:cascade_defense")
        if cascade_stats:
            response_stats["cascade_defense"] = json.loads(cascade_stats)
        
        honey_stats = await self.redis.get("stats:honey_database")
        if honey_stats:
            response_stats["honey_database"] = json.loads(honey_stats)
        
        return response_stats

# ==================== REST OF THE SYSTEM (UNCHANGED) ====================
# [أبقيت باقي النظام كما هو مع تعديل بسيط لدعم الميزات الجديدة]

# ==================== UPDATED LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Redis Cluster
        app.state.redis = redis_cluster.RedisCluster(
            startup_nodes=SecurityConfig.REDIS_CLUSTER_NODES,
            decode_responses=True,
            skip_full_coverage_check=True,
            max_connections=100,
            socket_keepalive=True
        )
    except:
        app.state.redis = await redis.Redis(
            host='localhost', port=6379, decode_responses=True,
            max_connections=50
        )
    
    # Databases
    app.state.postgres_pool = await asyncpg.create_pool(
        SecurityConfig.POSTGRES_URL,
        min_size=10,
        max_size=50
    )
    
    app.state.mongo_client = AsyncIOMotorClient(SecurityConfig.MONGO_URL)
    app.state.mongo_db = app.state.mongo_client.secure_chat
    
    app.state.es_client = AsyncElasticsearch(
        SecurityConfig.ELASTICSEARCH_URLS,
        max_retries=3,
        retry_on_timeout=True
    )
    
    # Kafka
    app.state.kafka_producer = AIOKafkaProducer(
        bootstrap_servers=SecurityConfig.KAFKA_BOOTSTRAP_SERVERS,
        compression_type="gzip",
        max_batch_size=32768
    )
    await app.state.kafka_producer.start()
    
    # Enhanced Security Services
    app.state.encryption = EnhancedEncryptionService()
    app.state.threat_intel = EnhancedThreatIntelligence(
        app.state.redis,
        app.state.postgres_pool,
        app.state.mongo_db
    )
    
    app.state.chat_security = ChatSecurityManager(
        app.state.redis,
        app.state.threat_intel
    )
    
    app.state.websocket_manager = SecureWebSocketManager(
        app.state.chat_security
    )
    
    # Analysis Engines
    app.state.content_analyzer = ContentAnalyzer()
    app.state.threat_detector = AdvancedThreatDetector()
    app.state.anomaly_detector = AnomalyDetectionEngine()
    app.state.network_analyzer = NetworkBehaviorAnalyzer()
    app.state.e2e_encryption = EndToEndEncryption()
    
    # Distributed Services
    app.state.rate_limiter = DistributedRateLimiter(app.state.redis)
    app.state.session_manager = DistributedSessionManager(app.state.redis)
    app.state.cache_manager = DistributedCacheManager(app.state.redis)
    
    # Background cleanup task for honey databases
    async def honey_db_cleanup():
        while True:
            try:
                await asyncio.sleep(3600)  # كل ساعة
                if hasattr(app.state, 'threat_intel'):
                    await app.state.threat_intel.honey_database.cleanup_old_honey_databases()
            except Exception as e:
                await logger.error(f"Honey DB cleanup failed: {str(e)}")
    
    app.state.honey_cleanup_task = asyncio.create_task(honey_db_cleanup())
    
    # Start background tasks
    app.state.background_tasks = set()
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'honey_cleanup_task'):
        app.state.honey_cleanup_task.cancel()
    
    await app.state.redis.close()
    await app.state.postgres_pool.close()
    app.state.mongo_client.close()
    await app.state.es_client.close()
    await app.state.kafka_producer.stop()

# ==================== NEW API ENDPOINTS FOR ENHANCED DEFENSE ====================
@app.post("/api/security/attack-response")
async def trigger_attack_response(
    request: Request,
    attack_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    تشغيل استجابات الدفاع المحسنة عند اكتشاف هجوم
    """
    if current_user.get('role') not in ['admin', 'security_team']:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    response = await app.state.threat_intel.detect_and_respond(request, attack_data)
    
    return {
        "status": "defense_activated",
        "attack_response": response,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/security/cascade-status/{cascade_id}")
async def get_cascade_status(
    cascade_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    الحصول على حالة التفاعل المتسلسل
    """
    if current_user.get('role') not in ['admin', 'security_team']:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    status = await app.state.threat_intel.cascade_defense.get_cascade_status(cascade_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Cascade not found")
    
    return {
        "cascade_id": cascade_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/security/honey-db/query")
async def query_honey_database(
    query_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    استعلام قاعدة البيانات الوهمية
    """
    if current_user.get('role') not in ['admin', 'security_team']:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    honey_db_id = query_request.get("honey_db_id")
    query_type = query_request.get("query_type", "sql")
    query_params = query_request.get("query_params", {})
    
    if not honey_db_id:
        raise HTTPException(status_code=400, detail="Missing honey_db_id")
    
    results = await app.state.threat_intel.honey_database.query_honey_database(
        honey_db_id, query_type, query_params
    )
    
    return {
        "honey_query": results,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/security/attack-statistics")
async def get_attack_statistics(
    current_user: dict = Depends(get_current_user)
):
    """
    الحصول على إحصائيات الهجمات والردود
    """
    if current_user.get('role') not in ['admin', 'security_team']:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    stats = await app.state.threat_intel.get_attack_statistics()
    
    return {
        "statistics": stats,
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== ENHANCED SECURITY MIDDLEWARE ====================
@app.middleware("http")
async def enhanced_security_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # اكتشاف هجمات محتملة
    attack_detected = await _detect_potential_attack(request)
    
    if attack_detected:
        # تسجيل الهجوم
        attack_data = {
            "attack_id": str(uuid.uuid4()),
            "attack_type": attack_detected["type"],
            "severity": attack_detected["severity"],
            "attacker_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "endpoint": str(request.url),
            "method": request.method,
            "parameters": dict(request.query_params),
            "detection_time_ms": (time.time() - start_time) * 1000,
            "fingerprint": {
                "headers": {k: request.headers.get(k) for k in SecurityConfig.FINGERPRINT_HEADERS 
                           if request.headers.get(k)},
                "cookies": request.cookies
            }
        }
        
        # تشغيل استجابات الدفاع في الخلفية
        if hasattr(request.app.state, 'threat_intel'):
            asyncio.create_task(
                request.app.state.threat_intel.detect_and_respond(request, attack_data)
            )
        
        # استمرار معالجة الطلب مع إضافة علامة الهجوم
        request.state.attack_detected = attack_data
    
    # رفض الطلبات الكبيرة جداً
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10_485_760:
            return JSONResponse(
                status_code=413,
                content={"detail": "Payload too large"}
            )
    
    try:
        response = await call_next(request)
        
        # رؤوس الأمان
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; connect-src 'self' wss:; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Request-ID": request_id,
            "X-Robots-Tag": "noindex, nofollow",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        }
        
        for key, value in security_headers.items():
            response.headers[key] = value
        
        # تسجيل
        response_time = (time.time() - start_time) * 1000
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "ip": request.client.host,
            "status_code": response.status_code,
            "response_time": f"{response_time:.2f}ms",
            "user_agent": request.headers.get("user-agent", "")
        }
        
        if hasattr(request.state, 'attack_detected'):
            log_data["attack_detected"] = request.state.attack_detected.get("attack_type")
            log_data["attack_id"] = request.state.attack_detected.get("attack_id")
        
        await logger.info("Request processed", **log_data)
        
        return response
        
    except Exception as e:
        await logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            ip=request.client.host,
            error=str(e)
        )
        raise

async def _detect_potential_attack(request: Request) -> Optional[Dict[str, Any]]:
    """اكتشاف هجوم محتمل"""
    
    # اكتشاف حقن SQL
    sql_patterns = SecurityConfig.SQL_KEYWORDS
    request_params = dict(request.query_params)
    
    for param, value in request_params.items():
        if isinstance(value, str):
            value_upper = value.upper()
            if any(pattern in value_upper for pattern in sql_patterns):
                return {
                    "type": "sql_injection",
                    "severity": "high",
                    "parameter": param,
                    "pattern_found": next(p for p in sql_patterns if p in value_upper)
                }
    
    # اكتشاف XSS
    xss_patterns = SecurityConfig.XSS_PATTERNS
    for param, value in request_params.items():
        if isinstance(value, str):
            if any(pattern in value.lower() for pattern in xss_patterns):
                return {
                    "type": "xss_attempt",
                    "severity": "medium",
                    "parameter": param,
                    "pattern_found": next(p for p in xss_patterns if p in value.lower())
                }
    
    # اكتشاف محاولات وصول غير مصرح
    sensitive_paths = ["/admin", "/api/admin", "/internal", "/debug", "/phpmyadmin", "/wp-admin"]
    if any(request.url.path.startswith(path) for path in sensitive_paths):
        # التحقق من الصلاحيات (سيتم في طبقة المصادقة)
        return {
            "type": "unauthorized_access",
            "severity": "medium",
            "sensitive_path": request.url.path
        }
    
    # اكتشاف محاولات المسح
    scanning_patterns = ["' OR '1'='1", "../", "./", "union select", "information_schema"]
    for param, value in request_params.items():
        if isinstance(value, str):
            value_lower = value.lower()
            if any(pattern in value_lower for pattern in scanning_patterns):
                return {
                    "type": "scanning_attempt",
                    "severity": "low",
                    "parameter": param,
                    "pattern_found": next(p for p in scanning_patterns if p in value_lower)
                }
    
    return None

# ==================== REST OF THE SYSTEM REMAINS UNCHANGED ====================
# [تم الحفاظ على بقية نقاط النهاية والوظائف كما هي]

# ==================== STARTUP ====================
if __name__ == "__main__":
    import uvicorn
    
    ssl_context = None
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        ssl_context = ("cert.pem", "key.pem")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=ssl_context[1] if ssl_context else None,
        ssl_certfile=ssl_context[0] if ssl_context else None,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=10000,
        limit_max_requests=100000,
        workers=4
    )