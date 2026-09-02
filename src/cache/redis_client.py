"""
src/cache/redis_client.py
──────────────────────────────────────────────────────────────────────────────
Redis Feature Store & Nearline GNN Risk Score Cache Client.

Production System Design Role:
  1. Nearline GNN Score Caching: Nearline/offline GNN batch pipeline writes
     node risk scores to Redis (TTL 24h). FastAPI fetches this in <1ms to
     bypass expensive real-time graph message passing (>50ms).
  2. Account Feature Store: Pre-computed tabular and graph topological features
     stored as JSON/hashes keyed by account_id.
  3. Sliding-Window Transaction Velocity: Redis Sorted Sets (ZADD) for tracking
     real-time 24-hour transaction frequency and volume spikes.
  4. Token-Bucket / Sliding Window Rate Limiter: API traffic protection.
  5. Fallback Mechanism: If Redis is unavailable, degrades gracefully to a local
     in-memory cache without interrupting service.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple, List
from loguru import logger

try:
    import redis
    REDIS_INSTALLED = True
except ImportError:
    REDIS_INSTALLED = False
    redis = None  # type: ignore


import socket


class RedisFeatureStore:
    """
    High-performance Redis client for Feature Store & GNN Score Caching.
    Includes in-memory fallback for local testing when Redis is offline.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 0.2,
        enable_fallback: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.enable_fallback = enable_fallback
        self._redis_client: Optional[redis.Redis] = None
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expire_at)
        self.is_connected = False

        # Fast socket probe before attempting redis connect
        port_open = False
        try:
            with socket.create_connection((host, port), timeout=0.15):
                port_open = True
        except (socket.timeout, ConnectionRefusedError, OSError):
            port_open = False

        if REDIS_INSTALLED and port_open:
            try:
                client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    socket_timeout=socket_timeout,
                    decode_responses=True,
                )
                client.ping()
                self._redis_client = client
                self.is_connected = True
                logger.info(f"Connected to Redis feature store at {host}:{port}/{db}")
            except Exception as err:
                logger.warning(
                    f"Redis connection failed ({err}). Operating in fallback in-memory mode."
                )
                self.is_connected = False
        else:
            logger.warning("redis-py package not installed. Using in-memory cache fallback.")

    def ping(self) -> bool:
        """Check if Redis connection is active."""
        if self._redis_client and self.is_connected:
            try:
                return bool(self._redis_client.ping())
            except Exception:
                self.is_connected = False
                return False
        return False

    # ── Nearline GNN Score Cache ───────────────────────────────────────────────

    def get_gnn_score(self, account_id: str) -> Optional[float]:
        """Fetch pre-computed nearline GNN score for account (sub-1ms SLA)."""
        key = f"gnn:score:{account_id}"
        if self._redis_client and self.is_connected:
            try:
                val = self._redis_client.get(key)
                if val is not None:
                    return float(val)
            except Exception as e:
                logger.debug(f"Redis get_gnn_score error: {e}")

        # Fallback in-memory check
        if key in self._memory_cache:
            val, expire_at = self._memory_cache[key]
            if expire_at == 0 or time.time() < expire_at:
                return float(val)
            else:
                del self._memory_cache[key]
        return None

    def set_gnn_score(self, account_id: str, score: float, ttl_seconds: int = 86400) -> bool:
        """Store pre-computed GNN risk score for account (default TTL = 24h)."""
        key = f"gnn:score:{account_id}"
        if self._redis_client and self.is_connected:
            try:
                self._redis_client.setex(key, ttl_seconds, str(score))
                return True
            except Exception as e:
                logger.debug(f"Redis set_gnn_score error: {e}")

        # Fallback memory store
        expire_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0
        self._memory_cache[key] = (score, expire_at)
        return True

    def seed_gnn_scores(self, scores_dict: Dict[str, float], ttl_seconds: int = 86400) -> int:
        """Pipeline / Bulk seed GNN risk scores (simulating batch job output)."""
        count = 0
        if self._redis_client and self.is_connected:
            try:
                pipe = self._redis_client.pipeline()
                for acc_id, score in scores_dict.items():
                    key = f"gnn:score:{acc_id}"
                    pipe.setex(key, ttl_seconds, str(score))
                pipe.execute()
                return len(scores_dict)
            except Exception as e:
                logger.warning(f"Pipeline seed failed: {e}. Seeding sequentially.")

        for acc_id, score in scores_dict.items():
            if self.set_gnn_score(acc_id, score, ttl_seconds):
                count += 1
        return count

    # ── Feature Store Read / Write ─────────────────────────────────────────────

    def get_account_features(self, account_id: str) -> Optional[Dict[str, float]]:
        """Fetch full feature vector for account from feature store."""
        key = f"features:{account_id}"
        if self._redis_client and self.is_connected:
            try:
                raw = self._redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.debug(f"Redis get_account_features error: {e}")

        if key in self._memory_cache:
            val, expire_at = self._memory_cache[key]
            if expire_at == 0 or time.time() < expire_at:
                return val
            else:
                del self._memory_cache[key]
        return None

    def set_account_features(
        self, account_id: str, features: Dict[str, float], ttl_seconds: int = 86400
    ) -> bool:
        """Save account feature vector into feature store."""
        key = f"features:{account_id}"
        if self._redis_client and self.is_connected:
            try:
                self._redis_client.setex(key, ttl_seconds, json.dumps(features))
                return True
            except Exception as e:
                logger.debug(f"Redis set_account_features error: {e}")

        expire_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0
        self._memory_cache[key] = (features, expire_at)
        return True

    # ── Sliding-Window Velocity Tracker ────────────────────────────────────────

    def record_transaction(self, account_id: str, amount: float, timestamp: Optional[float] = None) -> bool:
        """Record a transaction event in Redis Sorted Set for real-time velocity."""
        ts = timestamp or time.time()
        key = f"tx_window:{account_id}"
        member = f"{ts}:{amount}"

        if self._redis_client and self.is_connected:
            try:
                # ZADD score=timestamp member=ts:amount
                self._redis_client.zadd(key, {member: ts})
                # Remove transactions older than 24 hours (86400 seconds)
                cutoff = ts - 86400
                self._redis_client.zremrangebyscore(key, "-inf", cutoff)
                self._redis_client.expire(key, 86400)
                return True
            except Exception as e:
                logger.debug(f"Redis record_transaction error: {e}")
        return False

    def get_velocity_24h(self, account_id: str, current_time: Optional[float] = None) -> Dict[str, float]:
        """Compute 24-hour transaction count and total volume."""
        now = current_time or time.time()
        cutoff = now - 86400
        key = f"tx_window:{account_id}"

        if self._redis_client and self.is_connected:
            try:
                members = self._redis_client.zrangebyscore(key, cutoff, "+inf")
                tx_count = float(len(members))
                tx_amount = 0.0
                for m in members:
                    parts = str(m).split(":")
                    if len(parts) >= 2:
                        tx_amount += float(parts[1])
                return {"tx_velocity_24h": tx_count, "amount_velocity_24h": tx_amount}
            except Exception as e:
                logger.debug(f"Redis get_velocity_24h error: {e}")

        return {"tx_velocity_24h": 0.0, "amount_velocity_24h": 0.0}

    # ── Rate Limiter ──────────────────────────────────────────────────────────

    def is_rate_limited(self, identifier: str, max_requests: int = 100, window_sec: int = 60) -> bool:
        """Sliding-window rate limiter using Redis keys."""
        key = f"ratelimit:{identifier}"
        now = time.time()
        if self._redis_client and self.is_connected:
            try:
                pipe = self._redis_client.pipeline()
                pipe.zremrangebyscore(key, "-inf", now - window_sec)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window_sec)
                results = pipe.execute()
                req_count = results[2]
                return req_count > max_requests
            except Exception as e:
                logger.debug(f"Redis rate limiter error: {e}")

        # Fallback memory check
        if key not in self._memory_cache:
            self._memory_cache[key] = ([], now + window_sec)
        timestamps, _ = self._memory_cache[key]
        recent = [t for t in timestamps if t > now - window_sec]
        recent.append(now)
        self._memory_cache[key] = (recent, now + window_sec)
        return len(recent) > max_requests

    def clear(self) -> None:
        """Clear memory cache and flush Redis test DB if connected."""
        self._memory_cache.clear()
        if self._redis_client and self.is_connected:
            try:
                self._redis_client.flushdb()
            except Exception:
                pass
