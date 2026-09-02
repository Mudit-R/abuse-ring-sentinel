"""
src/cache/__init__.py
──────────────────────────────────────────────────────────────────────────────
Redis feature store & GNN nearline caching layer.
"""

from src.cache.redis_client import RedisFeatureStore

__all__ = ["RedisFeatureStore"]
