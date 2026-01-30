"""
Distributed cache invalidation via Redis pub/sub.

Provides:
- EpochManager: Tracks and broadcasts policy/ACL epoch changes
- Cache invalidation across multiple nodes
- Automatic subscription to epoch change channels

If Redis is not available, falls back to local-only epochs.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Try to import redis
try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None  # type: ignore
    HAS_REDIS = False


@dataclass
class EpochState:
    """Current epoch state for all epoch types."""
    policy: int = 1
    entitlement: int = 1
    acl: int = 1
    index: int = 1
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "policy": self.policy,
            "entitlement": self.entitlement,
            "acl": self.acl,
            "index": self.index,
        }


class EpochManager:
    """
    Manages cache invalidation epochs with optional Redis pub/sub.
    
    Epochs are version numbers that increment when policies, entitlements,
    ACLs, or indexes change. Cache entries include epochs in their keys,
    so bumping an epoch effectively invalidates all related entries.
    """
    
    CHANNEL = "abac_rag:epochs"
    
    def __init__(self, redis_url: Optional[str] = None):
        self._state = EpochState()
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[str, int], None]] = []
        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Load initial epochs from environment
        self._load_from_env()
        
        # Try to connect to Redis for distributed invalidation
        if redis_url or os.environ.get("CACHE_REDIS_URL"):
            self._connect_redis(redis_url or os.environ.get("CACHE_REDIS_URL"))
    
    def _load_from_env(self) -> None:
        """Load initial epochs from environment variables."""
        self._state.policy = int(os.environ.get("CACHE_POLICY_EPOCH", "1"))
        self._state.entitlement = int(os.environ.get("CACHE_ENTITLEMENT_EPOCH", "1"))
        self._state.acl = int(os.environ.get("CACHE_ACL_EPOCH", "1"))
        self._state.index = int(os.environ.get("CACHE_INDEX_EPOCH", "1"))
    
    def _connect_redis(self, redis_url: str) -> None:
        """Connect to Redis and start listening for epoch changes."""
        if not HAS_REDIS:
            return
        
        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()  # Test connection
            
            # Load current epochs from Redis
            self._sync_from_redis()
            
            # Start pub/sub listener
            self._start_listener()
        except Exception:
            self._redis = None
    
    def _sync_from_redis(self) -> None:
        """Sync epoch state from Redis."""
        if not self._redis:
            return
        
        try:
            stored = self._redis.get("abac_rag:epoch_state")
            if stored:
                data = json.loads(stored)
                with self._lock:
                    self._state.policy = max(self._state.policy, data.get("policy", 1))
                    self._state.entitlement = max(self._state.entitlement, data.get("entitlement", 1))
                    self._state.acl = max(self._state.acl, data.get("acl", 1))
                    self._state.index = max(self._state.index, data.get("index", 1))
        except Exception:
            pass
    
    def _save_to_redis(self) -> None:
        """Save current epoch state to Redis."""
        if not self._redis:
            return
        
        try:
            self._redis.set("abac_rag:epoch_state", json.dumps(self._state.to_dict()))
        except Exception:
            pass
    
    def _start_listener(self) -> None:
        """Start background thread to listen for epoch changes."""
        if not self._redis or self._running:
            return
        
        self._pubsub = self._redis.pubsub()
        self._pubsub.subscribe(self.CHANNEL)
        self._running = True
        
        def listen():
            while self._running:
                try:
                    message = self._pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        self._handle_message(message["data"])
                except Exception:
                    time.sleep(1)  # Back off on error
        
        self._listener_thread = threading.Thread(target=listen, daemon=True)
        self._listener_thread.start()
    
    def _handle_message(self, data: str) -> None:
        """Handle incoming epoch change message."""
        try:
            msg = json.loads(data)
            epoch_type = msg.get("type")
            new_value = msg.get("value")

            if epoch_type and new_value:
                with self._lock:
                    current = getattr(self._state, epoch_type, None)
                    if current is not None and new_value > current:
                        setattr(self._state, epoch_type, new_value)
                        # Notify local subscribers
                        for sub in self._subscribers:
                            try:
                                sub(epoch_type, new_value)
                            except Exception:
                                pass
        except Exception:
            pass

    def bump_epoch(self, epoch_type: str) -> int:
        """
        Increment an epoch and broadcast the change.

        Args:
            epoch_type: One of 'policy', 'entitlement', 'acl', 'index'

        Returns:
            The new epoch value
        """
        with self._lock:
            current = getattr(self._state, epoch_type, None)
            if current is None:
                raise ValueError(f"Unknown epoch type: {epoch_type}")

            new_value = current + 1
            setattr(self._state, epoch_type, new_value)

            # Broadcast to other nodes
            if self._redis:
                try:
                    msg = json.dumps({"type": epoch_type, "value": new_value})
                    self._redis.publish(self.CHANNEL, msg)
                    self._save_to_redis()
                except Exception:
                    pass

            # Notify local subscribers
            for sub in self._subscribers:
                try:
                    sub(epoch_type, new_value)
                except Exception:
                    pass

            return new_value

    def get_epochs(self) -> Dict[str, int]:
        """Get current epoch values."""
        with self._lock:
            return self._state.to_dict()

    def get_epoch(self, epoch_type: str) -> int:
        """Get a specific epoch value."""
        with self._lock:
            return getattr(self._state, epoch_type, 1)

    def subscribe(self, callback: Callable[[str, int], None]) -> None:
        """Subscribe to epoch change notifications."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, int], None]) -> None:
        """Unsubscribe from epoch change notifications."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def stop(self) -> None:
        """Stop the listener thread."""
        self._running = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception:
                pass
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)

    @property
    def is_distributed(self) -> bool:
        """Check if distributed invalidation is enabled."""
        return self._redis is not None


# Global epoch manager
_epoch_manager: Optional[EpochManager] = None


def get_epoch_manager() -> EpochManager:
    """Get the global epoch manager."""
    global _epoch_manager
    if _epoch_manager is None:
        redis_url = os.environ.get("CACHE_REDIS_URL") if os.environ.get("CACHE_BACKEND") == "redis" else None
        _epoch_manager = EpochManager(redis_url)
    return _epoch_manager


def reset_epoch_manager() -> None:
    """Reset the global epoch manager. Useful for testing."""
    global _epoch_manager
    if _epoch_manager:
        _epoch_manager.stop()
    _epoch_manager = None

