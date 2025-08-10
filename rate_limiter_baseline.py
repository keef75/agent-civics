"""
#BASELINE
Rate Limiter Implementation using Token Bucket Algorithm
Supports synchronous and asynchronous patterns with thread safety.
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RateLimiterMetrics:
    """Observability metrics for rate limiter"""
    requests_allowed: int = 0
    requests_denied: int = 0
    current_tokens: float = 0
    last_refill_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'requests_allowed': self.requests_allowed,
            'requests_denied': self.requests_denied,
            'current_tokens': self.current_tokens,
            'success_rate': self.requests_allowed / (self.requests_allowed + self.requests_denied) if (self.requests_allowed + self.requests_denied) > 0 else 0,
            'last_refill_time': self.last_refill_time
        }


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter with async/await support.
    Designed to handle 10,000+ requests per second with backpressure.
    """
    
    def __init__(self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("Refill rate must be positive")
            
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = RateLimiterMetrics()
        self.metrics.current_tokens = self.tokens
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            self.metrics.current_tokens = self.tokens
            self.metrics.last_refill_time = now
    
    def allow(self, tokens: int = 1) -> bool:
        """
        Synchronous rate limiting check.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")
        if tokens > self.capacity:
            raise ValueError("Requested tokens exceed capacity")
            
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.metrics.requests_allowed += 1
                self.metrics.current_tokens = self.tokens
                return True
            else:
                self.metrics.requests_denied += 1
                return False
    
    async def allow_async(self, tokens: int = 1) -> bool:
        """
        Asynchronous rate limiting check.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")
        if tokens > self.capacity:
            raise ValueError("Requested tokens exceed capacity")
            
        async with self._async_lock:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.metrics.requests_allowed += 1
                self.metrics.current_tokens = self.tokens
                return True
            else:
                self.metrics.requests_denied += 1
                return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate time to wait until request would be allowed.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait, 0 if request would be allowed now
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")
        if tokens > self.capacity:
            raise ValueError("Requested tokens exceed capacity")
            
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate
    
    async def wait_for_tokens(self, tokens: int = 1, max_wait: Optional[float] = None) -> bool:
        """
        Backpressure handling - wait until tokens are available.
        
        Args:
            tokens: Number of tokens to wait for
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if tokens became available, False if timed out
        """
        wait_time = self.wait_time(tokens)
        
        if max_wait is not None and wait_time > max_wait:
            return False
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return await self.allow_async(tokens)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self._lock:
            self._refill_tokens()
            return self.metrics.to_dict()
    
    def reset(self) -> None:
        """Reset rate limiter to initial state"""
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()
            self.metrics = RateLimiterMetrics()
            self.metrics.current_tokens = self.tokens


class MultiKeyRateLimiter:
    """Rate limiter supporting multiple keys (e.g., per-user, per-IP)"""
    
    def __init__(self, capacity: int, refill_rate: float, cleanup_interval: float = 300):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.cleanup_interval = cleanup_interval
        
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
    
    def _cleanup_inactive(self) -> None:
        """Clean up inactive rate limiters to prevent memory leaks"""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return
            
        inactive_keys = []
        for key, limiter in self._limiters.items():
            if now - limiter.last_refill > self.cleanup_interval and limiter.tokens == self.capacity:
                inactive_keys.append(key)
        
        for key in inactive_keys:
            del self._limiters[key]
        
        self._last_cleanup = now
    
    def get_limiter(self, key: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for specific key"""
        with self._lock:
            self._cleanup_inactive()
            
            if key not in self._limiters:
                self._limiters[key] = TokenBucketRateLimiter(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate
                )
            
            return self._limiters[key]
    
    def allow(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed for specific key"""
        return self.get_limiter(key).allow(tokens)
    
    async def allow_async(self, key: str, tokens: int = 1) -> bool:
        """Async check if request is allowed for specific key"""
        return await self.get_limiter(key).allow_async(tokens)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all keys"""
        with self._lock:
            return {key: limiter.get_metrics() for key, limiter in self._limiters.items()}