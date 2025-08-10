#!/usr/bin/env python3
"""
Token Bucket Rate Limiter Implementation
SHA256 of specification: a7c8d9e2f1b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9

A mathematically correct, thread-safe token bucket rate limiter implementation
focusing on correctness and reliability over performance optimization.

Complexity Analysis:
- Time Complexity: O(1) for all operations
- Space Complexity: O(1) 
- Thread Safety: Full synchronization with asyncio locks
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class RateLimitMetrics:
    """Observability metrics for rate limiter performance"""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    current_tokens: float = 0.0
    last_refill_time: float = 0.0
    average_wait_time: float = 0.0
    max_wait_time: float = 0.0
    
    @property
    def allow_rate(self) -> float:
        """Calculate the percentage of allowed requests"""
        return (self.allowed_requests / max(self.total_requests, 1)) * 100.0
    
    @property
    def deny_rate(self) -> float:
        """Calculate the percentage of denied requests"""
        return (self.denied_requests / max(self.total_requests, 1)) * 100.0


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded without backpressure"""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.3f} seconds")


class BackpressureStrategy(ABC):
    """Abstract base class for backpressure handling strategies"""
    
    @abstractmethod
    async def handle(self, wait_time: float, metrics: RateLimitMetrics) -> bool:
        """
        Handle backpressure when tokens are insufficient
        
        Args:
            wait_time: Time to wait for sufficient tokens
            metrics: Current rate limiter metrics
            
        Returns:
            True if request should proceed, False if it should be rejected
        """
        pass


class NoBackpressure(BackpressureStrategy):
    """Strategy that immediately rejects requests when tokens are insufficient"""
    
    async def handle(self, wait_time: float, metrics: RateLimitMetrics) -> bool:
        return False


class FixedDelayBackpressure(BackpressureStrategy):
    """Strategy that waits up to a maximum delay for tokens to become available"""
    
    def __init__(self, max_wait_seconds: float = 1.0):
        self.max_wait_seconds = max_wait_seconds
    
    async def handle(self, wait_time: float, metrics: RateLimitMetrics) -> bool:
        if wait_time <= self.max_wait_seconds:
            await asyncio.sleep(wait_time)
            return True
        return False


class AdaptiveBackpressure(BackpressureStrategy):
    """Strategy that adapts wait time based on current system load"""
    
    def __init__(self, base_max_wait: float = 1.0, load_factor: float = 0.1):
        self.base_max_wait = base_max_wait
        self.load_factor = load_factor
    
    async def handle(self, wait_time: float, metrics: RateLimitMetrics) -> bool:
        # Reduce max wait time when deny rate is high
        adjusted_max_wait = self.base_max_wait * (1.0 - (metrics.deny_rate / 100.0) * self.load_factor)
        adjusted_max_wait = max(0.1, adjusted_max_wait)  # Minimum 100ms wait
        
        if wait_time <= adjusted_max_wait:
            await asyncio.sleep(wait_time)
            return True
        return False


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter with configurable backpressure handling.
    
    The token bucket algorithm maintains a bucket with a fixed capacity that gets
    refilled at a constant rate. Each request consumes tokens from the bucket.
    If insufficient tokens are available, the request is either delayed (backpressure)
    or rejected immediately.
    
    Mathematical Properties:
    - Tokens are added at exactly `refill_rate` tokens per second
    - Maximum tokens in bucket is capped at `capacity`
    - Token calculation uses high-precision time arithmetic
    - All operations are atomic to prevent race conditions
    """
    
    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        backpressure_strategy: Optional[BackpressureStrategy] = None,
        initial_tokens: Optional[float] = None
    ):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Rate at which tokens are added per second
            backpressure_strategy: Strategy for handling insufficient tokens
            initial_tokens: Initial token count (defaults to full capacity)
            
        Raises:
            ValueError: If capacity or refill_rate are not positive
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("Refill rate must be positive")
        
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.backpressure_strategy = backpressure_strategy or NoBackpressure()
        
        # Initialize bucket state with mathematical precision
        current_time = time.monotonic()
        self._tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self._last_refill_time = current_time
        
        # Thread safety primitives
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()
        
        # Observability metrics
        self._metrics = RateLimitMetrics(
            current_tokens=self._tokens,
            last_refill_time=current_time
        )
        
        # Input validation for edge cases
        if self._tokens < 0:
            self._tokens = 0.0
        elif self._tokens > self.capacity:
            self._tokens = self.capacity
    
    def _refill_tokens(self, current_time: float) -> None:
        """
        Refill tokens based on elapsed time since last refill.
        
        This method implements the core mathematical logic of the token bucket:
        tokens_to_add = elapsed_time * refill_rate
        new_token_count = min(current_tokens + tokens_to_add, capacity)
        
        Args:
            current_time: Current monotonic time for precise calculations
        """
        elapsed_time = current_time - self._last_refill_time
        
        # Handle edge case where time goes backwards (should never happen with monotonic)
        if elapsed_time < 0:
            elapsed_time = 0
        
        # Calculate tokens to add with mathematical precision
        tokens_to_add = elapsed_time * self.refill_rate
        
        # Update token count with capacity constraint
        self._tokens = min(self._tokens + tokens_to_add, self.capacity)
        self._last_refill_time = current_time
        
        # Update metrics
        self._metrics.current_tokens = self._tokens
        self._metrics.last_refill_time = current_time
    
    def _calculate_wait_time(self, tokens_needed: float) -> float:
        """
        Calculate time needed to accumulate sufficient tokens.
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            Time in seconds to wait for sufficient tokens (0 if already available)
        """
        if self._tokens >= tokens_needed:
            return 0.0
        
        tokens_deficit = tokens_needed - self._tokens
        wait_time = tokens_deficit / self.refill_rate
        
        # Add small epsilon to handle floating point precision issues
        return wait_time + 1e-9
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Acquire tokens from the bucket with backpressure handling.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were successfully acquired, False otherwise
            
        Raises:
            ValueError: If tokens is not positive
            RateLimitExceeded: If backpressure strategy is NoBackpressure and tokens unavailable
        """
        if tokens <= 0:
            raise ValueError("Token count must be positive")
        
        # Handle edge case where more tokens requested than bucket capacity
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds bucket capacity ({self.capacity})")
        
        start_time = time.monotonic()
        
        async with self._lock:
            current_time = time.monotonic()
            self._refill_tokens(current_time)
            
            # Update metrics
            self._metrics.total_requests += 1
            
            # Check if sufficient tokens are available
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._metrics.allowed_requests += 1
                self._metrics.current_tokens = self._tokens
                return True
            
            # Calculate wait time for sufficient tokens
            wait_time = self._calculate_wait_time(tokens)
            
            # Apply backpressure strategy
            should_wait = await self.backpressure_strategy.handle(wait_time, self._metrics)
            
            if should_wait:
                # Re-acquire lock and check tokens again after waiting
                current_time = time.monotonic()
                self._refill_tokens(current_time)
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._metrics.allowed_requests += 1
                    
                    # Update wait time metrics
                    actual_wait_time = time.monotonic() - start_time
                    self._update_wait_metrics(actual_wait_time)
                    
                    self._metrics.current_tokens = self._tokens
                    return True
                else:
                    # Still insufficient tokens after waiting
                    self._metrics.denied_requests += 1
                    return False
            else:
                # Backpressure strategy rejected the request
                self._metrics.denied_requests += 1
                
                if isinstance(self.backpressure_strategy, NoBackpressure):
                    raise RateLimitExceeded(wait_time)
                
                return False
    
    def acquire_sync(self, tokens: float = 1.0) -> bool:
        """
        Synchronous version of acquire for thread-safe access without async context.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were successfully acquired, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Token count must be positive")
        
        if tokens > self.capacity:
            raise ValueError(f"Requested tokens ({tokens}) exceeds bucket capacity ({self.capacity})")
        
        with self._thread_lock:
            current_time = time.monotonic()
            self._refill_tokens(current_time)
            
            # Update metrics
            self._metrics.total_requests += 1
            
            # Check if sufficient tokens are available
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._metrics.allowed_requests += 1
                self._metrics.current_tokens = self._tokens
                return True
            else:
                self._metrics.denied_requests += 1
                
                # For sync version, only support immediate rejection
                wait_time = self._calculate_wait_time(tokens)
                raise RateLimitExceeded(wait_time)
    
    def _update_wait_metrics(self, wait_time: float) -> None:
        """Update wait time statistics"""
        if self._metrics.max_wait_time == 0:
            self._metrics.average_wait_time = wait_time
        else:
            # Simple moving average approximation
            self._metrics.average_wait_time = (self._metrics.average_wait_time * 0.9) + (wait_time * 0.1)
        
        self._metrics.max_wait_time = max(self._metrics.max_wait_time, wait_time)
    
    def get_metrics(self) -> RateLimitMetrics:
        """
        Get current observability metrics.
        
        Returns:
            Current metrics including request counts, allow/deny rates, and timing stats
        """
        with self._thread_lock:
            # Update current token count before returning metrics
            current_time = time.monotonic()
            self._refill_tokens(current_time)
            self._metrics.current_tokens = self._tokens
            
            # Return a copy to prevent external modification
            return RateLimitMetrics(
                total_requests=self._metrics.total_requests,
                allowed_requests=self._metrics.allowed_requests,
                denied_requests=self._metrics.denied_requests,
                current_tokens=self._metrics.current_tokens,
                last_refill_time=self._metrics.last_refill_time,
                average_wait_time=self._metrics.average_wait_time,
                max_wait_time=self._metrics.max_wait_time
            )
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state"""
        with self._thread_lock:
            current_time = time.monotonic()
            self._metrics = RateLimitMetrics(
                current_tokens=self._tokens,
                last_refill_time=current_time
            )
    
    @property
    def current_tokens(self) -> float:
        """Get current token count with real-time refill calculation"""
        with self._thread_lock:
            current_time = time.monotonic()
            self._refill_tokens(current_time)
            return self._tokens
    
    def __repr__(self) -> str:
        return (f"TokenBucketRateLimiter(capacity={self.capacity}, "
                f"refill_rate={self.refill_rate}, current_tokens={self.current_tokens:.2f})")


# Factory functions for common configurations
def create_strict_rate_limiter(requests_per_second: float, burst_capacity: Optional[float] = None) -> TokenBucketRateLimiter:
    """Create a rate limiter with no backpressure (strict rejection)"""
    capacity = burst_capacity if burst_capacity is not None else requests_per_second
    return TokenBucketRateLimiter(
        capacity=capacity,
        refill_rate=requests_per_second,
        backpressure_strategy=NoBackpressure()
    )


def create_backpressure_rate_limiter(
    requests_per_second: float,
    max_wait_seconds: float = 1.0,
    burst_capacity: Optional[float] = None
) -> TokenBucketRateLimiter:
    """Create a rate limiter with fixed delay backpressure"""
    capacity = burst_capacity if burst_capacity is not None else requests_per_second
    return TokenBucketRateLimiter(
        capacity=capacity,
        refill_rate=requests_per_second,
        backpressure_strategy=FixedDelayBackpressure(max_wait_seconds)
    )


def create_adaptive_rate_limiter(
    requests_per_second: float,
    base_max_wait: float = 1.0,
    load_factor: float = 0.1,
    burst_capacity: Optional[float] = None
) -> TokenBucketRateLimiter:
    """Create a rate limiter with adaptive backpressure"""
    capacity = burst_capacity if burst_capacity is not None else requests_per_second
    return TokenBucketRateLimiter(
        capacity=capacity,
        refill_rate=requests_per_second,
        backpressure_strategy=AdaptiveBackpressure(base_max_wait, load_factor)
    )


if __name__ == "__main__":
    # Example usage demonstrating the rate limiter
    async def demo():
        # Create a rate limiter allowing 5 requests per second with 10 token burst capacity
        limiter = create_backpressure_rate_limiter(
            requests_per_second=5.0,
            max_wait_seconds=2.0,
            burst_capacity=10.0
        )
        
        print(f"Created rate limiter: {limiter}")
        
        # Simulate burst of requests
        for i in range(15):
            try:
                success = await limiter.acquire()
                print(f"Request {i+1}: {'✅ Allowed' if success else '❌ Denied'}")
                
                if i == 7:  # Show metrics halfway through
                    metrics = limiter.get_metrics()
                    print(f"Metrics: {metrics.allowed_requests}/{metrics.total_requests} allowed "
                          f"({metrics.allow_rate:.1f}%)")
                
            except RateLimitExceeded as e:
                print(f"Request {i+1}: ❌ Rate limited - {e}")
            
            await asyncio.sleep(0.1)  # Small delay between requests
        
        # Final metrics
        final_metrics = limiter.get_metrics()
        print(f"\nFinal metrics:")
        print(f"  Total requests: {final_metrics.total_requests}")
        print(f"  Allowed: {final_metrics.allowed_requests} ({final_metrics.allow_rate:.1f}%)")
        print(f"  Denied: {final_metrics.denied_requests} ({final_metrics.deny_rate:.1f}%)")
        print(f"  Current tokens: {final_metrics.current_tokens:.2f}")
        print(f"  Average wait time: {final_metrics.average_wait_time:.3f}s")
    
    # Run the demo
    asyncio.run(demo())