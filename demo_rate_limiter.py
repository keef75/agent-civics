#!/usr/bin/env python3
"""
High-Performance Rate Limiter Demo
==================================

Demonstrates key features of the token bucket rate limiter:
1. Basic rate limiting
2. Burst handling
3. Multi-key rate limiting
4. Performance metrics
5. Async/await support
"""

import asyncio
import time
from rate_limiter_beta import (
    HighPerformanceTokenBucket, 
    MultiKeyRateLimiter, 
    create_api_rate_limiter
)


def demo_basic_rate_limiting():
    """Demonstrate basic rate limiting functionality."""
    print("=== Basic Rate Limiting Demo ===")
    
    # Create limiter: 5 requests per second, 10 token capacity
    limiter = HighPerformanceTokenBucket(capacity=10, refill_rate=5)
    
    print(f"Initial tokens: {limiter.current_tokens}")
    
    # Rapid consumption
    for i in range(15):
        success = limiter.try_consume(1)
        print(f"Request {i+1}: {'✓' if success else '✗'} (tokens: {limiter.current_tokens:.1f})")
    
    print("\nWaiting 2 seconds for refill...")
    time.sleep(2)  # Allow 10 tokens to refill (5/sec * 2 sec)
    
    print(f"After refill: {limiter.current_tokens:.1f} tokens")
    print()


def demo_burst_handling():
    """Demonstrate burst allowance."""
    print("=== Burst Handling Demo ===")
    
    # Create limiter with 2x burst capacity
    burst_limiter = HighPerformanceTokenBucket(
        capacity=10, 
        refill_rate=5, 
        burst_allowance=2.0
    )
    
    print(f"Burst capacity: {burst_limiter.current_tokens} tokens")
    
    # Test large burst
    success = burst_limiter.try_consume(20)
    print(f"Consume 20 tokens: {'✓' if success else '✗'} (remaining: {burst_limiter.current_tokens:.1f})")
    
    # Should fail now
    success = burst_limiter.try_consume(1)
    print(f"Consume 1 more: {'✓' if success else '✗'}")
    print()


def demo_multi_key_limiting():
    """Demonstrate multi-key rate limiting."""
    print("=== Multi-Key Rate Limiting Demo ===")
    
    multi_limiter = MultiKeyRateLimiter(default_capacity=3, default_refill_rate=1)
    
    users = ["user1", "user2", "user3"]
    
    # Test each user
    for user in users:
        print(f"\n{user} requests:")
        for i in range(5):
            success = multi_limiter.try_consume(user, 1)
            metrics = multi_limiter.get_metrics(user)
            print(f"  Request {i+1}: {'✓' if success else '✗'} "
                  f"(tokens: {metrics.current_tokens:.1f})")
    
    print("\nAll user metrics:")
    all_metrics = multi_limiter.get_all_metrics()
    for user, metrics in all_metrics.items():
        print(f"  {user}: {metrics.success_rate:.1%} success rate, "
              f"{metrics.total_requests} total requests")
    print()


async def demo_async_rate_limiting():
    """Demonstrate async rate limiting with backpressure."""
    print("=== Async Rate Limiting Demo ===")
    
    async_limiter = HighPerformanceTokenBucket(capacity=2, refill_rate=1)
    
    # Consume all tokens
    async_limiter.try_consume(2)
    print(f"Consumed all tokens: {async_limiter.current_tokens}")
    
    # Try async consumption with timeout
    print("Waiting for tokens to refill...")
    start_time = time.time()
    
    success = await async_limiter.consume(1, max_wait_time=2.0)
    elapsed = time.time() - start_time
    
    print(f"Async consume result: {'✓' if success else '✗'} "
          f"(waited {elapsed:.1f}s, tokens: {async_limiter.current_tokens:.1f})")
    print()


def demo_performance_metrics():
    """Demonstrate performance metrics collection."""
    print("=== Performance Metrics Demo ===")
    
    metrics_limiter = HighPerformanceTokenBucket(capacity=100, refill_rate=50)
    
    # Generate mixed load
    import random
    for _ in range(1000):
        tokens = random.randint(1, 5)
        metrics_limiter.try_consume(tokens)
    
    metrics = metrics_limiter.get_metrics()
    
    print(f"Total requests: {metrics.total_requests}")
    print(f"Successful: {metrics.allowed_requests}")
    print(f"Rejected: {metrics.rejected_requests}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Avg processing time: {metrics.avg_processing_time_ns/1000:.1f}μs")
    print(f"Peak processing time: {metrics.peak_processing_time_ns/1000:.1f}μs")
    print(f"Current tokens: {metrics.current_tokens:.1f}")
    print()


def demo_api_rate_limiter():
    """Demonstrate API-specific rate limiter."""
    print("=== API Rate Limiter Demo ===")
    
    # 100 requests per second with 1.5x burst
    api_limiter = create_api_rate_limiter(requests_per_second=100, burst_capacity=1.5)
    
    print(f"API limiter capacity: {api_limiter.current_tokens} tokens")
    print(f"Can handle burst: {api_limiter.try_consume(150)} (150 requests)")
    print(f"Remaining tokens: {api_limiter.current_tokens}")
    print()


async def main():
    """Run all demos."""
    print("High-Performance Token Bucket Rate Limiter Demo")
    print("=" * 50)
    print()
    
    demo_basic_rate_limiting()
    demo_burst_handling()
    demo_multi_key_limiting()
    await demo_async_rate_limiting()
    demo_performance_metrics()
    demo_api_rate_limiter()
    
    print("Demo completed! 🚀")


if __name__ == "__main__":
    asyncio.run(main())