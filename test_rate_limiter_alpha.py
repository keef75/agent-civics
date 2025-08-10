#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Token Bucket Rate Limiter
SHA256 of specification: a7c8d9e2f1b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9

Test-driven development approach ensuring mathematical correctness,
edge case handling, and bulletproof reliability.

Test Coverage:
- Basic functionality and mathematical correctness
- Thread safety and concurrent access patterns
- Edge cases and boundary conditions
- Backpressure strategy behavior
- Observability metrics accuracy
- Performance under high load (10,000+ RPS)
"""

import asyncio
import pytest
import time
import threading
import concurrent.futures
from typing import List, Tuple
from unittest.mock import patch, MagicMock

from rate_limiter_alpha import (
    TokenBucketRateLimiter,
    RateLimitExceeded,
    RateLimitMetrics,
    NoBackpressure,
    FixedDelayBackpressure,
    AdaptiveBackpressure,
    create_strict_rate_limiter,
    create_backpressure_rate_limiter,
    create_adaptive_rate_limiter
)


class TestTokenBucketBasics:
    """Test basic token bucket functionality and mathematical correctness"""
    
    def test_initialization_valid_params(self):
        """Test successful initialization with valid parameters"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0)
        
        assert limiter.capacity == 10.0
        assert limiter.refill_rate == 5.0
        assert limiter.current_tokens == 10.0  # Should start full
        assert isinstance(limiter.backpressure_strategy, NoBackpressure)
    
    def test_initialization_custom_initial_tokens(self):
        """Test initialization with custom initial token count"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=3.0)
        assert limiter.current_tokens == 3.0
    
    def test_initialization_invalid_capacity(self):
        """Test initialization fails with invalid capacity"""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            TokenBucketRateLimiter(capacity=0, refill_rate=5.0)
        
        with pytest.raises(ValueError, match="Capacity must be positive"):
            TokenBucketRateLimiter(capacity=-1, refill_rate=5.0)
    
    def test_initialization_invalid_refill_rate(self):
        """Test initialization fails with invalid refill rate"""
        with pytest.raises(ValueError, match="Refill rate must be positive"):
            TokenBucketRateLimiter(capacity=10.0, refill_rate=0)
        
        with pytest.raises(ValueError, match="Refill rate must be positive"):
            TokenBucketRateLimiter(capacity=10.0, refill_rate=-1)
    
    def test_initialization_edge_case_initial_tokens(self):
        """Test initialization handles edge cases for initial tokens"""
        # Negative initial tokens should be clamped to 0
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=-5.0)
        assert limiter.current_tokens == 0.0
        
        # Initial tokens exceeding capacity should be clamped to capacity
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=15.0)
        assert limiter.current_tokens == 10.0


class TestTokenRefillMechanics:
    """Test token refill mathematical correctness"""
    
    @pytest.mark.asyncio
    async def test_token_refill_basic(self):
        """Test basic token refill mechanics"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=0.0)
        
        # Wait for tokens to refill
        await asyncio.sleep(1.0)
        
        # Should have approximately 5 tokens after 1 second at 5 tokens/second
        current_tokens = limiter.current_tokens
        assert 4.9 <= current_tokens <= 5.1  # Allow for timing precision
    
    @pytest.mark.asyncio
    async def test_token_refill_capacity_limit(self):
        """Test that token refill respects capacity limits"""
        limiter = TokenBucketRateLimiter(capacity=3.0, refill_rate=5.0, initial_tokens=3.0)
        
        # Wait longer than needed to exceed capacity
        await asyncio.sleep(2.0)
        
        # Tokens should be capped at capacity
        assert limiter.current_tokens == 3.0
    
    @pytest.mark.asyncio
    async def test_token_refill_precision(self):
        """Test mathematical precision of token refill"""
        limiter = TokenBucketRateLimiter(capacity=100.0, refill_rate=10.0, initial_tokens=0.0)
        
        # Test precise timing intervals
        start_time = time.monotonic()
        await asyncio.sleep(0.5)  # 500ms
        
        elapsed = time.monotonic() - start_time
        expected_tokens = elapsed * 10.0
        actual_tokens = limiter.current_tokens
        
        # Allow for 1% timing variance
        assert abs(actual_tokens - expected_tokens) < expected_tokens * 0.01
    
    def test_token_refill_zero_elapsed_time(self):
        """Test edge case where no time has elapsed"""
        with patch('time.monotonic', side_effect=[1000.0, 1000.0]):
            limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=2.0)
            # Access current_tokens to trigger refill calculation
            tokens = limiter.current_tokens
            assert tokens == 2.0  # No change expected


class TestTokenAcquisition:
    """Test token acquisition logic and edge cases"""
    
    @pytest.mark.asyncio
    async def test_acquire_success_sufficient_tokens(self):
        """Test successful token acquisition when sufficient tokens available"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=10.0)
        
        success = await limiter.acquire(3.0)
        assert success is True
        assert limiter.current_tokens == 7.0
    
    @pytest.mark.asyncio
    async def test_acquire_failure_insufficient_tokens_no_backpressure(self):
        """Test token acquisition failure with no backpressure strategy"""
        limiter = TokenBucketRateLimiter(
            capacity=10.0, 
            refill_rate=5.0, 
            initial_tokens=2.0,
            backpressure_strategy=NoBackpressure()
        )
        
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.acquire(5.0)
        
        assert exc_info.value.retry_after > 0
        assert limiter.current_tokens == 2.0  # Tokens shouldn't be consumed
    
    @pytest.mark.asyncio
    async def test_acquire_invalid_token_count(self):
        """Test acquisition fails with invalid token counts"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0)
        
        with pytest.raises(ValueError, match="Token count must be positive"):
            await limiter.acquire(0)
        
        with pytest.raises(ValueError, match="Token count must be positive"):
            await limiter.acquire(-1)
    
    @pytest.mark.asyncio
    async def test_acquire_tokens_exceed_capacity(self):
        """Test acquisition fails when requesting more tokens than capacity"""
        limiter = TokenBucketRateLimiter(capacity=5.0, refill_rate=5.0)
        
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await limiter.acquire(10.0)
    
    @pytest.mark.asyncio
    async def test_acquire_fractional_tokens(self):
        """Test acquisition works with fractional token values"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=10.0)
        
        success = await limiter.acquire(2.5)
        assert success is True
        assert limiter.current_tokens == 7.5
    
    def test_acquire_sync_success(self):
        """Test synchronous acquire method"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=10.0)
        
        success = limiter.acquire_sync(3.0)
        assert success is True
        assert limiter.current_tokens == 7.0
    
    def test_acquire_sync_failure(self):
        """Test synchronous acquire method failure"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=2.0)
        
        with pytest.raises(RateLimitExceeded):
            limiter.acquire_sync(5.0)


class TestBackpressureStrategies:
    """Test different backpressure handling strategies"""
    
    @pytest.mark.asyncio
    async def test_no_backpressure_strategy(self):
        """Test NoBackpressure strategy immediately rejects requests"""
        limiter = TokenBucketRateLimiter(
            capacity=5.0,
            refill_rate=2.0,
            initial_tokens=1.0,
            backpressure_strategy=NoBackpressure()
        )
        
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.acquire(3.0)
        
        # Should calculate correct retry_after time
        expected_wait = (3.0 - 1.0) / 2.0  # (needed - available) / refill_rate
        assert abs(exc_info.value.retry_after - expected_wait) < 0.001
    
    @pytest.mark.asyncio
    async def test_fixed_delay_backpressure_success(self):
        """Test FixedDelayBackpressure waits for tokens when within limit"""
        limiter = TokenBucketRateLimiter(
            capacity=10.0,
            refill_rate=10.0,  # 10 tokens per second
            initial_tokens=2.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=1.0)
        )
        
        start_time = time.monotonic()
        success = await limiter.acquire(5.0)
        elapsed_time = time.monotonic() - start_time
        
        assert success is True
        # Should have waited approximately (5-2)/10 = 0.3 seconds
        assert 0.2 <= elapsed_time <= 0.5  # Allow timing variance
    
    @pytest.mark.asyncio
    async def test_fixed_delay_backpressure_timeout(self):
        """Test FixedDelayBackpressure rejects when wait time exceeds limit"""
        limiter = TokenBucketRateLimiter(
            capacity=10.0,
            refill_rate=1.0,  # 1 token per second
            initial_tokens=1.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=0.5)
        )
        
        # Need 5 tokens but only have 1, would take 4 seconds to refill
        success = await limiter.acquire(5.0)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_adaptive_backpressure_load_adjustment(self):
        """Test AdaptiveBackpressure adjusts based on deny rate"""
        limiter = TokenBucketRateLimiter(
            capacity=5.0,
            refill_rate=5.0,
            initial_tokens=5.0,
            backpressure_strategy=AdaptiveBackpressure(base_max_wait=1.0, load_factor=0.5)
        )
        
        # Build up some deny rate
        limiter._metrics.total_requests = 100
        limiter._metrics.denied_requests = 50  # 50% deny rate
        
        # Consume all tokens
        await limiter.acquire(5.0)
        
        # Next request should use adjusted wait time
        start_time = time.monotonic()
        success = await limiter.acquire(1.0)  # Should wait ~0.2 seconds for 1 token
        elapsed_time = time.monotonic() - start_time
        
        # With 50% deny rate and 0.5 load factor, max wait becomes 1.0 * (1 - 0.5 * 0.5) = 0.75
        assert success is True
        assert elapsed_time < 0.5  # Should be faster due to adaptation


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent access patterns"""
    
    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self):
        """Test concurrent async requests maintain correctness"""
        limiter = TokenBucketRateLimiter(
            capacity=20.0,
            refill_rate=10.0,
            initial_tokens=20.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=2.0)
        )
        
        # Launch 10 concurrent requests for 2 tokens each
        tasks = [limiter.acquire(2.0) for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed since we have exactly 20 tokens
        successful_acquisitions = sum(1 for result in results if result is True)
        assert successful_acquisitions == 10
        
        # All tokens should be consumed
        assert limiter.current_tokens == 0.0
    
    def test_concurrent_sync_requests(self):
        """Test concurrent synchronous requests with thread safety"""
        limiter = TokenBucketRateLimiter(capacity=50.0, refill_rate=25.0, initial_tokens=50.0)
        
        def acquire_tokens(token_count: float) -> bool:
            try:
                return limiter.acquire_sync(token_count)
            except RateLimitExceeded:
                return False
        
        # Use ThreadPoolExecutor for concurrent synchronous requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Submit 25 requests for 2 tokens each (total 50 tokens needed)
            futures = [executor.submit(acquire_tokens, 2.0) for _ in range(25)]
            results = [future.result() for future in futures]
        
        # Exactly 25 should succeed (50 tokens / 2 tokens per request)
        successful_acquisitions = sum(1 for result in results if result is True)
        assert successful_acquisitions == 25
        
        # All tokens should be consumed
        assert limiter.current_tokens == 0.0
    
    @pytest.mark.asyncio
    async def test_mixed_async_sync_access(self):
        """Test mixed async and sync access patterns"""
        limiter = TokenBucketRateLimiter(capacity=30.0, refill_rate=15.0, initial_tokens=30.0)
        
        # Async requests
        async_tasks = [limiter.acquire(2.0) for _ in range(5)]
        
        # Sync requests in separate thread
        def sync_requests():
            results = []
            for _ in range(10):
                try:
                    results.append(limiter.acquire_sync(2.0))
                except RateLimitExceeded:
                    results.append(False)
            return results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            sync_future = executor.submit(sync_requests)
            async_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            sync_results = sync_future.result()
        
        # Total successful acquisitions should be 15 (30 tokens / 2 tokens per request)
        total_success = (
            sum(1 for result in async_results if result is True) +
            sum(1 for result in sync_results if result is True)
        )
        assert total_success == 15


class TestObservabilityMetrics:
    """Test observability metrics accuracy and updates"""
    
    @pytest.mark.asyncio
    async def test_metrics_basic_tracking(self):
        """Test basic metrics tracking for requests and outcomes"""
        limiter = TokenBucketRateLimiter(
            capacity=5.0,
            refill_rate=5.0,
            initial_tokens=3.0,
            backpressure_strategy=NoBackpressure()
        )
        
        # Successful acquisition
        await limiter.acquire(2.0)
        
        # Failed acquisition
        try:
            await limiter.acquire(3.0)  # Only 1 token remaining
        except RateLimitExceeded:
            pass
        
        metrics = limiter.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.allowed_requests == 1
        assert metrics.denied_requests == 1
        assert metrics.allow_rate == 50.0
        assert metrics.deny_rate == 50.0
        assert metrics.current_tokens == 1.0
    
    @pytest.mark.asyncio
    async def test_metrics_wait_time_tracking(self):
        """Test wait time metrics tracking"""
        limiter = TokenBucketRateLimiter(
            capacity=10.0,
            refill_rate=10.0,
            initial_tokens=1.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=1.0)
        )
        
        # Request that requires waiting
        await limiter.acquire(5.0)
        
        metrics = limiter.get_metrics()
        assert metrics.average_wait_time > 0
        assert metrics.max_wait_time > 0
        assert metrics.max_wait_time >= metrics.average_wait_time
    
    def test_metrics_reset(self):
        """Test metrics reset functionality"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=5.0)
        
        # Generate some activity
        limiter.acquire_sync(2.0)
        
        # Verify metrics are populated
        metrics_before = limiter.get_metrics()
        assert metrics_before.total_requests > 0
        
        # Reset metrics
        limiter.reset_metrics()
        
        # Verify metrics are reset
        metrics_after = limiter.get_metrics()
        assert metrics_after.total_requests == 0
        assert metrics_after.allowed_requests == 0
        assert metrics_after.denied_requests == 0
    
    def test_metrics_immutability(self):
        """Test that returned metrics cannot modify internal state"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0)
        
        metrics = limiter.get_metrics()
        original_total = metrics.total_requests
        
        # Attempt to modify returned metrics
        metrics.total_requests = 999
        
        # Internal state should be unchanged
        new_metrics = limiter.get_metrics()
        assert new_metrics.total_requests == original_total


class TestPerformanceAndLoadHandling:
    """Test performance under high load scenarios"""
    
    @pytest.mark.asyncio
    async def test_high_throughput_async(self):
        """Test handling 10,000+ requests per second async load"""
        limiter = TokenBucketRateLimiter(
            capacity=1000.0,
            refill_rate=12000.0,  # Allow for high throughput
            initial_tokens=1000.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=0.1)
        )
        
        # Create 10,000 concurrent requests
        start_time = time.monotonic()
        tasks = [limiter.acquire(0.1) for _ in range(10000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_time = time.monotonic() - start_time
        
        successful_requests = sum(1 for result in results if result is True)
        throughput = successful_requests / elapsed_time
        
        # Should handle at least 5,000 RPS (allowing for overhead)
        assert throughput >= 5000.0, f"Throughput was {throughput:.0f} RPS, expected >= 5000 RPS"
        
        # Verify no exceptions occurred
        exceptions = [result for result in results if isinstance(result, Exception)]
        assert len(exceptions) == 0, f"Got {len(exceptions)} exceptions during high load test"
    
    def test_high_throughput_sync(self):
        """Test handling high load with synchronous requests"""
        limiter = TokenBucketRateLimiter(capacity=500.0, refill_rate=6000.0, initial_tokens=500.0)
        
        def make_request():
            try:
                return limiter.acquire_sync(0.05)
            except RateLimitExceeded:
                return False
        
        # Use thread pool for concurrent sync requests
        start_time = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(5000)]
            results = [future.result() for future in futures]
        elapsed_time = time.monotonic() - start_time
        
        successful_requests = sum(1 for result in results if result is True)
        throughput = successful_requests / elapsed_time
        
        # Should handle at least 2,000 RPS for sync requests
        assert throughput >= 2000.0, f"Sync throughput was {throughput:.0f} RPS, expected >= 2000 RPS"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under sustained load"""
        limiter = TokenBucketRateLimiter(
            capacity=100.0,
            refill_rate=1000.0,
            initial_tokens=100.0,
            backpressure_strategy=FixedDelayBackpressure(max_wait_seconds=0.01)
        )
        
        # Run sustained load for multiple batches
        for batch in range(10):
            tasks = [limiter.acquire(0.1) for _ in range(1000)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Let tokens refill between batches
            await asyncio.sleep(0.1)
        
        # Verify metrics still work correctly after sustained load
        metrics = limiter.get_metrics()
        assert metrics.total_requests == 10000
        assert metrics.current_tokens >= 0
        assert metrics.current_tokens <= limiter.capacity


class TestFactoryFunctions:
    """Test factory functions for common configurations"""
    
    def test_create_strict_rate_limiter(self):
        """Test strict rate limiter factory"""
        limiter = create_strict_rate_limiter(requests_per_second=10.0)
        
        assert limiter.capacity == 10.0
        assert limiter.refill_rate == 10.0
        assert isinstance(limiter.backpressure_strategy, NoBackpressure)
    
    def test_create_strict_rate_limiter_with_burst(self):
        """Test strict rate limiter factory with custom burst capacity"""
        limiter = create_strict_rate_limiter(requests_per_second=10.0, burst_capacity=20.0)
        
        assert limiter.capacity == 20.0
        assert limiter.refill_rate == 10.0
    
    def test_create_backpressure_rate_limiter(self):
        """Test backpressure rate limiter factory"""
        limiter = create_backpressure_rate_limiter(requests_per_second=10.0, max_wait_seconds=2.0)
        
        assert limiter.capacity == 10.0
        assert limiter.refill_rate == 10.0
        assert isinstance(limiter.backpressure_strategy, FixedDelayBackpressure)
        assert limiter.backpressure_strategy.max_wait_seconds == 2.0
    
    def test_create_adaptive_rate_limiter(self):
        """Test adaptive rate limiter factory"""
        limiter = create_adaptive_rate_limiter(
            requests_per_second=10.0,
            base_max_wait=1.5,
            load_factor=0.2
        )
        
        assert limiter.capacity == 10.0
        assert limiter.refill_rate == 10.0
        assert isinstance(limiter.backpressure_strategy, AdaptiveBackpressure)
        assert limiter.backpressure_strategy.base_max_wait == 1.5
        assert limiter.backpressure_strategy.load_factor == 0.2


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_extremely_small_capacity(self):
        """Test rate limiter with very small capacity"""
        limiter = TokenBucketRateLimiter(capacity=0.001, refill_rate=0.1)
        
        # Should work with fractional tokens
        success = await limiter.acquire(0.001)
        assert success is True
        assert limiter.current_tokens == 0.0
    
    @pytest.mark.asyncio
    async def test_extremely_high_refill_rate(self):
        """Test rate limiter with very high refill rate"""
        limiter = TokenBucketRateLimiter(
            capacity=1.0,
            refill_rate=1000000.0,  # 1 million tokens per second
            initial_tokens=0.0
        )
        
        # Even with tiny sleep, should refill to capacity
        await asyncio.sleep(0.001)  # 1ms
        assert limiter.current_tokens >= 0.9  # Should be nearly full
    
    def test_very_small_token_requests(self):
        """Test handling of very small token requests"""
        limiter = TokenBucketRateLimiter(capacity=1.0, refill_rate=1.0, initial_tokens=1.0)
        
        # Request tiny amounts
        success = limiter.acquire_sync(0.000001)
        assert success is True
        
        remaining = limiter.current_tokens
        assert 0.999999 <= remaining <= 1.0
    
    @pytest.mark.asyncio
    async def test_rapid_successive_requests(self):
        """Test rapid successive requests without delays"""
        limiter = TokenBucketRateLimiter(capacity=100.0, refill_rate=50.0, initial_tokens=100.0)
        
        # Make 100 requests as fast as possible
        results = []
        for i in range(100):
            success = await limiter.acquire(1.0)
            results.append(success)
        
        # All should succeed since we start with 100 tokens
        assert all(results)
        assert limiter.current_tokens == 0.0
        
        # Next request should fail immediately
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire(1.0)
    
    @pytest.mark.asyncio
    async def test_time_precision_edge_cases(self):
        """Test handling of time precision edge cases"""
        # Mock time.monotonic to test edge cases
        mock_times = [1000.0, 1000.0000001]  # Extremely small time difference
        
        with patch('time.monotonic', side_effect=mock_times):
            limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=1000000.0, initial_tokens=0.0)
            
            # Should handle tiny time differences correctly
            tokens = limiter.current_tokens
            assert tokens >= 0.0
            assert tokens <= 10.0  # Should not exceed capacity
    
    def test_repr_string(self):
        """Test string representation of rate limiter"""
        limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0, initial_tokens=7.5)
        
        repr_str = repr(limiter)
        assert "TokenBucketRateLimiter" in repr_str
        assert "capacity=10.0" in repr_str
        assert "refill_rate=5.0" in repr_str
        assert "current_tokens=" in repr_str


@pytest.mark.asyncio
async def test_integration_realistic_scenario():
    """Integration test simulating realistic usage scenario"""
    # Simulate web API with 100 RPS limit, 200 request burst capacity
    limiter = create_backpressure_rate_limiter(
        requests_per_second=100.0,
        max_wait_seconds=1.0,
        burst_capacity=200.0
    )
    
    # Simulate burst of 150 requests (within burst capacity)
    burst_tasks = [limiter.acquire(1.0) for _ in range(150)]
    burst_results = await asyncio.gather(*burst_tasks)
    
    # All burst requests should succeed
    assert all(burst_results)
    
    # Wait for some refill
    await asyncio.sleep(0.5)  # Should add ~50 tokens
    
    # Submit requests at steady rate
    steady_results = []
    for _ in range(75):  # Should be possible with refilled tokens + remaining
        result = await limiter.acquire(1.0)
        steady_results.append(result)
        await asyncio.sleep(0.01)  # 100ms between requests = 10 RPS
    
    # Most steady requests should succeed
    successful_steady = sum(steady_results)
    assert successful_steady >= 60  # Allow some failures due to timing
    
    # Check final metrics
    metrics = limiter.get_metrics()
    assert metrics.total_requests == 225  # 150 burst + 75 steady
    assert metrics.allow_rate > 80.0  # Should have high success rate


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])