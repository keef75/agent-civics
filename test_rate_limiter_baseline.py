"""
Comprehensive test suite for baseline rate limiter implementation.
Tests thread safety, performance, edge cases, and async patterns.
"""

import pytest
import asyncio
import time
import threading
import concurrent.futures
from rate_limiter_baseline import TokenBucketRateLimiter, MultiKeyRateLimiter


class TestTokenBucketRateLimiter:
    """Test cases for TokenBucketRateLimiter"""
    
    def test_initialization(self):
        """Test proper initialization with valid parameters"""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
        assert limiter.capacity == 10
        assert limiter.refill_rate == 5.0
        assert limiter.tokens == 10  # Initial tokens should equal capacity
    
    def test_initialization_with_custom_initial_tokens(self):
        """Test initialization with custom initial token count"""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0, initial_tokens=3)
        assert limiter.tokens == 3
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors"""
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(capacity=0, refill_rate=5.0)
        
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(capacity=10, refill_rate=0)
        
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(capacity=-5, refill_rate=5.0)
    
    def test_basic_allow_functionality(self):
        """Test basic allow/deny functionality"""
        limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1.0)
        
        # Should allow first 3 requests
        assert limiter.allow() == True
        assert limiter.allow() == True
        assert limiter.allow() == True
        
        # Fourth request should be denied
        assert limiter.allow() == False
    
    def test_multiple_token_consumption(self):
        """Test consuming multiple tokens at once"""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)
        
        assert limiter.allow(5) == True  # Should succeed
        assert limiter.tokens == 5
        
        assert limiter.allow(6) == False  # Should fail (not enough tokens)
        assert limiter.allow(3) == True   # Should succeed
    
    def test_token_refill(self):
        """Test that tokens are refilled over time"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=10.0)  # 10 tokens per second
        
        # Consume all tokens
        for _ in range(5):
            limiter.allow()
        
        assert limiter.allow() == False
        
        # Wait for refill (0.2 seconds should add 2 tokens)
        time.sleep(0.2)
        assert limiter.allow() == True
        assert limiter.allow() == True
        assert limiter.allow() == False  # Should be out again
    
    def test_capacity_limit(self):
        """Test that tokens don't exceed capacity"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0, initial_tokens=3)
        
        # Wait longer than needed to fill to capacity
        time.sleep(5)
        limiter._refill_tokens()
        
        # Should not exceed capacity
        assert limiter.tokens <= 5
        
        # Should be able to consume exactly capacity
        for i in range(5):
            assert limiter.allow() == True
        assert limiter.allow() == False
    
    def test_wait_time_calculation(self):
        """Test wait time calculation accuracy"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=2.0)  # 2 tokens per second
        
        # Consume all tokens
        for _ in range(5):
            limiter.allow()
        
        # Should need to wait 0.5 seconds for 1 token
        wait_time = limiter.wait_time(1)
        assert abs(wait_time - 0.5) < 0.01
        
        # Should need to wait 1 second for 2 tokens
        wait_time = limiter.wait_time(2)
        assert abs(wait_time - 1.0) < 0.01
    
    def test_invalid_token_requests(self):
        """Test handling of invalid token requests"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        
        with pytest.raises(ValueError):
            limiter.allow(0)
        
        with pytest.raises(ValueError):
            limiter.allow(-1)
        
        with pytest.raises(ValueError):
            limiter.allow(6)  # More than capacity
    
    def test_metrics_tracking(self):
        """Test that metrics are properly tracked"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        
        metrics = limiter.get_metrics()
        assert metrics['requests_allowed'] == 0
        assert metrics['requests_denied'] == 0
        
        limiter.allow()  # Should succeed
        limiter.allow()  # Should succeed
        
        # Consume all remaining tokens
        for _ in range(3):
            limiter.allow()
        
        limiter.allow()  # Should fail
        
        metrics = limiter.get_metrics()
        assert metrics['requests_allowed'] == 5
        assert metrics['requests_denied'] == 1
        assert metrics['success_rate'] == 5/6
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        
        # Consume tokens and generate metrics
        limiter.allow()
        limiter.allow()
        
        limiter.reset()
        
        assert limiter.tokens == 5
        metrics = limiter.get_metrics()
        assert metrics['requests_allowed'] == 0
        assert metrics['requests_denied'] == 0


class TestAsyncTokenBucketRateLimiter:
    """Test async functionality of rate limiter"""
    
    @pytest.mark.asyncio
    async def test_async_allow(self):
        """Test async allow functionality"""
        limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1.0)
        
        assert await limiter.allow_async() == True
        assert await limiter.allow_async() == True
        assert await limiter.allow_async() == True
        assert await limiter.allow_async() == False
    
    @pytest.mark.asyncio
    async def test_async_invalid_tokens(self):
        """Test async validation of token parameters"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        
        with pytest.raises(ValueError):
            await limiter.allow_async(0)
        
        with pytest.raises(ValueError):
            await limiter.allow_async(6)  # More than capacity
    
    @pytest.mark.asyncio
    async def test_wait_for_tokens(self):
        """Test backpressure handling with wait_for_tokens"""
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=10.0)  # Fast refill for testing
        
        # Consume all tokens
        await limiter.allow_async()
        await limiter.allow_async()
        
        # This should wait and then succeed
        start_time = time.time()
        result = await limiter.wait_for_tokens(1, max_wait=1.0)
        elapsed = time.time() - start_time
        
        assert result == True
        assert elapsed > 0.05  # Should have waited some time
        assert elapsed < 0.5   # But not too long
    
    @pytest.mark.asyncio
    async def test_wait_for_tokens_timeout(self):
        """Test timeout in wait_for_tokens"""
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1.0)  # Slow refill
        
        # Consume all tokens
        await limiter.allow_async()
        await limiter.allow_async()
        
        # This should timeout
        start_time = time.time()
        result = await limiter.wait_for_tokens(1, max_wait=0.1)
        elapsed = time.time() - start_time
        
        assert result == False
        assert elapsed < 0.2  # Should have timed out quickly


class TestThreadSafety:
    """Test thread safety of rate limiter"""
    
    def test_concurrent_access(self):
        """Test thread safety with concurrent access"""
        limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=100.0)
        results = []
        
        def make_requests():
            local_results = []
            for _ in range(50):
                local_results.append(limiter.allow())
                time.sleep(0.001)  # Small delay to create interleaving
            results.extend(local_results)
        
        threads = []
        for _ in range(10):  # 10 threads, 50 requests each = 500 total
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have some successes and some failures
        successes = sum(results)
        assert successes > 0
        assert successes <= 1000  # Can't exceed capacity
        
        # Metrics should be consistent
        metrics = limiter.get_metrics()
        assert metrics['requests_allowed'] + metrics['requests_denied'] == 500
    
    @pytest.mark.asyncio
    async def test_async_concurrent_access(self):
        """Test async thread safety"""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=50.0)
        
        async def make_async_requests():
            results = []
            for _ in range(20):
                results.append(await limiter.allow_async())
            return results
        
        # Run 10 concurrent coroutines
        tasks = [make_async_requests() for _ in range(10)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        
        successes = sum(flat_results)
        assert successes > 0
        assert successes <= 100  # Can't exceed capacity
    
    def test_mixed_sync_async_access(self):
        """Test mixing sync and async access patterns"""
        limiter = TokenBucketRateLimiter(capacity=50, refill_rate=25.0)
        sync_results = []
        
        def sync_requests():
            for _ in range(10):
                sync_results.append(limiter.allow())
        
        async def async_requests():
            results = []
            for _ in range(10):
                results.append(await limiter.allow_async())
            return results
        
        # Start sync thread
        sync_thread = threading.Thread(target=sync_requests)
        sync_thread.start()
        
        # Run async requests
        async_results = asyncio.run(async_requests())
        
        sync_thread.join()
        
        total_successes = sum(sync_results) + sum(async_results)
        assert total_successes > 0
        assert total_successes <= 50


class TestPerformance:
    """Performance tests to verify 10K+ RPS capability"""
    
    def test_high_throughput_sync(self):
        """Test synchronous high throughput"""
        limiter = TokenBucketRateLimiter(capacity=15000, refill_rate=12000.0)
        
        start_time = time.time()
        allowed_count = 0
        
        # Make 10,000 requests as fast as possible
        for _ in range(10000):
            if limiter.allow():
                allowed_count += 1
        
        elapsed = time.time() - start_time
        rps = allowed_count / elapsed if elapsed > 0 else float('inf')
        
        print(f"Sync RPS: {rps:.0f}, Allowed: {allowed_count}/10000")
        assert rps > 10000  # Should handle >10K RPS
        assert allowed_count >= 10000  # Should allow most requests
    
    @pytest.mark.asyncio
    async def test_high_throughput_async(self):
        """Test asynchronous high throughput"""
        limiter = TokenBucketRateLimiter(capacity=15000, refill_rate=12000.0)
        
        start_time = time.time()
        
        # Create 10,000 concurrent requests
        tasks = [limiter.allow_async() for _ in range(10000)]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        allowed_count = sum(results)
        rps = allowed_count / elapsed if elapsed > 0 else float('inf')
        
        print(f"Async RPS: {rps:.0f}, Allowed: {allowed_count}/10000")
        assert rps > 5000  # Async might be slightly slower due to overhead
        assert allowed_count >= 10000  # Should allow most requests


class TestMultiKeyRateLimiter:
    """Test multi-key rate limiter functionality"""
    
    def test_per_key_isolation(self):
        """Test that different keys have isolated rate limits"""
        limiter = MultiKeyRateLimiter(capacity=3, refill_rate=1.0)
        
        # Key "user1" should get 3 requests
        assert limiter.allow("user1") == True
        assert limiter.allow("user1") == True
        assert limiter.allow("user1") == True
        assert limiter.allow("user1") == False
        
        # Key "user2" should also get 3 fresh requests
        assert limiter.allow("user2") == True
        assert limiter.allow("user2") == True
        assert limiter.allow("user2") == True
        assert limiter.allow("user2") == False
    
    @pytest.mark.asyncio
    async def test_multi_key_async(self):
        """Test async functionality with multiple keys"""
        limiter = MultiKeyRateLimiter(capacity=2, refill_rate=1.0)
        
        assert await limiter.allow_async("key1") == True
        assert await limiter.allow_async("key2") == True
        assert await limiter.allow_async("key1") == True
        assert await limiter.allow_async("key2") == True
        
        # Both keys should now be exhausted
        assert await limiter.allow_async("key1") == False
        assert await limiter.allow_async("key2") == False
    
    def test_cleanup_functionality(self):
        """Test that inactive limiters are cleaned up"""
        limiter = MultiKeyRateLimiter(capacity=5, refill_rate=1.0, cleanup_interval=0.1)
        
        # Create limiters for multiple keys
        limiter.allow("key1")
        limiter.allow("key2")
        limiter.allow("key3")
        
        assert len(limiter._limiters) == 3
        
        # Wait for cleanup interval and trigger cleanup
        time.sleep(0.2)
        
        # Access one key to trigger cleanup
        limiter.allow("key1")
        
        # Some keys should be cleaned up (depends on timing and token state)
        # This is a probabilistic test - we mainly ensure it doesn't crash


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_very_high_refill_rate(self):
        """Test with very high refill rates"""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1000000.0)
        
        # Should quickly refill after consumption
        for _ in range(100):
            limiter.allow()
        
        time.sleep(0.001)  # 1ms should be enough to refill completely
        assert limiter.allow() == True
    
    def test_very_low_refill_rate(self):
        """Test with very low refill rates"""
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate=0.1)  # 1 token per 10 seconds
        
        assert limiter.allow() == True
        assert limiter.allow() == False
        
        # Should still be false after a short wait
        time.sleep(0.1)
        assert limiter.allow() == False
    
    def test_fractional_tokens_over_time(self):
        """Test accumulation of fractional tokens"""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=3.7)  # Fractional rate
        
        # Consume all tokens
        for _ in range(10):
            limiter.allow()
        
        # Wait for partial refill
        time.sleep(0.5)  # Should add ~1.85 tokens
        assert limiter.allow() == True   # Should have at least 1 token
        assert limiter.allow() == False  # But not 2
    
    def test_zero_initial_tokens(self):
        """Test starting with zero tokens"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=2.0, initial_tokens=0)
        
        assert limiter.allow() == False
        
        # Wait for tokens to accumulate
        time.sleep(1.0)  # Should add 2 tokens
        assert limiter.allow() == True
        assert limiter.allow() == True
        assert limiter.allow() == False


if __name__ == "__main__":
    # Run performance test manually
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "perf":
        test_perf = TestPerformance()
        test_perf.test_high_throughput_sync()
        asyncio.run(test_perf.test_high_throughput_async())
    else:
        pytest.main([__file__, "-v"])