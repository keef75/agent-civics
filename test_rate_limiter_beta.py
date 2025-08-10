"""
Comprehensive Test Suite for High-Performance Token Bucket Rate Limiter
======================================================================

Tests covering:
- Basic functionality
- Thread safety and concurrent access
- Edge cases and error conditions
- Performance characteristics
- Backpressure handling
- Metrics accuracy
- Memory efficiency
"""

import asyncio
import threading
import time
import unittest
import concurrent.futures
import statistics
from unittest.mock import Mock
from typing import List, Tuple

from rate_limiter_beta import (
    HighPerformanceTokenBucket, 
    MultiKeyRateLimiter, 
    AtomicFloat,
    RateLimiterMetrics,
    create_api_rate_limiter,
    create_download_rate_limiter
)


class MockTimeFunction:
    """Mock time function for deterministic testing."""
    
    def __init__(self, initial_time: float = 0.0):
        self.current_time = initial_time
        self._lock = threading.Lock()
    
    def __call__(self) -> float:
        with self._lock:
            return self.current_time
    
    def advance(self, delta: float) -> None:
        with self._lock:
            self.current_time += delta
    
    def set_time(self, new_time: float) -> None:
        with self._lock:
            self.current_time = new_time


class TestAtomicFloat(unittest.TestCase):
    """Test atomic float operations."""
    
    def test_basic_operations(self):
        """Test basic atomic float operations."""
        atomic = AtomicFloat(10.0)
        
        self.assertEqual(atomic.get(), 10.0)
        
        atomic.set(20.0)
        self.assertEqual(atomic.get(), 20.0)
        
        result = atomic.add_and_get(5.0)
        self.assertEqual(result, 25.0)
        self.assertEqual(atomic.get(), 25.0)
        
        result = atomic.subtract_and_get(10.0)
        self.assertEqual(result, 15.0)
        self.assertEqual(atomic.get(), 15.0)
    
    def test_compare_and_swap(self):
        """Test atomic compare-and-swap operations."""
        atomic = AtomicFloat(10.0)
        
        # Successful swap
        self.assertTrue(atomic.compare_and_swap(10.0, 20.0))
        self.assertEqual(atomic.get(), 20.0)
        
        # Failed swap
        self.assertFalse(atomic.compare_and_swap(10.0, 30.0))
        self.assertEqual(atomic.get(), 20.0)
    
    def test_concurrent_operations(self):
        """Test atomic operations under concurrent access."""
        atomic = AtomicFloat(0.0)
        num_threads = 10
        operations_per_thread = 100
        
        def worker():
            for _ in range(operations_per_thread):
                atomic.add_and_get(1.0)
        
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        expected = num_threads * operations_per_thread
        self.assertEqual(atomic.get(), expected)


class TestHighPerformanceTokenBucket(unittest.TestCase):
    """Test high-performance token bucket implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_time = MockTimeFunction(0.0)
    
    def test_initialization(self):
        """Test proper initialization."""
        bucket = HighPerformanceTokenBucket(
            capacity=100, 
            refill_rate=10, 
            time_func=self.mock_time
        )
        
        self.assertEqual(bucket.capacity, 100)
        self.assertEqual(bucket.refill_rate, 10)
        self.assertEqual(bucket.current_tokens, 100)
    
    def test_initialization_validation(self):
        """Test initialization parameter validation."""
        with self.assertRaises(ValueError):
            HighPerformanceTokenBucket(0, 10)
        
        with self.assertRaises(ValueError):
            HighPerformanceTokenBucket(10, 0)
        
        with self.assertRaises(ValueError):
            HighPerformanceTokenBucket(-10, 10)
    
    def test_basic_consumption(self):
        """Test basic token consumption."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=1, 
            time_func=self.mock_time
        )
        
        # Should consume successfully
        self.assertTrue(bucket.try_consume(5))
        self.assertEqual(bucket.current_tokens, 5)
        
        # Should consume remaining tokens
        self.assertTrue(bucket.try_consume(5))
        self.assertEqual(bucket.current_tokens, 0)
        
        # Should fail when no tokens left
        self.assertFalse(bucket.try_consume(1))
        self.assertEqual(bucket.current_tokens, 0)
    
    def test_token_refill(self):
        """Test token refill mechanism."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=self.mock_time
        )
        
        # Consume all tokens
        bucket.try_consume(10)
        self.assertEqual(bucket.current_tokens, 0)
        
        # Advance time by 1 second (should add 5 tokens)
        self.mock_time.advance(1.0)
        self.assertEqual(bucket.current_tokens, 5)
        
        # Advance time by another second (should reach capacity)
        self.mock_time.advance(1.0)
        self.assertEqual(bucket.current_tokens, 10)
        
        # Advance more time (should not exceed capacity)
        self.mock_time.advance(2.0)
        self.assertEqual(bucket.current_tokens, 10)
    
    def test_burst_allowance(self):
        """Test burst allowance functionality."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5,
            burst_allowance=2.0,
            time_func=self.mock_time
        )
        
        # Should be able to consume up to burst capacity
        self.assertTrue(bucket.try_consume(20))  # 10 * 2.0 = 20
        self.assertEqual(bucket.current_tokens, 0)
    
    def test_fractional_tokens(self):
        """Test handling of fractional token amounts."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=2.5, 
            time_func=self.mock_time
        )
        
        # Consume fractional tokens
        self.assertTrue(bucket.try_consume(1.5))
        self.assertEqual(bucket.current_tokens, 8.5)
        
        # Refill fractional tokens
        self.mock_time.advance(0.4)  # 0.4 * 2.5 = 1.0 token
        self.assertAlmostEqual(bucket.current_tokens, 9.5, places=6)
    
    def test_zero_token_consumption(self):
        """Test edge case of zero token consumption."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=self.mock_time
        )
        
        # Zero token consumption should always succeed
        self.assertTrue(bucket.try_consume(0))
        self.assertTrue(bucket.try_consume(0.0))
        self.assertEqual(bucket.current_tokens, 10)
    
    def test_negative_token_consumption(self):
        """Test edge case of negative token consumption."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=self.mock_time
        )
        
        # Negative token consumption should always succeed
        self.assertTrue(bucket.try_consume(-1))
        self.assertTrue(bucket.try_consume(-5.5))
        self.assertEqual(bucket.current_tokens, 10)
    
    def test_concurrent_consumption(self):
        """Test thread-safe concurrent token consumption."""
        bucket = HighPerformanceTokenBucket(capacity=1000, refill_rate=1000)
        
        num_threads = 10
        tokens_per_thread = 50
        successful_consumptions = [0] * num_threads
        
        def worker(thread_id: int):
            for _ in range(tokens_per_thread):
                if bucket.try_consume(1):
                    successful_consumptions[thread_id] += 1
        
        threads = [
            threading.Thread(target=worker, args=(i,)) 
            for i in range(num_threads)
        ]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_successful = sum(successful_consumptions)
        
        # Should consume exactly the initial capacity
        self.assertLessEqual(total_successful, 1000)
        self.assertGreater(total_successful, 900)  # Allow for some timing variance
        
        # Test completed quickly (performance check)
        self.assertLess(time.time() - start_time, 1.0)
    
    def test_high_frequency_consumption(self):
        """Test performance under high-frequency consumption."""
        bucket = HighPerformanceTokenBucket(capacity=10000, refill_rate=10000)
        
        start_time = time.perf_counter()
        
        for _ in range(10000):
            bucket.try_consume(1)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should handle 10k operations in well under 1 second
        self.assertLess(total_time, 0.5)
        
        # Calculate operations per second
        ops_per_second = 10000 / total_time
        self.assertGreater(ops_per_second, 20000)  # Should exceed 20k ops/sec
    
    async def test_async_consumption(self):
        """Test async consumption with backpressure."""
        bucket = HighPerformanceTokenBucket(
            capacity=5, 
            refill_rate=2, 
            time_func=self.mock_time
        )
        
        # Consume all tokens
        bucket.try_consume(5)
        
        # Should wait and eventually succeed
        self.mock_time.advance(3.0)  # Add 6 tokens over time
        result = await bucket.consume(5, max_wait_time=1.0)
        self.assertTrue(result)
    
    async def test_async_timeout(self):
        """Test async consumption timeout."""
        bucket = HighPerformanceTokenBucket(
            capacity=5, 
            refill_rate=1, 
            time_func=self.mock_time
        )
        
        # Consume all tokens
        bucket.try_consume(5)
        
        # Should timeout when not enough tokens available
        result = await bucket.consume(10, max_wait_time=0.1)
        self.assertFalse(result)
    
    def test_metrics_tracking(self):
        """Test metrics collection and accuracy."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=self.mock_time
        )
        
        # Test successful consumption metrics
        bucket.try_consume(3)
        bucket.try_consume(2)
        
        # Test failed consumption metrics
        bucket.try_consume(10)  # Should fail
        bucket.try_consume(8)   # Should fail
        
        metrics = bucket.get_metrics()
        
        self.assertEqual(metrics.total_requests, 4)
        self.assertEqual(metrics.allowed_requests, 2)
        self.assertEqual(metrics.rejected_requests, 2)
        self.assertEqual(metrics.success_rate, 0.5)
        self.assertEqual(metrics.rejection_rate, 0.5)
        self.assertGreater(metrics.avg_processing_time_ns, 0)
        self.assertGreater(metrics.peak_processing_time_ns, 0)
    
    def test_reset_functionality(self):
        """Test rate limiter reset."""
        bucket = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=self.mock_time
        )
        
        # Consume tokens and generate metrics
        bucket.try_consume(5)
        bucket.try_consume(10)  # Should fail
        
        # Reset should restore initial state
        bucket.reset()
        
        self.assertEqual(bucket.current_tokens, 10)
        
        metrics = bucket.get_metrics()
        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.allowed_requests, 0)
        self.assertEqual(metrics.rejected_requests, 0)
    
    def test_race_condition_handling(self):
        """Test handling of race conditions in concurrent access."""
        bucket = HighPerformanceTokenBucket(capacity=100, refill_rate=100)
        
        results = []
        
        def aggressive_consumer():
            local_results = []
            for _ in range(200):  # Try to consume more than capacity
                local_results.append(bucket.try_consume(1))
            results.extend(local_results)
        
        # Launch multiple aggressive consumers
        threads = [threading.Thread(target=aggressive_consumer) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have consumed more tokens than available
        successful_consumptions = sum(results)
        self.assertLessEqual(successful_consumptions, 100)
        
        # Should not have negative tokens
        self.assertGreaterEqual(bucket.current_tokens, 0)


class TestMultiKeyRateLimiter(unittest.TestCase):
    """Test multi-key rate limiter functionality."""
    
    def test_basic_multi_key_operations(self):
        """Test basic operations with multiple keys."""
        limiter = MultiKeyRateLimiter(default_capacity=10, default_refill_rate=5)
        
        # Different keys should have independent buckets
        self.assertTrue(limiter.try_consume("user1", 5))
        self.assertTrue(limiter.try_consume("user2", 5))
        self.assertTrue(limiter.try_consume("user1", 5))
        self.assertTrue(limiter.try_consume("user2", 5))
        
        # Both should now be at capacity
        self.assertFalse(limiter.try_consume("user1", 1))
        self.assertFalse(limiter.try_consume("user2", 1))
    
    def test_custom_limiter_creation(self):
        """Test creating limiters with custom parameters."""
        limiter = MultiKeyRateLimiter(default_capacity=10, default_refill_rate=5)
        
        # Create custom limiter for specific key
        custom_limiter = limiter.get_limiter("premium_user", capacity=100, refill_rate=50)
        
        self.assertEqual(custom_limiter.capacity, 100)
        self.assertEqual(custom_limiter.refill_rate, 50)
        
        # Should be able to consume more tokens
        self.assertTrue(limiter.try_consume("premium_user", 50))
        
        # Regular user should still be limited
        self.assertFalse(limiter.try_consume("regular_user", 50))
    
    def test_limiter_cleanup(self):
        """Test automatic cleanup of unused limiters."""
        limiter = MultiKeyRateLimiter(default_capacity=10, default_refill_rate=5)
        limiter._cleanup_interval = 0.1  # Short interval for testing
        
        # Create multiple limiters
        for i in range(10):
            limiter.try_consume(f"user{i}", 1)
        
        self.assertEqual(len(limiter._limiters), 10)
        
        # Wait for cleanup
        time.sleep(0.3)
        
        # Access one limiter to trigger cleanup
        limiter.try_consume("new_user", 1)
        
        # Old limiters should be cleaned up (may vary based on timing)
        self.assertLessEqual(len(limiter._limiters), 10)
    
    async def test_async_multi_key(self):
        """Test async operations with multiple keys."""
        limiter = MultiKeyRateLimiter(default_capacity=5, default_refill_rate=10)
        
        # Consume all tokens for one key
        for _ in range(5):
            self.assertTrue(limiter.try_consume("user1", 1))
        
        # Should timeout for more consumption
        result = await limiter.consume("user1", 1, max_wait_time=0.1)
        self.assertFalse(result)
        
        # Other key should still work
        result = await limiter.consume("user2", 1, max_wait_time=0.1)
        self.assertTrue(result)
    
    def test_metrics_per_key(self):
        """Test metrics collection per key."""
        limiter = MultiKeyRateLimiter(default_capacity=10, default_refill_rate=5)
        
        # Generate different patterns for different keys
        limiter.try_consume("user1", 3)
        limiter.try_consume("user1", 2)
        limiter.try_consume("user2", 1)
        limiter.try_consume("user2", 15)  # Should fail
        
        metrics1 = limiter.get_metrics("user1")
        metrics2 = limiter.get_metrics("user2")
        
        self.assertEqual(metrics1.total_requests, 2)
        self.assertEqual(metrics1.allowed_requests, 2)
        self.assertEqual(metrics1.rejected_requests, 0)
        
        self.assertEqual(metrics2.total_requests, 2)
        self.assertEqual(metrics2.allowed_requests, 1)
        self.assertEqual(metrics2.rejected_requests, 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test high-level convenience functions."""
    
    def test_api_rate_limiter_creation(self):
        """Test API rate limiter creation."""
        limiter = create_api_rate_limiter(requests_per_second=100)
        
        self.assertEqual(limiter.capacity, 100)
        self.assertEqual(limiter.refill_rate, 100)
        
        # Should handle burst traffic
        burst_limiter = create_api_rate_limiter(requests_per_second=100, burst_capacity=2.0)
        self.assertTrue(burst_limiter.try_consume(200))  # 100 * 2.0
    
    def test_download_rate_limiter_creation(self):
        """Test download rate limiter creation."""
        limiter = create_download_rate_limiter(bytes_per_second=1024*1024)  # 1MB/s
        
        self.assertEqual(limiter.capacity, 1024*1024)
        self.assertEqual(limiter.refill_rate, 1024*1024)
        
        # Should handle file chunks
        self.assertTrue(limiter.try_consume(64*1024))  # 64KB chunk


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def test_throughput_benchmark(self):
        """Test throughput under various conditions."""
        limiter = HighPerformanceTokenBucket(capacity=50000, refill_rate=50000)
        
        # Single-threaded throughput
        start_time = time.perf_counter()
        operations = 10000
        
        for _ in range(operations):
            limiter.try_consume(1)
        
        single_thread_time = time.perf_counter() - start_time
        single_thread_ops_per_sec = operations / single_thread_time
        
        # Should achieve high single-thread performance
        self.assertGreater(single_thread_ops_per_sec, 50000)
        
        # Multi-threaded throughput
        limiter.reset()
        
        def worker(ops_per_worker: int):
            for _ in range(ops_per_worker):
                limiter.try_consume(1)
        
        num_threads = 4
        ops_per_worker = 2500
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, ops_per_worker) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        multi_thread_time = time.perf_counter() - start_time
        multi_thread_ops_per_sec = (num_threads * ops_per_worker) / multi_thread_time
        
        # Multi-threaded should scale well
        self.assertGreater(multi_thread_ops_per_sec, 30000)
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import sys
        
        # Measure baseline memory
        baseline = sys.getsizeof(object())
        
        # Create rate limiter and measure memory
        limiter = HighPerformanceTokenBucket(capacity=1000, refill_rate=1000)
        limiter_size = sys.getsizeof(limiter)
        
        # Should be memory efficient (under 1KB per limiter)
        self.assertLess(limiter_size - baseline, 1024)
        
        # Test multi-key limiter efficiency
        multi_limiter = MultiKeyRateLimiter(default_capacity=100, default_refill_rate=100)
        
        # Create 100 limiters
        for i in range(100):
            multi_limiter.try_consume(f"user{i}", 1)
        
        # Should still be efficient
        multi_limiter_size = sys.getsizeof(multi_limiter)
        self.assertLess(multi_limiter_size / 100, 2048)  # Under 2KB per managed limiter
    
    def test_latency_consistency(self):
        """Test latency consistency under load."""
        limiter = HighPerformanceTokenBucket(capacity=10000, refill_rate=10000)
        
        latencies = []
        operations = 1000
        
        for _ in range(operations):
            start = time.perf_counter_ns()
            limiter.try_consume(1)
            end = time.perf_counter_ns()
            latencies.append(end - start)
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        
        # Should have consistent low latencies
        self.assertLess(avg_latency / 1000, 10)  # Under 10μs average
        self.assertLess(p99_latency / 1000, 50)  # Under 50μs P99


class TestEdgeCases(unittest.TestCase):
    """Test various edge cases and error conditions."""
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        # Very large capacity and rate
        large_limiter = HighPerformanceTokenBucket(
            capacity=1e9, 
            refill_rate=1e9
        )
        self.assertTrue(large_limiter.try_consume(1e6))
        
        # Very small capacity and rate
        small_limiter = HighPerformanceTokenBucket(
            capacity=0.001, 
            refill_rate=0.001
        )
        self.assertTrue(small_limiter.try_consume(0.0001))
        self.assertFalse(small_limiter.try_consume(0.01))
    
    def test_time_regression(self):
        """Test behavior when time moves backwards."""
        mock_time = MockTimeFunction(100.0)
        limiter = HighPerformanceTokenBucket(
            capacity=10, 
            refill_rate=5, 
            time_func=mock_time
        )
        
        # Consume tokens
        limiter.try_consume(5)
        
        # Move time backwards
        mock_time.set_time(50.0)
        
        # Should not refill tokens when time moves backwards
        self.assertEqual(limiter.current_tokens, 5)
        
        # Move time forward again
        mock_time.advance(2.0)  # Now at 52.0, but last refill was at 100.0
        
        # Should handle gracefully
        tokens_before = limiter.current_tokens
        limiter.try_consume(0)  # Trigger refill attempt
        tokens_after = limiter.current_tokens
        
        # Tokens should not decrease
        self.assertGreaterEqual(tokens_after, tokens_before)
    
    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        limiter = HighPerformanceTokenBucket(
            capacity=1.0, 
            refill_rate=0.1,
            time_func=MockTimeFunction(0.0)
        )
        
        # Consume very small amounts
        for _ in range(1000):
            self.assertTrue(limiter.try_consume(0.001))
        
        # Should be empty now
        self.assertLess(limiter.current_tokens, 0.001)
        self.assertFalse(limiter.try_consume(0.001))


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)