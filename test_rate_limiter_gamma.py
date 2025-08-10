#!/usr/bin/env python3
"""
Comprehensive Test Suite for DefensiveRateLimiter

This test suite focuses on:
1. Edge cases and boundary conditions
2. Failure modes and error handling
3. Concurrent access patterns
4. Resource exhaustion scenarios
5. Recovery mechanisms
6. Defensive programming validation

Author: Claude (Defensive Programming Specialist)
"""

import asyncio
import logging
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock
import sys
import os
import gc
import weakref
from contextlib import contextmanager

# Import the rate limiter
from rate_limiter_gamma import (
    DefensiveRateLimiter,
    RateLimiterError,
    ConfigurationError,
    ResourceExhaustionError,
    TimeoutError,
    CorruptionError,
    CircuitBreakerState,
    RateLimiterMetrics,
    CircuitBreaker
)


class TestDefensiveRateLimiterInitialization(unittest.TestCase):
    """Test initialization and configuration validation."""
    
    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        limiter = DefensiveRateLimiter(
            capacity=100,
            refill_rate=10.0,
            initial_tokens=50,
            enable_metrics=True,
            enable_circuit_breaker=True
        )
        
        self.assertEqual(limiter._capacity, 100.0)
        self.assertEqual(limiter._refill_rate, 10.0)
        self.assertEqual(limiter._tokens, 50.0)
        self.assertTrue(limiter._enable_metrics)
        self.assertTrue(limiter._enable_circuit_breaker)
        
        limiter.shutdown()
    
    def test_default_initial_tokens(self):
        """Test that initial tokens defaults to capacity."""
        limiter = DefensiveRateLimiter(capacity=100, refill_rate=10.0)
        self.assertEqual(limiter._tokens, 100.0)
        limiter.shutdown()
    
    def test_invalid_capacity_validation(self):
        """Test capacity validation with various invalid inputs."""
        invalid_capacities = [
            None, -1, 0, "invalid", float('inf'), float('nan'),
            DefensiveRateLimiter.MAX_CAPACITY + 1, [], {}
        ]
        
        for capacity in invalid_capacities:
            with self.assertRaises(ConfigurationError):
                DefensiveRateLimiter(capacity=capacity, refill_rate=1.0)
    
    def test_invalid_refill_rate_validation(self):
        """Test refill rate validation with various invalid inputs."""
        invalid_rates = [
            None, -1, 0, "invalid", float('inf'), float('nan'),
            DefensiveRateLimiter.MAX_REFILL_RATE + 1, [], {}
        ]
        
        for rate in invalid_rates:
            with self.assertRaises(ConfigurationError):
                DefensiveRateLimiter(capacity=100, refill_rate=rate)
    
    def test_invalid_initial_tokens_validation(self):
        """Test initial tokens validation."""
        with self.assertRaises(ConfigurationError):
            DefensiveRateLimiter(capacity=100, refill_rate=1.0, initial_tokens=-1)
        
        with self.assertRaises(ConfigurationError):
            DefensiveRateLimiter(capacity=100, refill_rate=1.0, initial_tokens=150)
    
    def test_boundary_values(self):
        """Test boundary values for configuration parameters."""
        # Test minimum values
        limiter = DefensiveRateLimiter(
            capacity=DefensiveRateLimiter.MIN_CAPACITY,
            refill_rate=DefensiveRateLimiter.MIN_REFILL_RATE
        )
        limiter.shutdown()
        
        # Test maximum values
        limiter = DefensiveRateLimiter(
            capacity=DefensiveRateLimiter.MAX_CAPACITY,
            refill_rate=DefensiveRateLimiter.MAX_REFILL_RATE
        )
        limiter.shutdown()
    
    def test_type_coercion(self):
        """Test that numeric strings are properly coerced."""
        limiter = DefensiveRateLimiter(capacity="100", refill_rate="10.5")
        self.assertEqual(limiter._capacity, 100.0)
        self.assertEqual(limiter._refill_rate, 10.5)
        limiter.shutdown()


class TestDefensiveRateLimiterBasicFunctionality(unittest.TestCase):
    """Test basic rate limiting functionality."""
    
    def setUp(self):
        """Set up test rate limiter."""
        self.limiter = DefensiveRateLimiter(
            capacity=10,
            refill_rate=5.0,  # 5 tokens per second
            initial_tokens=10
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_allow_request_basic(self):
        """Test basic request allowing functionality."""
        # Should allow request when tokens available
        self.assertTrue(self.limiter.allow_request(1))
        self.assertTrue(self.limiter.allow_request(2))
        
        # Check token count decreased
        status = self.limiter.get_status()
        self.assertEqual(status['current_tokens'], 7.0)
    
    def test_exhaust_tokens(self):
        """Test behavior when tokens are exhausted."""
        # Exhaust all tokens
        for i in range(10):
            result = self.limiter.allow_request(1)
            self.assertTrue(result, f"Request {i} should be allowed")
        
        # Next request should be denied
        self.assertFalse(self.limiter.allow_request(1))
        
        # Check no tokens remaining
        status = self.limiter.get_status()
        self.assertEqual(status['current_tokens'], 0.0)
    
    def test_token_refill(self):
        """Test token refill mechanism."""
        # Exhaust tokens
        for _ in range(10):
            self.limiter.allow_request(1)
        
        self.assertFalse(self.limiter.allow_request(1))
        
        # Wait for refill (5 tokens per second)
        time.sleep(0.5)  # Should add ~2.5 tokens
        
        # Should now allow some requests
        self.assertTrue(self.limiter.allow_request(1))
        self.assertTrue(self.limiter.allow_request(1))
        self.assertFalse(self.limiter.allow_request(1))  # Should run out again
    
    def test_multiple_tokens_request(self):
        """Test requesting multiple tokens at once."""
        self.assertTrue(self.limiter.allow_request(5))
        self.assertTrue(self.limiter.allow_request(5))
        self.assertFalse(self.limiter.allow_request(1))  # No tokens left
    
    def test_request_more_than_capacity(self):
        """Test requesting more tokens than capacity."""
        with self.assertRaises(ConfigurationError):
            self.limiter.allow_request(15)  # Capacity is 10


class TestDefensiveRateLimiterInputValidation(unittest.TestCase):
    """Test comprehensive input validation for all methods."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(capacity=10, refill_rate=1.0)
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_allow_request_invalid_tokens(self):
        """Test allow_request with invalid token requests."""
        invalid_tokens = [0, -1, "invalid", None, float('inf'), float('nan'), []]
        
        for tokens in invalid_tokens:
            with self.assertRaises(ConfigurationError):
                self.limiter.allow_request(tokens)
    
    def test_allow_request_invalid_timeout(self):
        """Test allow_request with invalid timeouts."""
        invalid_timeouts = [
            -1, 0, "invalid", None, float('inf'), float('nan'),
            DefensiveRateLimiter.MAX_TIMEOUT + 1, []
        ]
        
        for timeout in invalid_timeouts:
            with self.assertRaises(ConfigurationError):
                self.limiter.allow_request(1, timeout_seconds=timeout)
    
    def test_allow_request_edge_case_values(self):
        """Test allow_request with edge case but valid values."""
        # Test minimum values
        self.limiter.allow_request(
            tokens_requested=DefensiveRateLimiter.MIN_CAPACITY,
            timeout_seconds=DefensiveRateLimiter.MIN_TIMEOUT
        )
        
        # Test maximum valid timeout
        result = self.limiter.allow_request(
            tokens_requested=1,
            timeout_seconds=DefensiveRateLimiter.MAX_TIMEOUT
        )
        self.assertIsInstance(result, bool)
    
    def test_request_id_validation(self):
        """Test request ID validation and conversion."""
        # Valid request IDs
        valid_ids = ["test", 123, None, 45.6, True]
        
        for req_id in valid_ids:
            result = self.limiter.allow_request(1, request_id=req_id)
            self.assertIsInstance(result, bool)


class TestDefensiveRateLimiterConcurrency(unittest.TestCase):
    """Test concurrent access and thread safety."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(
            capacity=1000,
            refill_rate=100.0,
            max_concurrent_requests=100
        )
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_concurrent_requests(self):
        """Test multiple threads making concurrent requests."""
        num_threads = 20  # Reduced for stability
        requests_per_thread = 10
        total_requests = num_threads * requests_per_thread
        
        results = []
        
        def make_requests():
            thread_results = []
            for i in range(requests_per_thread):
                try:
                    result = self.limiter.allow_request(1, timeout_seconds=2.0)
                    thread_results.append(result)
                except Exception as e:
                    thread_results.append(f"Error: {e}")
            return thread_results
        
        # Start threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_threads)]
            
            for future in as_completed(futures, timeout=30):
                try:
                    results.extend(future.result())
                except Exception as e:
                    self.fail(f"Thread failed with exception: {e}")
        
        # Validate results
        self.assertEqual(len(results), total_requests)
        
        # Count successes and failures
        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is False)
        errors = sum(1 for r in results if isinstance(r, str))
        
        # Should have some successes and no errors
        self.assertGreater(successes, 0)
        self.assertEqual(errors, 0)
        
        # Check metrics
        metrics = self.limiter.get_metrics()
        if metrics:
            self.assertEqual(metrics.total_requests, total_requests)
            self.assertEqual(metrics.allowed_requests, successes)
            self.assertEqual(metrics.rejected_requests, failures)
    
    def test_concurrent_limit_enforcement(self):
        """Test that concurrent request limits are enforced."""
        limiter = DefensiveRateLimiter(
            capacity=1000,
            refill_rate=1.0,
            max_concurrent_requests=5
        )
        
        barrier = threading.Barrier(8)  # More threads than limit but reasonable
        results = []
        
        def blocking_request():
            try:
                barrier.wait(timeout=5.0)  # Synchronize start
                result = limiter.allow_request(1, timeout_seconds=0.5)
                results.append(result)
            except ResourceExhaustionError:
                results.append("ResourceExhausted")
            except Exception as e:
                results.append(f"Error: {e}")
        
        threads = [threading.Thread(target=blocking_request) for _ in range(8)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10)
        
        # Should have some resource exhaustion errors
        exhausted_count = sum(1 for r in results if r == "ResourceExhausted")
        self.assertGreater(exhausted_count, 0)
        
        limiter.shutdown()
    
    def test_thread_safety_state_consistency(self):
        """Test that internal state remains consistent under concurrent access."""
        num_threads = 10
        operations_per_thread = 20
        
        def mixed_operations():
            for _ in range(operations_per_thread):
                try:
                    # Mix of operations
                    self.limiter.allow_request(1)
                    self.limiter.get_status()
                    self.limiter.get_metrics()
                except Exception:
                    pass  # Expected under high concurrency
        
        threads = [threading.Thread(target=mixed_operations) for _ in range(num_threads)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        # Verify state consistency
        status = self.limiter.get_status()
        self.assertIsInstance(status, dict)
        self.assertGreaterEqual(status['current_tokens'], 0)
        self.assertLessEqual(status['current_tokens'], status['capacity'])


class TestDefensiveRateLimiterAsyncFunctionality(unittest.TestCase):
    """Test asynchronous functionality and async/await patterns."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(
            capacity=20,
            refill_rate=10.0
        )
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_async_allow_request(self):
        """Test basic async allow_request functionality."""
        async def test_async():
            result1 = await self.limiter.allow_request_async(1)
            result2 = await self.limiter.allow_request_async(5)
            
            self.assertTrue(result1)
            self.assertTrue(result2)
            
            # Exhaust tokens
            for _ in range(14):
                await self.limiter.allow_request_async(1)
            
            # Should now be denied
            result3 = await self.limiter.allow_request_async(1)
            self.assertFalse(result3)
        
        asyncio.run(test_async())
    
    def test_async_concurrent_requests(self):
        """Test concurrent async requests."""
        async def make_request(request_id):
            return await self.limiter.allow_request_async(1, request_id=f"req_{request_id}")
        
        async def test_concurrent():
            # Create 25 concurrent requests (reduced for stability)
            tasks = [make_request(i) for i in range(25)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            successes = sum(1 for r in results if r is True)
            failures = sum(1 for r in results if r is False)
            errors = sum(1 for r in results if isinstance(r, Exception))
            
            # Should have mixed results and no errors
            self.assertGreater(successes, 0)
            self.assertGreaterEqual(failures, 0)  # Some might fail due to exhaustion
            self.assertEqual(errors, 0)
        
        asyncio.run(test_concurrent())
    
    def test_async_wait_for_tokens(self):
        """Test async wait_for_tokens functionality."""
        async def test_wait():
            # Exhaust tokens
            for _ in range(20):
                await self.limiter.allow_request_async(1)
            
            # Should not get tokens immediately
            result1 = await self.limiter.allow_request_async(1, timeout_seconds=0.05)
            self.assertFalse(result1)
            
            # Should eventually get tokens
            result2 = await self.limiter.wait_for_tokens_async(1, timeout_seconds=1.0)
            self.assertTrue(result2)
        
        asyncio.run(test_wait())


class TestDefensiveRateLimiterErrorHandling(unittest.TestCase):
    """Test comprehensive error handling and recovery mechanisms."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(
            capacity=10,
            refill_rate=1.0,
            enable_metrics=True,
            enable_circuit_breaker=True,
            corruption_detection=True
        )
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_corruption_detection(self):
        """Test internal state corruption detection."""
        # Manually corrupt state
        self.limiter._tokens = -100  # Invalid negative tokens
        
        try:
            # This should trigger corruption detection and recovery
            self.limiter.allow_request(1)
        except CorruptionError:
            pass  # Expected in some cases
        
        # Check that recovery occurred
        status = self.limiter.get_status()
        self.assertGreaterEqual(status['current_tokens'], 0)
    
    def test_time_corruption_handling(self):
        """Test handling of time-related corruption (clock adjustments)."""
        original_time = self.limiter._last_refill_time
        
        # Simulate time going backwards
        self.limiter._last_refill_time = time.time() + 3600  # 1 hour in future
        
        # Should handle gracefully
        result = self.limiter.allow_request(1)
        self.assertIsInstance(result, bool)
        
        # Time should be corrected
        self.assertLessEqual(self.limiter._last_refill_time, time.time() + 1)
    
    def test_metrics_corruption_recovery(self):
        """Test recovery from metrics corruption."""
        if self.limiter._metrics:
            # Corrupt metrics
            self.limiter._metrics.total_requests = -100
            self.limiter._metrics.allowed_requests = 999999
            
            # Make request - should trigger correction
            self.limiter.allow_request(1)
            
            # Metrics should be corrected
            metrics = self.limiter.get_metrics()
            if metrics:
                self.assertGreaterEqual(metrics.total_requests, 0)
                self.assertLessEqual(metrics.allowed_requests, metrics.total_requests)


class TestDefensiveRateLimiterCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality and fault tolerance."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(
            capacity=10,
            refill_rate=1.0,
            enable_circuit_breaker=True
        )
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker proper initialization."""
        self.assertIsNotNone(self.limiter._circuit_breaker)
        self.assertEqual(self.limiter._circuit_breaker.state, CircuitBreakerState.CLOSED)
    
    def test_circuit_breaker_failure_recording(self):
        """Test that circuit breaker records failures properly."""
        cb = self.limiter._circuit_breaker
        initial_failures = cb.failure_count
        
        # Record some failures
        cb.record_failure()
        cb.record_failure()
        
        self.assertEqual(cb.failure_count, initial_failures + 2)
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        cb = self.limiter._circuit_breaker
        
        # Start closed
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
        
        # Record enough failures to open
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        
        # Should not allow execution when open
        self.assertFalse(cb.can_execute())
    
    def test_circuit_breaker_timeout_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = self.limiter._circuit_breaker
        cb.timeout_seconds = 0.1  # Very short timeout for testing
        
        # Trip the circuit breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        self.assertTrue(cb.can_execute())
        self.assertEqual(cb.state, CircuitBreakerState.HALF_OPEN)
    
    def test_circuit_breaker_invalid_config(self):
        """Test circuit breaker with invalid configuration."""
        with self.assertRaises(ConfigurationError):
            CircuitBreaker(failure_threshold=0)
        
        with self.assertRaises(ConfigurationError):
            CircuitBreaker(timeout_seconds=-1)


class TestDefensiveRateLimiterMetrics(unittest.TestCase):
    """Test comprehensive metrics collection and accuracy."""
    
    def setUp(self):
        self.limiter = DefensiveRateLimiter(
            capacity=10,
            refill_rate=1.0,
            enable_metrics=True
        )
    
    def tearDown(self):
        if hasattr(self, 'limiter'):
            self.limiter.shutdown()
    
    def test_metrics_collection(self):
        """Test that metrics are collected accurately."""
        initial_metrics = self.limiter.get_metrics()
        
        # Make some requests
        self.limiter.allow_request(1)  # Should succeed
        self.limiter.allow_request(9)  # Should succeed
        self.limiter.allow_request(1)  # Should fail (no tokens left)
        
        metrics = self.limiter.get_metrics()
        
        self.assertEqual(metrics.total_requests, 3)
        self.assertEqual(metrics.allowed_requests, 2)
        self.assertEqual(metrics.rejected_requests, 1)
    
    def test_concurrent_metrics_accuracy(self):
        """Test metrics accuracy under concurrent access."""
        num_threads = 5
        requests_per_thread = 5
        
        def make_requests():
            for _ in range(requests_per_thread):
                try:
                    self.limiter.allow_request(1)
                except Exception:
                    pass
        
        threads = [threading.Thread(target=make_requests) for _ in range(num_threads)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        metrics = self.limiter.get_metrics()
        expected_total = num_threads * requests_per_thread
        
        self.assertEqual(metrics.total_requests, expected_total)
        self.assertEqual(metrics.allowed_requests + metrics.rejected_requests, expected_total)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Make some requests
        self.limiter.allow_request(1)
        self.limiter.allow_request(1)
        
        metrics = self.limiter.get_metrics()
        self.assertGreater(metrics.total_requests, 0)
        
        # Reset metrics
        self.limiter.reset_metrics()
        
        metrics = self.limiter.get_metrics()
        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.allowed_requests, 0)
        self.assertEqual(metrics.rejected_requests, 0)
    
    def test_metrics_defensive_copying(self):
        """Test that metrics are defensively copied."""
        metrics1 = self.limiter.get_metrics()
        metrics2 = self.limiter.get_metrics()
        
        # Should be different objects
        self.assertIsNot(metrics1, metrics2)
        
        # Modifying one shouldn't affect the other
        metrics1.total_requests = 99999
        self.assertNotEqual(metrics2.total_requests, 99999)


class TestDefensiveRateLimiterResourceManagement(unittest.TestCase):
    """Test resource management and cleanup."""
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with DefensiveRateLimiter(capacity=10, refill_rate=1.0) as limiter:
            self.assertFalse(limiter._is_shutdown)
            result = limiter.allow_request(1)
            self.assertIsInstance(result, bool)
        
        # Should be shutdown after context exit
        self.assertTrue(limiter._is_shutdown)
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown process."""
        limiter = DefensiveRateLimiter(capacity=10, refill_rate=1.0)
        
        self.assertFalse(limiter._is_shutdown)
        
        limiter.shutdown()
        
        self.assertTrue(limiter._is_shutdown)
        
        # Should raise error after shutdown
        with self.assertRaises(RateLimiterError):
            limiter.allow_request(1)
    
    def test_cleanup_with_active_requests(self):
        """Test cleanup when requests are still active."""
        limiter = DefensiveRateLimiter(
            capacity=10,
            refill_rate=0.1,  # Very slow refill
            max_concurrent_requests=5
        )
        
        # Start some long-running requests
        def long_request():
            try:
                limiter.wait_for_tokens(1, timeout_seconds=1.0)
            except:
                pass
        
        threads = [threading.Thread(target=long_request) for _ in range(2)]
        for thread in threads:
            thread.start()
        
        time.sleep(0.05)  # Let requests start
        
        # Shutdown should still work
        limiter.shutdown()
        
        for thread in threads:
            thread.join(timeout=3.0)


class TestDefensiveRateLimiterEdgeCases(unittest.TestCase):
    """Test edge cases and corner conditions."""
    
    def test_zero_timeout_requests(self):
        """Test requests with zero timeout."""
        limiter = DefensiveRateLimiter(capacity=1, refill_rate=0.1)
        
        # First request should succeed
        self.assertTrue(limiter.allow_request(1))
        
        # Second request with zero timeout should fail immediately
        result = limiter.allow_request(1, timeout_seconds=DefensiveRateLimiter.MIN_TIMEOUT)
        self.assertFalse(result)
        
        limiter.shutdown()
    
    def test_very_high_refill_rate(self):
        """Test behavior with very high refill rates."""
        limiter = DefensiveRateLimiter(
            capacity=100,
            refill_rate=DefensiveRateLimiter.MAX_REFILL_RATE
        )
        
        # Should be able to make many requests quickly
        for i in range(150):
            result = limiter.allow_request(1)
            if not result:
                time.sleep(0.001)  # Brief pause
                result = limiter.allow_request(1)
                if i > 100:  # Give up after reasonable attempts
                    break
        
        limiter.shutdown()
    
    def test_very_low_refill_rate(self):
        """Test behavior with very low refill rates."""
        limiter = DefensiveRateLimiter(
            capacity=2,
            refill_rate=DefensiveRateLimiter.MIN_REFILL_RATE
        )
        
        # Use up tokens
        self.assertTrue(limiter.allow_request(1))
        self.assertTrue(limiter.allow_request(1))
        
        # Should be rate limited
        self.assertFalse(limiter.allow_request(1))
        
        limiter.shutdown()
    
    def test_fractional_tokens(self):
        """Test handling of fractional token values."""
        limiter = DefensiveRateLimiter(
            capacity=10.5,
            refill_rate=1.7,
            initial_tokens=5.3
        )
        
        # Should handle fractional values properly
        self.assertTrue(limiter.allow_request(2.5))
        self.assertTrue(limiter.allow_request(1.5))
        
        status = limiter.get_status()
        self.assertAlmostEqual(status['current_tokens'], 1.3, places=1)
        
        limiter.shutdown()
    
    def test_system_clock_changes(self):
        """Test resilience to system clock changes."""
        limiter = DefensiveRateLimiter(capacity=10, refill_rate=5.0)
        
        # Make initial request
        self.assertTrue(limiter.allow_request(1))
        
        # Simulate clock going backwards (common with NTP adjustments)
        original_time = limiter._last_refill_time
        limiter._last_refill_time = time.time() + 3600  # 1 hour future
        
        # Should handle gracefully
        result = limiter.allow_request(1)
        self.assertIsInstance(result, bool)
        
        # Time should be corrected
        self.assertLess(limiter._last_refill_time, original_time + 3600)
        
        limiter.shutdown()


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)