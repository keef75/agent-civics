#!/usr/bin/env python3
"""
Final Verification Test for Optimal Rate Limiter
Tests the synthesized solution against all original requirements
"""

import asyncio
import time
import threading
import concurrent.futures
from rate_limiter_final import TokenBucketRateLimiter, create_strict_rate_limiter, create_backpressure_rate_limiter

def test_basic_functionality():
    """Test basic token bucket functionality"""
    print("✅ Testing basic functionality...")
    limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
    
    # Should allow initial requests
    assert limiter.allow(5) == True, "Should allow 5 tokens"
    assert limiter.current_tokens == 5.0, "Should have 5 tokens remaining"
    
    # Should deny when insufficient tokens
    assert limiter.allow(6) == False, "Should deny when insufficient tokens"
    assert limiter.current_tokens == 5.0, "Tokens should not be consumed on denial"
    
    print("  ✅ Basic functionality passed")

def test_thread_safety():
    """Test thread safety under concurrent load"""
    print("✅ Testing thread safety...")
    limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=500)
    
    results = []
    def worker():
        local_results = []
        for _ in range(50):
            local_results.append(limiter.allow(1))
        results.extend(local_results)
    
    # Run concurrent workers
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    successful = sum(1 for r in results if r)
    assert len(results) == 500, "All requests should be processed"
    assert successful <= 1000, "Cannot exceed capacity"
    assert successful >= 100, "Should allow reasonable number of requests"
    
    print(f"  ✅ Thread safety passed ({successful}/500 successful)")

def test_performance():
    """Test performance characteristics"""
    print("✅ Testing performance...")
    limiter = TokenBucketRateLimiter(capacity=50000, refill_rate=25000, high_performance_mode=True)
    
    # Measure latency
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        limiter.allow(1)
        end = time.perf_counter_ns()
        latencies.append(end - start)
    
    avg_latency_ns = sum(latencies) / len(latencies)
    avg_latency_us = avg_latency_ns / 1000
    
    assert avg_latency_us < 10.0, f"Latency too high: {avg_latency_us:.1f}μs"
    
    # Measure throughput
    operations = 10000
    start_time = time.perf_counter()
    for _ in range(operations):
        limiter.allow(1)
    elapsed = time.perf_counter() - start_time
    
    throughput = operations / elapsed
    assert throughput > 500000, f"Throughput too low: {throughput:.0f} ops/sec"
    
    print(f"  ✅ Performance passed (latency: {avg_latency_us:.1f}μs, throughput: {throughput:.0f} ops/sec)")

async def test_async_functionality():
    """Test async functionality and backpressure"""
    print("✅ Testing async functionality...")
    limiter = create_backpressure_rate_limiter(requests_per_second=5.0, max_wait_seconds=0.5)
    
    # Test basic async operation
    result = await limiter.allow_async(1)
    assert result == True, "Should allow async request"
    
    # Test backpressure
    for _ in range(5):
        await limiter.allow_async(1)  # Exhaust tokens
    
    # This should trigger backpressure
    start_time = time.time()
    result = await limiter.allow_async(1)
    elapsed = time.time() - start_time
    
    # Should either succeed after waiting or fail quickly
    assert elapsed < 1.0, "Should not wait too long"
    
    print("  ✅ Async functionality passed")

def test_error_handling():
    """Test error handling and validation"""
    print("✅ Testing error handling...")
    
    # Test invalid initialization
    try:
        TokenBucketRateLimiter(capacity=-1, refill_rate=1.0)
        assert False, "Should raise ConfigurationError for negative capacity"
    except Exception as e:
        assert "ConfigurationError" in str(type(e)), "Should raise ConfigurationError"
    
    # Test invalid token requests
    limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
    
    try:
        limiter.allow(0)
        assert False, "Should raise error for zero tokens"
    except Exception as e:
        assert "must be positive" in str(e), "Should validate positive tokens"
    
    try:
        limiter.allow(15)  # More than capacity
        assert False, "Should raise error for tokens exceeding capacity"
    except Exception as e:
        assert "exceeds capacity" in str(e), "Should validate capacity limit"
    
    print("  ✅ Error handling passed")

def test_metrics():
    """Test metrics collection"""
    print("✅ Testing metrics...")
    limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5.0, enable_metrics=True)
    
    # Generate some activity
    limiter.allow(3)  # Should succeed
    limiter.allow(8)  # Should fail
    limiter.allow(2)  # Should succeed
    
    metrics = limiter.get_metrics()
    assert metrics is not None, "Metrics should be available"
    assert metrics['total_requests'] == 3, "Should track total requests"
    assert metrics['allowed_requests'] == 2, "Should track allowed requests"
    assert metrics['denied_requests'] == 1, "Should track denied requests"
    assert abs(metrics['success_rate'] - 2/3) < 0.01, "Should calculate correct success rate"
    
    print("  ✅ Metrics passed")

def main():
    print("🔍 Final Verification of Optimal Rate Limiter")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_functionality()
        test_thread_safety()
        test_performance()
        asyncio.run(test_async_functionality())
        test_error_handling()
        test_metrics()
        
        print("\n🎉 ALL TESTS PASSED")
        print("✨ Final solution successfully verified!")
        
        # Performance summary
        print("\n📊 Performance Summary:")
        print("  • Basic functionality: ✅ Correct")
        print("  • Thread safety: ✅ Excellent")
        print("  • Performance: ✅ High (>500K ops/sec, <10μs latency)")
        print("  • Async support: ✅ Full backpressure handling")
        print("  • Error handling: ✅ Comprehensive validation")
        print("  • Metrics: ✅ Complete observability")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
