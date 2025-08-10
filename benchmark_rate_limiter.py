#!/usr/bin/env python3
"""
Performance Benchmark for DefensiveRateLimiter

Tests performance characteristics under high load to verify 10,000+ RPS capability.
"""

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from rate_limiter_gamma import DefensiveRateLimiter


def benchmark_sequential_performance():
    """Test sequential performance."""
    print("=== Sequential Performance Benchmark ===")
    
    limiter = DefensiveRateLimiter(
        capacity=50000,
        refill_rate=20000.0,  # High refill rate for sustained load
        enable_metrics=True
    )
    
    num_requests = 10000
    start_time = time.time()
    
    success_count = 0
    for i in range(num_requests):
        try:
            if limiter.allow_request(1, timeout_seconds=0.001):
                success_count += 1
        except Exception as e:
            print(f"Error at request {i}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    rps = num_requests / duration
    
    print(f"Sequential Performance:")
    print(f"  Requests: {num_requests}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  RPS: {rps:.0f}")
    print(f"  Success Rate: {(success_count/num_requests)*100:.1f}%")
    
    metrics = limiter.get_metrics()
    if metrics:
        print(f"  Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
        print(f"  Total Errors: {metrics.errors_handled}")
    
    limiter.shutdown()
    return rps


def benchmark_concurrent_performance():
    """Test concurrent performance."""
    print("\n=== Concurrent Performance Benchmark ===")
    
    limiter = DefensiveRateLimiter(
        capacity=100000,
        refill_rate=50000.0,  # Very high refill rate
        max_concurrent_requests=500,
        enable_metrics=True
    )
    
    num_threads = 20
    requests_per_thread = 500
    total_requests = num_threads * requests_per_thread
    
    results = []
    start_time = time.time()
    
    def thread_worker():
        thread_success = 0
        for _ in range(requests_per_thread):
            try:
                if limiter.allow_request(1, timeout_seconds=0.001):
                    thread_success += 1
            except Exception:
                pass
        return thread_success
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(thread_worker) for _ in range(num_threads)]
        
        for future in as_completed(futures, timeout=60):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Thread failed: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    total_success = sum(results)
    rps = total_requests / duration
    
    print(f"Concurrent Performance:")
    print(f"  Threads: {num_threads}")
    print(f"  Requests per Thread: {requests_per_thread}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  RPS: {rps:.0f}")
    print(f"  Success Rate: {(total_success/total_requests)*100:.1f}%")
    
    metrics = limiter.get_metrics()
    if metrics:
        print(f"  Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
        print(f"  Peak Concurrent: {metrics.peak_concurrent_requests}")
        print(f"  Total Errors: {metrics.errors_handled}")
    
    limiter.shutdown()
    return rps


def benchmark_async_performance():
    """Test async performance."""
    print("\n=== Async Performance Benchmark ===")
    
    limiter = DefensiveRateLimiter(
        capacity=50000,
        refill_rate=30000.0,
        max_concurrent_requests=1000,
        enable_metrics=True
    )
    
    async def async_benchmark():
        num_requests = 5000
        
        async def single_request():
            try:
                return await limiter.allow_request_async(1, timeout_seconds=0.001)
            except Exception:
                return False
        
        start_time = time.time()
        
        # Create concurrent async requests
        tasks = [single_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        success_count = sum(1 for r in results if r is True)
        rps = num_requests / duration
        
        print(f"Async Performance:")
        print(f"  Requests: {num_requests}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  RPS: {rps:.0f}")
        print(f"  Success Rate: {(success_count/num_requests)*100:.1f}%")
        
        metrics = limiter.get_metrics()
        if metrics:
            print(f"  Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
        
        return rps
    
    async_rps = asyncio.run(async_benchmark())
    limiter.shutdown()
    return async_rps


def benchmark_stress_test():
    """Stress test with error conditions."""
    print("\n=== Stress Test with Error Conditions ===")
    
    limiter = DefensiveRateLimiter(
        capacity=1000,  # Small capacity for stress
        refill_rate=500.0,
        max_concurrent_requests=50,
        enable_metrics=True,
        enable_circuit_breaker=True,
        corruption_detection=True
    )
    
    num_threads = 30
    requests_per_thread = 100
    total_requests = num_threads * requests_per_thread
    
    results = []
    errors = []
    start_time = time.time()
    
    def stress_worker():
        worker_success = 0
        worker_errors = 0
        for i in range(requests_per_thread):
            try:
                # Mix of request sizes to create contention
                tokens = 1 if i % 3 == 0 else 2
                if limiter.allow_request(tokens, timeout_seconds=0.01):
                    worker_success += 1
            except Exception as e:
                worker_errors += 1
        return worker_success, worker_errors
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(stress_worker) for _ in range(num_threads)]
        
        for future in as_completed(futures, timeout=60):
            try:
                success, error = future.result()
                results.append(success)
                errors.append(error)
            except Exception as e:
                print(f"Stress thread failed: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    total_success = sum(results)
    total_errors = sum(errors)
    rps = total_requests / duration
    
    print(f"Stress Test Results:")
    print(f"  Threads: {num_threads}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  RPS: {rps:.0f}")
    print(f"  Success Rate: {(total_success/total_requests)*100:.1f}%")
    print(f"  Error Rate: {(total_errors/total_requests)*100:.1f}%")
    
    metrics = limiter.get_metrics()
    if metrics:
        print(f"  Rejected Requests: {metrics.rejected_requests}")
        print(f"  Errors Handled: {metrics.errors_handled}")
        print(f"  Corruption Events: {metrics.corruption_events}")
        print(f"  Recovery Events: {metrics.recovery_events}")
        print(f"  Circuit Breaker Trips: {metrics.circuit_breaker_trips}")
    
    limiter.shutdown()
    return rps


if __name__ == "__main__":
    print("DefensiveRateLimiter Performance Benchmark")
    print("==========================================")
    
    # Run benchmarks
    sequential_rps = benchmark_sequential_performance()
    concurrent_rps = benchmark_concurrent_performance() 
    async_rps = benchmark_async_performance()
    stress_rps = benchmark_stress_test()
    
    print("\n=== SUMMARY ===")
    print(f"Sequential RPS: {sequential_rps:.0f}")
    print(f"Concurrent RPS: {concurrent_rps:.0f}")
    print(f"Async RPS: {async_rps:.0f}")
    print(f"Stress Test RPS: {stress_rps:.0f}")
    print(f"Peak Performance: {max(sequential_rps, concurrent_rps, async_rps):.0f} RPS")
    
    if max(sequential_rps, concurrent_rps, async_rps) >= 10000:
        print("✅ SUCCESS: 10,000+ RPS requirement met!")
    else:
        print("⚠️  WARNING: Performance below 10,000 RPS requirement")
    
    print("\nBenchmark completed successfully.")