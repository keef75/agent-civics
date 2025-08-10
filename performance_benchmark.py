#\!/usr/bin/env python3
"""
Performance Benchmark Suite for Rate Limiters
Measures throughput, latency, memory usage, and concurrent performance
"""

import time
import threading
import asyncio
import concurrent.futures
import statistics
import gc
import sys
from typing import List, Dict, Any
import tracemalloc

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_implementation(self, impl_module, impl_name: str) -> Dict[str, Any]:
        """Comprehensive performance benchmark"""
        print(f"\n{'='*60}")
        print(f"Performance Benchmark: {impl_name}")
        print(f"{'='*60}")
        
        results = {
            "implementation": impl_name,
            "throughput": {},
            "latency": {},
            "concurrency": {},
            "memory": {},
            "errors": []
        }
        
        try:
            # Create appropriate limiter
            if hasattr(impl_module, 'TokenBucketRateLimiter'):
                limiter = impl_module.TokenBucketRateLimiter(capacity=50000, refill_rate=25000)
                method_name = 'allow'
            elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                limiter = impl_module.HighPerformanceTokenBucket(capacity=50000, refill_rate=25000)
                method_name = 'try_consume'
            elif hasattr(impl_module, 'DefensiveRateLimiter'):
                limiter = impl_module.DefensiveRateLimiter(capacity=50000, refill_rate=25000)
                method_name = 'allow_request'
            else:
                raise Exception("No compatible limiter found")
            
            # Benchmark throughput
            results["throughput"] = self._benchmark_throughput(limiter, method_name, impl_name)
            
            # Benchmark latency distribution
            results["latency"] = self._benchmark_latency(limiter, method_name, impl_name)
            
            # Benchmark concurrent performance
            results["concurrency"] = self._benchmark_concurrency(limiter, method_name, impl_name)
            
            # Benchmark memory usage
            results["memory"] = self._benchmark_memory(impl_module, impl_name)
            
            # Cleanup
            if hasattr(limiter, 'shutdown'):
                limiter.shutdown()
                
        except Exception as e:
            results["errors"].append(f"Benchmark failed: {str(e)}")
            print(f"❌ Benchmark failed: {e}")
        
        return results
    
    def _benchmark_throughput(self, limiter, method_name: str, impl_name: str) -> Dict[str, Any]:
        """Measure peak throughput"""
        print("  📊 Testing throughput...")
        
        operations = 10000
        start_time = time.perf_counter()
        
        successful = 0
        for _ in range(operations):
            try:
                if method_name == 'allow':
                    result = limiter.allow(1)
                elif method_name == 'try_consume':
                    result = limiter.try_consume(1)
                else:  # allow_request
                    result = limiter.allow_request(1)
                
                if result:
                    successful += 1
            except:
                pass
        
        elapsed = time.perf_counter() - start_time
        throughput = operations / elapsed if elapsed > 0 else 0
        
        result = {
            "operations": operations,
            "successful": successful,
            "elapsed_seconds": elapsed,
            "throughput_ops_per_sec": throughput,
            "success_rate": successful / operations
        }
        
        print(f"    Throughput: {throughput:,.0f} ops/sec")
        print(f"    Success rate: {result['success_rate']:.1%}")
        
        return result
    
    def _benchmark_latency(self, limiter, method_name: str, impl_name: str) -> Dict[str, Any]:
        """Measure latency distribution"""
        print("  ⏱️  Testing latency distribution...")
        
        operations = 5000
        latencies = []
        
        for _ in range(operations):
            start = time.perf_counter_ns()
            try:
                if method_name == 'allow':
                    limiter.allow(1)
                elif method_name == 'try_consume':
                    limiter.try_consume(1)
                else:  # allow_request
                    limiter.allow_request(1)
            except:
                pass
            end = time.perf_counter_ns()
            latencies.append(end - start)
        
        # Calculate statistics
        avg_ns = statistics.mean(latencies)
        median_ns = statistics.median(latencies)
        min_ns = min(latencies)
        max_ns = max(latencies)
        
        # Percentiles
        percentiles = statistics.quantiles(latencies, n=100)
        p95_ns = percentiles[94]
        p99_ns = percentiles[98]
        
        result = {
            "operations": operations,
            "avg_latency_ns": avg_ns,
            "median_latency_ns": median_ns,
            "min_latency_ns": min_ns,
            "max_latency_ns": max_ns,
            "p95_latency_ns": p95_ns,
            "p99_latency_ns": p99_ns,
            "avg_latency_us": avg_ns / 1000,
            "p95_latency_us": p95_ns / 1000,
            "p99_latency_us": p99_ns / 1000
        }
        
        print(f"    Avg latency: {result['avg_latency_us']:.1f}μs")
        print(f"    P95 latency: {result['p95_latency_us']:.1f}μs")
        print(f"    P99 latency: {result['p99_latency_us']:.1f}μs")
        
        return result
    
    def _benchmark_concurrency(self, limiter, method_name: str, impl_name: str) -> Dict[str, Any]:
        """Test concurrent performance scaling"""
        print("  🔀 Testing concurrency scaling...")
        
        results = {}
        
        for num_threads in [1, 2, 4, 8, 16]:
            operations_per_thread = 1000
            total_operations = num_threads * operations_per_thread
            
            completed_ops = []
            def worker():
                local_ops = 0
                for _ in range(operations_per_thread):
                    try:
                        if method_name == 'allow':
                            if limiter.allow(1):
                                local_ops += 1
                        elif method_name == 'try_consume':
                            if limiter.try_consume(1):
                                local_ops += 1
                        else:  # allow_request
                            if limiter.allow_request(1):
                                local_ops += 1
                    except:
                        pass
                completed_ops.append(local_ops)
            
            # Execute concurrent test
            start_time = time.perf_counter()
            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10)
            
            elapsed = time.perf_counter() - start_time
            total_completed = sum(completed_ops)
            throughput = total_completed / elapsed if elapsed > 0 else 0
            
            results[f"{num_threads}_threads"] = {
                "threads": num_threads,
                "total_operations": total_operations,
                "completed_operations": total_completed,
                "elapsed_seconds": elapsed,
                "throughput_ops_per_sec": throughput,
                "success_rate": total_completed / total_operations
            }
            
            print(f"    {num_threads:2d} threads: {throughput:8,.0f} ops/sec "
                  f"({total_completed}/{total_operations} success)")
        
        return results
    
    def _benchmark_memory(self, impl_module, impl_name: str) -> Dict[str, Any]:
        """Measure memory usage patterns"""
        print("  💾 Testing memory usage...")
        
        tracemalloc.start()
        
        # Baseline memory
        gc.collect()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_stats = baseline_snapshot.statistics('lineno')
        baseline_memory = sum(stat.size for stat in baseline_stats)
        
        # Create multiple limiters to test memory scaling
        limiters = []
        try:
            for i in range(100):
                if hasattr(impl_module, 'TokenBucketRateLimiter'):
                    limiter = impl_module.TokenBucketRateLimiter(capacity=100, refill_rate=50)
                elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                    limiter = impl_module.HighPerformanceTokenBucket(capacity=100, refill_rate=50)
                elif hasattr(impl_module, 'DefensiveRateLimiter'):
                    limiter = impl_module.DefensiveRateLimiter(capacity=100, refill_rate=50)
                else:
                    break
                limiters.append(limiter)
            
            # Measure peak memory
            gc.collect()
            peak_snapshot = tracemalloc.take_snapshot()
            peak_stats = peak_snapshot.statistics('lineno')
            peak_memory = sum(stat.size for stat in peak_stats)
            
            # Clean up
            for limiter in limiters:
                if hasattr(limiter, 'shutdown'):
                    limiter.shutdown()
            del limiters
            
            # Measure cleanup memory
            gc.collect()
            cleanup_snapshot = tracemalloc.take_snapshot()
            cleanup_stats = cleanup_snapshot.statistics('lineno')
            cleanup_memory = sum(stat.size for stat in cleanup_stats)
            
        except Exception as e:
            peak_memory = baseline_memory
            cleanup_memory = baseline_memory
        
        tracemalloc.stop()
        
        memory_per_limiter = (peak_memory - baseline_memory) / max(len(limiters), 1)
        memory_leak = cleanup_memory - baseline_memory
        
        result = {
            "baseline_memory_bytes": baseline_memory,
            "peak_memory_bytes": peak_memory,
            "cleanup_memory_bytes": cleanup_memory,
            "memory_per_limiter_bytes": memory_per_limiter,
            "potential_leak_bytes": memory_leak,
            "limiters_created": len(limiters) if 'limiters' in locals() else 0
        }
        
        print(f"    Memory per limiter: {memory_per_limiter:,.0f} bytes")
        print(f"    Potential leak: {memory_leak:,.0f} bytes")
        
        return result
    
    def generate_performance_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance comparison report"""
        report = {
            "summary": {},
            "rankings": {},
            "detailed_analysis": {}
        }
        
        # Extract key metrics for ranking
        metrics = {}
        for result in all_results:
            if result["errors"]:
                continue
                
            impl = result["implementation"]
            metrics[impl] = {
                "throughput": result["throughput"].get("throughput_ops_per_sec", 0),
                "avg_latency_us": result["latency"].get("avg_latency_us", float('inf')),
                "p99_latency_us": result["latency"].get("p99_latency_us", float('inf')),
                "memory_per_limiter": result["memory"].get("memory_per_limiter_bytes", float('inf')),
                "max_concurrent_throughput": max(
                    (v["throughput_ops_per_sec"] for v in result["concurrency"].values()),
                    default=0
                )
            }
        
        # Create rankings
        if metrics:
            report["rankings"] = {
                "throughput": sorted(metrics.keys(), 
                                   key=lambda x: metrics[x]["throughput"], reverse=True),
                "latency": sorted(metrics.keys(), 
                                key=lambda x: metrics[x]["avg_latency_us"]),
                "memory_efficiency": sorted(metrics.keys(), 
                                          key=lambda x: metrics[x]["memory_per_limiter"]),
                "concurrency": sorted(metrics.keys(), 
                                    key=lambda x: metrics[x]["max_concurrent_throughput"], reverse=True)
            }
        
        # Calculate overall performance scores
        performance_scores = {}
        for impl in metrics:
            # Normalize scores (0-1 scale)
            throughput_score = min(metrics[impl]["throughput"] / 100000, 1.0)  # Target: 100K ops/sec
            latency_score = max(0, min(1.0, (100 - metrics[impl]["avg_latency_us"]) / 100))  # Target: <100μs
            memory_score = max(0, min(1.0, (10000 - metrics[impl]["memory_per_limiter"]) / 10000))  # Target: <10KB
            concurrency_score = min(metrics[impl]["max_concurrent_throughput"] / 50000, 1.0)  # Target: 50K ops/sec
            
            # Weighted average
            overall_score = (
                throughput_score * 0.3 +
                latency_score * 0.3 +
                concurrency_score * 0.3 +
                memory_score * 0.1
            )
            
            performance_scores[impl] = {
                "overall": overall_score,
                "throughput": throughput_score,
                "latency": latency_score,
                "memory": memory_score,
                "concurrency": concurrency_score
            }
        
        # Find best performer
        if performance_scores:
            best_performer = max(performance_scores.keys(), 
                               key=lambda x: performance_scores[x]["overall"])
            
            report["summary"] = {
                "best_performer": best_performer,
                "best_score": performance_scores[best_performer]["overall"],
                "performance_scores": performance_scores
            }
        
        report["detailed_analysis"] = all_results
        
        return report

def main():
    benchmark = PerformanceBenchmark()
    all_results = []
    
    # Test each implementation
    implementations = [
        ("rate_limiter_baseline", "Baseline"),
        ("rate_limiter_alpha", "Alpha (Correctness-focused)"), 
        ("rate_limiter_beta", "Beta (Performance-focused)"),
        ("rate_limiter_gamma", "Gamma (Defensive programming)")
    ]
    
    for module_name, display_name in implementations:
        try:
            module = __import__(module_name)
            result = benchmark.benchmark_implementation(module, display_name)
            all_results.append(result)
        except ImportError as e:
            print(f"❌ Could not import {module_name}: {e}")
        except Exception as e:
            print(f"❌ Error benchmarking {display_name}: {e}")
    
    # Generate performance report
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON REPORT")
    print(f"{'='*80}")
    
    report = benchmark.generate_performance_report(all_results)
    
    if "summary" in report and report["summary"]:
        summary = report["summary"]
        print(f"\nBest Overall Performer: {summary['best_performer']}")
        print(f"Performance Score: {summary['best_score']:.3f}/1.000")
        
        print(f"\nPerformance Rankings:")
        if "rankings" in report:
            rankings = report["rankings"]
            print(f"  Throughput:     {' > '.join(rankings.get('throughput', []))}")
            print(f"  Latency:        {' > '.join(rankings.get('latency', []))}")
            print(f"  Memory:         {' > '.join(rankings.get('memory_efficiency', []))}")
            print(f"  Concurrency:    {' > '.join(rankings.get('concurrency', []))}")
        
        print(f"\nDetailed Performance Scores:")
        for impl, scores in summary.get("performance_scores", {}).items():
            print(f"  {impl}:")
            print(f"    Overall:     {scores['overall']:.3f}")
            print(f"    Throughput:  {scores['throughput']:.3f}")
            print(f"    Latency:     {scores['latency']:.3f}")
            print(f"    Concurrency: {scores['concurrency']:.3f}")
            print(f"    Memory:      {scores['memory']:.3f}")

if __name__ == "__main__":
    main()
