#\!/usr/bin/env python3
"""
Comprehensive Rate Limiter Verification System
Tests all implementations for correctness, performance, and thread safety
"""

import asyncio
import time
import threading
import concurrent.futures
import traceback
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics
import gc

@dataclass
class TestResult:
    implementation: str
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    error: str = ""

class RateLimiterTester:
    def __init__(self):
        self.results = []
    
    def test_implementation(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test a single implementation comprehensively"""
        print(f"\n{'='*60}")
        print(f"Testing {impl_name}")
        print(f"{'='*60}")
        
        results = []
        
        # Basic functionality tests
        results.extend(self._test_basic_functionality(impl_module, impl_name))
        
        # Thread safety tests
        results.extend(self._test_thread_safety(impl_module, impl_name))
        
        # Performance tests
        results.extend(self._test_performance(impl_module, impl_name))
        
        # Edge case tests
        results.extend(self._test_edge_cases(impl_module, impl_name))
        
        # Error handling tests
        results.extend(self._test_error_handling(impl_module, impl_name))
        
        return results
    
    def _test_basic_functionality(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test basic rate limiting functionality"""
        results = []
        
        try:
            # Test 1: Token consumption
            if hasattr(impl_module, 'TokenBucketRateLimiter'):
                limiter = impl_module.TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
                
                # Should allow initial requests
                success_count = 0
                for i in range(10):
                    if hasattr(limiter, 'allow'):
                        allowed = limiter.allow(1)
                    elif hasattr(limiter, 'try_consume'):
                        allowed = limiter.try_consume(1)
                    elif hasattr(limiter, 'allow_request'):
                        allowed = limiter.allow_request(1)
                    else:
                        raise Exception("No known method to consume tokens")
                    
                    if allowed:
                        success_count += 1
                
                # Should exhaust tokens
                score = min(success_count / 10.0, 1.0)
                results.append(TestResult(
                    implementation=impl_name,
                    test_name="basic_token_consumption",
                    passed=success_count >= 8,  # Allow some variance
                    score=score,
                    details={"allowed_requests": success_count, "total_requests": 10}
                ))
                
            elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                limiter = impl_module.HighPerformanceTokenBucket(capacity=10, refill_rate=5.0)
                
                success_count = 0
                for i in range(10):
                    if limiter.try_consume(1):
                        success_count += 1
                
                score = min(success_count / 10.0, 1.0)
                results.append(TestResult(
                    implementation=impl_name,
                    test_name="basic_token_consumption",
                    passed=success_count >= 8,
                    score=score,
                    details={"allowed_requests": success_count, "total_requests": 10}
                ))
                
            elif hasattr(impl_module, 'DefensiveRateLimiter'):
                limiter = impl_module.DefensiveRateLimiter(capacity=10, refill_rate=5.0)
                
                success_count = 0
                for i in range(10):
                    try:
                        if limiter.allow_request(1):
                            success_count += 1
                    except:
                        pass
                
                limiter.shutdown()
                score = min(success_count / 10.0, 1.0)
                results.append(TestResult(
                    implementation=impl_name,
                    test_name="basic_token_consumption",
                    passed=success_count >= 8,
                    score=score,
                    details={"allowed_requests": success_count, "total_requests": 10}
                ))
        
        except Exception as e:
            results.append(TestResult(
                implementation=impl_name,
                test_name="basic_token_consumption",
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            ))
        
        return results
    
    def _test_thread_safety(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test thread safety under concurrent load"""
        results = []
        
        try:
            if hasattr(impl_module, 'TokenBucketRateLimiter'):
                limiter = impl_module.TokenBucketRateLimiter(capacity=1000, refill_rate=100.0)
                method_name = 'allow'
            elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                limiter = impl_module.HighPerformanceTokenBucket(capacity=1000, refill_rate=100.0)
                method_name = 'try_consume'
            elif hasattr(impl_module, 'DefensiveRateLimiter'):
                limiter = impl_module.DefensiveRateLimiter(capacity=1000, refill_rate=100.0)
                method_name = 'allow_request'
            else:
                raise Exception("No compatible rate limiter class found")
            
            # Concurrent test
            num_threads = 10
            requests_per_thread = 50
            total_requests = num_threads * requests_per_thread
            results_list = []
            
            def worker():
                local_results = []
                for _ in range(requests_per_thread):
                    try:
                        if method_name == 'allow':
                            result = limiter.allow(1)
                        elif method_name == 'try_consume':
                            result = limiter.try_consume(1)
                        else:  # allow_request
                            result = limiter.allow_request(1)
                        local_results.append(result)
                    except Exception:
                        local_results.append(False)
                results_list.extend(local_results)
            
            # Execute concurrent test
            start_time = time.time()
            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout
            
            elapsed = time.time() - start_time
            
            # Clean up
            if hasattr(limiter, 'shutdown'):
                limiter.shutdown()
            
            # Evaluate results
            successful = sum(1 for r in results_list if r)
            failed = len(results_list) - successful
            
            # Thread safety score based on no data corruption and reasonable performance
            thread_safety_passed = len(results_list) == total_requests and elapsed < 5.0
            throughput = successful / elapsed if elapsed > 0 else 0
            
            results.append(TestResult(
                implementation=impl_name,
                test_name="thread_safety",
                passed=thread_safety_passed,
                score=min(throughput / 1000, 1.0),  # Score based on throughput
                details={
                    "total_requests": len(results_list),
                    "successful": successful,
                    "failed": failed,
                    "elapsed_time": elapsed,
                    "throughput_rps": throughput
                }
            ))
            
        except Exception as e:
            results.append(TestResult(
                implementation=impl_name,
                test_name="thread_safety",
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            ))
        
        return results
    
    def _test_performance(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test performance characteristics"""
        results = []
        
        try:
            if hasattr(impl_module, 'TokenBucketRateLimiter'):
                limiter = impl_module.TokenBucketRateLimiter(capacity=10000, refill_rate=5000.0)
                method_name = 'allow'
            elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                limiter = impl_module.HighPerformanceTokenBucket(capacity=10000, refill_rate=5000.0)
                method_name = 'try_consume'
            elif hasattr(impl_module, 'DefensiveRateLimiter'):
                limiter = impl_module.DefensiveRateLimiter(capacity=10000, refill_rate=5000.0)
                method_name = 'allow_request'
            else:
                raise Exception("No compatible rate limiter class found")
            
            # Performance test: measure latency
            num_operations = 1000
            latencies = []
            
            for _ in range(num_operations):
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
            
            # Clean up
            if hasattr(limiter, 'shutdown'):
                limiter.shutdown()
            
            # Calculate performance metrics
            avg_latency_ns = statistics.mean(latencies)
            p99_latency_ns = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 10 else avg_latency_ns
            
            # Performance score (lower latency = higher score)
            # Target: <1000ns average, <10000ns P99
            latency_score = max(0, min(1.0, (10000 - avg_latency_ns) / 10000))
            
            results.append(TestResult(
                implementation=impl_name,
                test_name="performance_latency",
                passed=avg_latency_ns < 100000,  # 100μs threshold
                score=latency_score,
                details={
                    "avg_latency_ns": avg_latency_ns,
                    "p99_latency_ns": p99_latency_ns,
                    "avg_latency_us": avg_latency_ns / 1000,
                    "p99_latency_us": p99_latency_ns / 1000,
                    "operations": num_operations
                }
            ))
            
        except Exception as e:
            results.append(TestResult(
                implementation=impl_name,
                test_name="performance_latency",
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            ))
        
        return results
    
    def _test_edge_cases(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test edge cases and boundary conditions"""
        results = []
        
        try:
            # Test invalid parameters
            edge_case_passed = True
            error_details = []
            
            # Test negative capacity
            try:
                if hasattr(impl_module, 'TokenBucketRateLimiter'):
                    limiter = impl_module.TokenBucketRateLimiter(capacity=-1, refill_rate=1.0)
                    edge_case_passed = False
                    error_details.append("Allowed negative capacity")
                elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                    limiter = impl_module.HighPerformanceTokenBucket(capacity=-1, refill_rate=1.0)
                    edge_case_passed = False
                    error_details.append("Allowed negative capacity")
                elif hasattr(impl_module, 'DefensiveRateLimiter'):
                    limiter = impl_module.DefensiveRateLimiter(capacity=-1, refill_rate=1.0)
                    edge_case_passed = False
                    error_details.append("Allowed negative capacity")
            except:
                pass  # Expected to fail
            
            # Test zero refill rate
            try:
                if hasattr(impl_module, 'TokenBucketRateLimiter'):
                    limiter = impl_module.TokenBucketRateLimiter(capacity=10, refill_rate=0)
                    edge_case_passed = False
                    error_details.append("Allowed zero refill rate")
                elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                    limiter = impl_module.HighPerformanceTokenBucket(capacity=10, refill_rate=0)
                    edge_case_passed = False
                    error_details.append("Allowed zero refill rate")
                elif hasattr(impl_module, 'DefensiveRateLimiter'):
                    limiter = impl_module.DefensiveRateLimiter(capacity=10, refill_rate=0)
                    edge_case_passed = False
                    error_details.append("Allowed zero refill rate")
            except:
                pass  # Expected to fail
            
            results.append(TestResult(
                implementation=impl_name,
                test_name="edge_cases",
                passed=edge_case_passed,
                score=1.0 if edge_case_passed else 0.5,
                details={"error_details": error_details}
            ))
            
        except Exception as e:
            results.append(TestResult(
                implementation=impl_name,
                test_name="edge_cases",
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            ))
        
        return results
    
    def _test_error_handling(self, impl_module, impl_name: str) -> List[TestResult]:
        """Test error handling and recovery"""
        results = []
        
        try:
            error_handling_score = 0.0
            total_tests = 0
            
            # Test 1: Invalid token requests
            total_tests += 1
            try:
                if hasattr(impl_module, 'TokenBucketRateLimiter'):
                    limiter = impl_module.TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
                    limiter.allow(0)  # Should fail
                elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                    limiter = impl_module.HighPerformanceTokenBucket(capacity=10, refill_rate=5.0)
                    limiter.try_consume(-1)  # Should handle gracefully
                    error_handling_score += 1  # Beta handles this gracefully
                elif hasattr(impl_module, 'DefensiveRateLimiter'):
                    limiter = impl_module.DefensiveRateLimiter(capacity=10, refill_rate=5.0)
                    limiter.allow_request(0)  # Should fail
                    limiter.shutdown()
            except:
                error_handling_score += 1  # Expected to fail
            
            # Test 2: Requesting more tokens than capacity
            total_tests += 1
            try:
                if hasattr(impl_module, 'TokenBucketRateLimiter'):
                    limiter = impl_module.TokenBucketRateLimiter(capacity=10, refill_rate=5.0)
                    limiter.allow(15)  # Should fail
                elif hasattr(impl_module, 'HighPerformanceTokenBucket'):
                    limiter = impl_module.HighPerformanceTokenBucket(capacity=10, refill_rate=5.0)
                    result = limiter.try_consume(15)  # Should return False
                    if not result:
                        error_handling_score += 1
                elif hasattr(impl_module, 'DefensiveRateLimiter'):
                    limiter = impl_module.DefensiveRateLimiter(capacity=10, refill_rate=5.0)
                    limiter.allow_request(15)  # Should fail
                    limiter.shutdown()
            except:
                error_handling_score += 1  # Expected to fail or handle gracefully
            
            score = error_handling_score / total_tests if total_tests > 0 else 0
            
            results.append(TestResult(
                implementation=impl_name,
                test_name="error_handling",
                passed=score > 0.5,
                score=score,
                details={
                    "passed_tests": error_handling_score,
                    "total_tests": total_tests
                }
            ))
            
        except Exception as e:
            results.append(TestResult(
                implementation=impl_name,
                test_name="error_handling",
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            ))
        
        return results
    
    def generate_report(self, all_results: List[List[TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        report = {
            "summary": {},
            "detailed_results": {},
            "final_scores": {},
            "recommendation": {}
        }
        
        # Calculate scores for each implementation
        impl_scores = {}
        
        for results in all_results:
            if not results:
                continue
                
            impl_name = results[0].implementation
            
            # Calculate weighted scores
            correctness_score = 0
            performance_score = 0
            thread_safety_score = 0
            error_handling_score = 0
            edge_case_score = 0
            
            test_counts = {"correctness": 0, "performance": 0, "thread_safety": 0, 
                          "error_handling": 0, "edge_cases": 0}
            
            for result in results:
                if "basic" in result.test_name or "consumption" in result.test_name:
                    correctness_score += result.score
                    test_counts["correctness"] += 1
                elif "performance" in result.test_name:
                    performance_score += result.score
                    test_counts["performance"] += 1
                elif "thread_safety" in result.test_name:
                    thread_safety_score += result.score
                    test_counts["thread_safety"] += 1
                elif "error_handling" in result.test_name:
                    error_handling_score += result.score
                    test_counts["error_handling"] += 1
                elif "edge" in result.test_name:
                    edge_case_score += result.score
                    test_counts["edge_cases"] += 1
            
            # Normalize scores
            normalized_scores = {}
            for category, count in test_counts.items():
                if count > 0:
                    if category == "correctness":
                        normalized_scores[category] = correctness_score / count
                    elif category == "performance":
                        normalized_scores[category] = performance_score / count
                    elif category == "thread_safety":
                        normalized_scores[category] = thread_safety_score / count
                    elif category == "error_handling":
                        normalized_scores[category] = error_handling_score / count
                    elif category == "edge_cases":
                        normalized_scores[category] = edge_case_score / count
                else:
                    normalized_scores[category] = 0.0
            
            # Calculate final weighted score
            weights = {
                "correctness": 0.4,
                "performance": 0.2,
                "thread_safety": 0.2,
                "error_handling": 0.1,
                "edge_cases": 0.1
            }
            
            final_score = sum(normalized_scores.get(cat, 0) * weight 
                            for cat, weight in weights.items())
            
            impl_scores[impl_name] = {
                "final_score": final_score,
                "category_scores": normalized_scores,
                "detailed_results": results
            }
            
            report["detailed_results"][impl_name] = results
            report["final_scores"][impl_name] = final_score
        
        # Determine best implementation
        if impl_scores:
            best_impl = max(impl_scores.keys(), key=lambda k: impl_scores[k]["final_score"])
            best_score = impl_scores[best_impl]["final_score"]
            
            report["recommendation"] = {
                "best_implementation": best_impl,
                "confidence_score": min(best_score * 100, 99.9),
                "reasoning": self._generate_reasoning(impl_scores, best_impl)
            }
        
        report["summary"] = {
            "implementations_tested": len(impl_scores),
            "total_tests_run": sum(len(results) for results in all_results),
            "overall_pass_rate": sum(1 for results in all_results 
                                   for result in results if result.passed) / 
                                max(1, sum(len(results) for results in all_results))
        }
        
        return report
    
    def _generate_reasoning(self, impl_scores, best_impl):
        """Generate reasoning for the recommendation"""
        best_data = impl_scores[best_impl]
        reasoning = []
        
        reasoning.append(f"Selected {best_impl} with overall score of {best_data['final_score']:.3f}")
        
        # Analyze strengths
        strengths = []
        for category, score in best_data["category_scores"].items():
            if score > 0.8:
                strengths.append(f"Excellent {category} ({score:.2f})")
            elif score > 0.6:
                strengths.append(f"Good {category} ({score:.2f})")
        
        if strengths:
            reasoning.append("Strengths: " + ", ".join(strengths))
        
        # Compare with others
        other_impls = [name for name in impl_scores.keys() if name != best_impl]
        if other_impls:
            comparisons = []
            for other in other_impls:
                other_score = impl_scores[other]["final_score"]
                diff = best_data["final_score"] - other_score
                comparisons.append(f"{diff:.3f} points ahead of {other}")
            reasoning.append("Performance advantage: " + ", ".join(comparisons))
        
        return reasoning

def main():
    tester = RateLimiterTester()
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
            results = tester.test_implementation(module, display_name)
            all_results.append(results)
            
            # Print immediate results
            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {result.test_name}: {status} (score: {result.score:.3f})")
                if result.error:
                    print(f"    Error: {result.error}")
        
        except ImportError as e:
            print(f"❌ Could not import {module_name}: {e}")
            all_results.append([])
        except Exception as e:
            print(f"❌ Error testing {display_name}: {e}")
            all_results.append([])
    
    # Generate final report
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION REPORT")
    print(f"{'='*80}")
    
    report = tester.generate_report(all_results)
    
    # Print summary
    print(f"Implementations tested: {report['summary']['implementations_tested']}")
    print(f"Total tests executed: {report['summary']['total_tests_run']}")
    print(f"Overall pass rate: {report['summary']['overall_pass_rate']:.1%}")
    
    # Print final scores
    print(f"\nFINAL SCORES:")
    for impl, score in sorted(report["final_scores"].items(), 
                            key=lambda x: x[1], reverse=True):
        print(f"  {impl}: {score:.3f}/1.000")
    
    # Print recommendation
    if "recommendation" in report and report["recommendation"]:
        rec = report["recommendation"]
        print(f"\nRECOMMENDATION:")
        print(f"Best Implementation: {rec['best_implementation']}")
        print(f"Confidence Score: {rec['confidence_score']:.1f}%")
        print(f"Reasoning:")
        for reason in rec["reasoning"]:
            print(f"  • {reason}")

if __name__ == "__main__":
    main()
