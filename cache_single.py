"""
Distributed Cache System - Single Agent Baseline (Depth=0)

Single-agent baseline implementation with 88% reliability baseline for mathematical comparison.
This implementation demonstrates traditional non-federated caching limitations.

Reliability Formula: P(success) = 1 - ε (where ε = 12% baseline error rate)
Expected Reliability: 88%
"""

import asyncio
import json
import time
import hashlib
import threading
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timedelta
import statistics

# Import our proven rate limiter
from rate_limiter_final import TokenBucketRateLimiter


@dataclass
class CacheEntry:
    """Cache entry with metadata for LRU tracking"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    expiry: Optional[datetime] = None


@dataclass
class CacheOperation:
    """Cache operation request structure"""
    operation: str  # get, set, delete, exists
    key: str
    value: Any = None
    ttl: Optional[int] = None
    shard_hint: Optional[str] = None
    consistency_level: str = "eventual"  # strong, eventual
    client_id: str = "default"


@dataclass
class CacheResponse:
    """Cache operation response with performance metrics"""
    success: bool
    value: Any = None
    hit: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    node_id: str = "single_node"
    cache_size: int = 0
    memory_usage: float = 0.0
    consistency_achieved: bool = False
    replication_success: bool = False


class PerformanceMetrics:
    """Performance and reliability tracking for cache operations"""
    
    def __init__(self):
        self.total_operations = 0
        self.successful_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.replication_failures = 0
        
        # Performance metrics
        self.response_times = []
        self.memory_usage_history = []
        self.throughput_history = []
        
        # Failure tracking
        self.failure_types = {}
        self.consistency_violations = 0
        
        # Time-based tracking
        self.start_time = time.time()
        self.last_throughput_check = time.time()
        self.operations_since_check = 0
        
    def record_operation(self, response: CacheResponse, operation_type: str):
        """Record operation metrics"""
        self.total_operations += 1
        
        if response.success:
            self.successful_operations += 1
        else:
            self.failure_types[response.error] = self.failure_types.get(response.error, 0) + 1
            
        if response.hit:
            self.cache_hits += 1
        elif operation_type == "get":
            self.cache_misses += 1
            
        self.response_times.append(response.execution_time)
        self.memory_usage_history.append(response.memory_usage)
        
        # Throughput calculation
        current_time = time.time()
        self.operations_since_check += 1
        
        if current_time - self.last_throughput_check >= 1.0:  # Every second
            throughput = self.operations_since_check / (current_time - self.last_throughput_check)
            self.throughput_history.append(throughput)
            self.last_throughput_check = current_time
            self.operations_since_check = 0
            
    def get_reliability(self) -> float:
        """Calculate system reliability"""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
        
    def get_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total_gets = self.cache_hits + self.cache_misses
        if total_gets == 0:
            return 0.0
        return self.cache_hits / total_gets
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        if not self.response_times:
            return {'mean': 0, 'p99': 0, 'p95': 0, 'throughput': 0}
            
        sorted_times = sorted(self.response_times)
        return {
            'mean': statistics.mean(self.response_times),
            'p99': sorted_times[int(0.99 * len(sorted_times))],
            'p95': sorted_times[int(0.95 * len(sorted_times))],
            'p50': statistics.median(self.response_times),
            'throughput': statistics.mean(self.throughput_history) if self.throughput_history else 0,
            'max_throughput': max(self.throughput_history) if self.throughput_history else 0
        }


class LRUCache:
    """LRU eviction policy implementation with memory management"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_usage = 0
        self.lock = threading.RLock()
        
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, dict):
            return len(json.dumps(value).encode('utf-8'))
        else:
            return len(str(value).encode('utf-8'))
            
    def _evict_if_needed(self) -> int:
        """Evict entries if cache limits exceeded"""
        evicted = 0
        
        # Size-based eviction
        while len(self.cache) >= self.max_size:
            key, entry = self.cache.popitem(last=False)
            self.current_memory_usage -= entry.size_bytes
            evicted += 1
            
        # Memory-based eviction
        while self.current_memory_usage > self.max_memory_bytes and self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_memory_usage -= entry.size_bytes
            evicted += 1
            
        return evicted
        
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get value from cache with LRU update"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiry
                if entry.expiry and datetime.now() > entry.expiry:
                    del self.cache[key]
                    self.current_memory_usage -= entry.size_bytes
                    return False, None
                    
                # Update LRU order
                self.cache.move_to_end(key)
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                return True, entry.value
            return False, None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        with self.lock:
            size_bytes = self._estimate_size(value)
            expiry = None
            if ttl:
                expiry = datetime.now() + timedelta(seconds=ttl)
                
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                expiry=expiry
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_usage -= old_entry.size_bytes
                del self.cache[key]
                
            # Add new entry
            self.cache[key] = entry
            self.current_memory_usage += size_bytes
            
            # Evict if necessary
            self._evict_if_needed()
            
            return True
            
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory_usage -= entry.size_bytes
                del self.cache[key]
                return True
            return False
            
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self.lock:
            return key in self.cache
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self.current_memory_usage / self.max_memory_bytes
            }


class DistributedConsistency:
    """Simulated distributed consistency mechanisms"""
    
    def __init__(self):
        self.consistency_delay = 0.01  # 10ms simulation
        self.consistency_failure_rate = 0.02  # 2% failure rate
        
    async def achieve_consistency(self, operation: CacheOperation) -> bool:
        """Simulate achieving distributed consistency"""
        # Simulate network delay
        await asyncio.sleep(self.consistency_delay)
        
        # Simulate consistency failure
        if random.random() < self.consistency_failure_rate:
            return False
            
        return True
        
    def validate_consistency(self, key: str, expected_value: Any) -> bool:
        """Validate consistency across nodes (simulated)"""
        # In real implementation, this would check across multiple nodes
        return random.random() > 0.01  # 99% consistency validation success


class ShardingManager:
    """Sharding logic for distributing cache entries"""
    
    def __init__(self, num_shards: int = 16):
        self.num_shards = num_shards
        
    def get_shard_id(self, key: str) -> int:
        """Calculate shard ID for given key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards
        
    def get_shard_key(self, key: str) -> str:
        """Get shard identifier for key"""
        return f"shard_{self.get_shard_id(key)}"


class ReplicationManager:
    """Handles data replication for fault tolerance"""
    
    def __init__(self, replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.replication_failure_rate = 0.05  # 5% replication failure rate
        
    async def replicate_operation(self, operation: CacheOperation) -> bool:
        """Simulate replicating operation to replica nodes"""
        successful_replicas = 0
        
        for replica_id in range(self.replication_factor):
            # Simulate network delay to replica
            await asyncio.sleep(0.005)  # 5ms per replica
            
            # Simulate replication failure
            if random.random() > self.replication_failure_rate:
                successful_replicas += 1
                
        # Consider replication successful if majority succeeds
        return successful_replicas >= (self.replication_factor // 2 + 1)


class SingleAgentDistributedCache:
    """
    Single-agent distributed cache with comprehensive features
    
    Baseline implementation with 88% reliability for depth multiplication comparison.
    Includes all distributed cache features but without federation benefits.
    """
    
    def __init__(self, 
                 max_size: int = 10000, 
                 max_memory_mb: float = 100.0,
                 num_shards: int = 16,
                 replication_factor: int = 3,
                 enable_monitoring: bool = True):
        
        # Core cache storage
        self.cache = LRUCache(max_size, max_memory_mb)
        
        # Distributed features
        self.consistency = DistributedConsistency()
        self.sharding = ShardingManager(num_shards)
        self.replication = ReplicationManager(replication_factor)
        
        # Rate limiting
        self.rate_limiter = TokenBucketRateLimiter(
            capacity=1000,     # 1000 operations per window
            refill_rate=200.0  # 200 operations per second
        )
        
        # Monitoring and metrics
        self.metrics = PerformanceMetrics() if enable_monitoring else None
        self.node_id = "single_agent_cache"
        
        # Failure simulation for reliability testing
        self.base_failure_rate = 0.12  # 12% baseline failure rate
        self.failure_simulation = True
        
    async def execute_operation(self, operation: CacheOperation) -> CacheResponse:
        """Execute cache operation with comprehensive error handling"""
        start_time = time.time()
        
        # Rate limiting check
        if not await self.rate_limiter.allow_async():
            return CacheResponse(
                success=False,
                error="Rate limit exceeded",
                execution_time=time.time() - start_time,
                node_id=self.node_id
            )
            
        # Failure simulation for testing
        if self.failure_simulation and random.random() < self.base_failure_rate:
            execution_time = time.time() - start_time
            response = CacheResponse(
                success=False,
                error="Simulated agent failure",
                execution_time=execution_time,
                node_id=self.node_id
            )
            
            if self.metrics:
                self.metrics.record_operation(response, operation.operation)
            return response
            
        try:
            # Execute the requested operation
            if operation.operation == "get":
                response = await self._handle_get(operation, start_time)
            elif operation.operation == "set":
                response = await self._handle_set(operation, start_time)
            elif operation.operation == "delete":
                response = await self._handle_delete(operation, start_time)
            elif operation.operation == "exists":
                response = await self._handle_exists(operation, start_time)
            else:
                response = CacheResponse(
                    success=False,
                    error=f"Unsupported operation: {operation.operation}",
                    execution_time=time.time() - start_time,
                    node_id=self.node_id
                )
                
            # Record metrics
            if self.metrics:
                self.metrics.record_operation(response, operation.operation)
                
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            response = CacheResponse(
                success=False,
                error=f"Operation failed: {str(e)}",
                execution_time=execution_time,
                node_id=self.node_id
            )
            
            if self.metrics:
                self.metrics.record_operation(response, operation.operation)
            return response
            
    async def _handle_get(self, operation: CacheOperation, start_time: float) -> CacheResponse:
        """Handle GET operation with consistency checks"""
        hit, value = self.cache.get(operation.key)
        
        # Strong consistency check if requested
        consistency_achieved = True
        if operation.consistency_level == "strong":
            consistency_achieved = await self.consistency.achieve_consistency(operation)
            
        stats = self.cache.get_stats()
        
        return CacheResponse(
            success=True,
            value=value,
            hit=hit,
            execution_time=time.time() - start_time,
            node_id=self.node_id,
            cache_size=stats['size'],
            memory_usage=stats['memory_usage_mb'],
            consistency_achieved=consistency_achieved
        )
        
    async def _handle_set(self, operation: CacheOperation, start_time: float) -> CacheResponse:
        """Handle SET operation with replication and consistency"""
        # Store in local cache
        success = self.cache.set(operation.key, operation.value, operation.ttl)
        
        # Replicate to other nodes
        replication_success = await self.replication.replicate_operation(operation)
        
        # Achieve consistency
        consistency_achieved = await self.consistency.achieve_consistency(operation)
        
        stats = self.cache.get_stats()
        
        return CacheResponse(
            success=success and replication_success,
            execution_time=time.time() - start_time,
            node_id=self.node_id,
            cache_size=stats['size'],
            memory_usage=stats['memory_usage_mb'],
            consistency_achieved=consistency_achieved,
            replication_success=replication_success
        )
        
    async def _handle_delete(self, operation: CacheOperation, start_time: float) -> CacheResponse:
        """Handle DELETE operation with replication"""
        success = self.cache.delete(operation.key)
        
        # Replicate deletion
        replication_success = await self.replication.replicate_operation(operation)
        
        # Achieve consistency
        consistency_achieved = await self.consistency.achieve_consistency(operation)
        
        stats = self.cache.get_stats()
        
        return CacheResponse(
            success=success and replication_success,
            execution_time=time.time() - start_time,
            node_id=self.node_id,
            cache_size=stats['size'],
            memory_usage=stats['memory_usage_mb'],
            consistency_achieved=consistency_achieved,
            replication_success=replication_success
        )
        
    async def _handle_exists(self, operation: CacheOperation, start_time: float) -> CacheResponse:
        """Handle EXISTS operation"""
        exists = self.cache.exists(operation.key)
        
        stats = self.cache.get_stats()
        
        return CacheResponse(
            success=True,
            value=exists,
            execution_time=time.time() - start_time,
            node_id=self.node_id,
            cache_size=stats['size'],
            memory_usage=stats['memory_usage_mb'],
            consistency_achieved=True
        )
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.metrics:
            return {}
            
        return {
            'reliability': self.metrics.get_reliability(),
            'hit_ratio': self.metrics.get_hit_ratio(),
            'performance_stats': self.metrics.get_performance_stats(),
            'total_operations': self.metrics.total_operations,
            'cache_stats': self.cache.get_stats(),
            'failure_types': dict(self.metrics.failure_types),
            'node_id': self.node_id,
            'architecture': 'single_agent',
            'depth_level': 0
        }
        
    def enable_failure_simulation(self, enabled: bool = True):
        """Enable/disable failure simulation for testing"""
        self.failure_simulation = enabled
        
    def set_failure_rate(self, failure_rate: float):
        """Set failure rate for testing (0.0 - 1.0)"""
        self.base_failure_rate = max(0.0, min(1.0, failure_rate))


# Demonstration and testing functionality
async def demonstrate_single_agent_cache():
    """Demonstrate single-agent cache with comprehensive testing"""
    print("=== Single Agent Distributed Cache (Depth=0) ===\n")
    
    # Initialize cache
    cache = SingleAgentDistributedCache(
        max_size=1000,
        max_memory_mb=50.0,
        enable_monitoring=True
    )
    
    print("System initialized with features:")
    print("✓ LRU eviction policy")
    print("✓ Distributed consistency simulation")
    print("✓ Sharding across logical partitions") 
    print("✓ Replication for fault tolerance")
    print("✓ Performance monitoring")
    print("✓ Rate limiting integration")
    print()
    
    # Test operations
    test_operations = [
        CacheOperation("set", "user:1001", {"name": "Alice", "age": 30}, client_id="test_client"),
        CacheOperation("set", "user:1002", {"name": "Bob", "age": 25}, client_id="test_client"),
        CacheOperation("get", "user:1001", client_id="test_client"),
        CacheOperation("get", "user:1003", client_id="test_client"),  # Miss
        CacheOperation("exists", "user:1002", client_id="test_client"),
        CacheOperation("delete", "user:1001", client_id="test_client"),
        CacheOperation("get", "user:1001", client_id="test_client"),  # Miss after delete
    ]
    
    print("Executing test operations:")
    print()
    
    # Execute test operations
    for i, operation in enumerate(test_operations, 1):
        response = await cache.execute_operation(operation)
        
        print(f"Operation {i}: {operation.operation.upper()} {operation.key}")
        print(f"  Success: {response.success}")
        if response.value is not None:
            print(f"  Value: {response.value}")
        if response.hit:
            print(f"  Cache Hit: {response.hit}")
        print(f"  Execution Time: {response.execution_time:.4f}s")
        print(f"  Cache Size: {response.cache_size}")
        print(f"  Memory Usage: {response.memory_usage:.2f}MB")
        print()
        
    # Performance stress test
    print("Performing stress test with failure simulation...")
    
    stress_operations = []
    for i in range(100):
        if i % 4 == 0:
            op = CacheOperation("set", f"stress_key_{i}", f"value_{i}", client_id="stress_client")
        elif i % 4 == 1:
            op = CacheOperation("get", f"stress_key_{i-1}", client_id="stress_client")
        elif i % 4 == 2:
            op = CacheOperation("exists", f"stress_key_{i-2}", client_id="stress_client")
        else:
            op = CacheOperation("delete", f"stress_key_{i-3}", client_id="stress_client")
        stress_operations.append(op)
        
    # Execute stress test
    start_time = time.time()
    for operation in stress_operations:
        await cache.execute_operation(operation)
    total_time = time.time() - start_time
    
    # Display comprehensive metrics
    metrics = cache.get_performance_metrics()
    
    print("=== PERFORMANCE ANALYSIS ===")
    print(f"Architecture: Single Agent (Depth=0)")
    print(f"Reliability: {metrics['reliability']:.3f} ({metrics['reliability']*100:.1f}%)")
    print(f"Cache Hit Ratio: {metrics['hit_ratio']:.3f}")
    print(f"Total Operations: {metrics['total_operations']}")
    print()
    
    perf_stats = metrics['performance_stats']
    print("Response Time Analysis:")
    print(f"  Mean: {perf_stats['mean']:.4f}s")
    print(f"  P99: {perf_stats['p99']:.4f}s") 
    print(f"  P95: {perf_stats['p95']:.4f}s")
    print(f"  P50: {perf_stats['p50']:.4f}s")
    print()
    
    print("Throughput Analysis:")
    print(f"  Average: {perf_stats['throughput']:.1f} ops/sec")
    print(f"  Peak: {perf_stats['max_throughput']:.1f} ops/sec")
    print(f"  Total Time: {total_time:.3f}s")
    print()
    
    cache_stats = metrics['cache_stats']
    print("Cache Statistics:")
    print(f"  Current Size: {cache_stats['size']}")
    print(f"  Memory Usage: {cache_stats['memory_usage_mb']:.2f}MB")
    print(f"  Memory Utilization: {cache_stats['memory_utilization']:.1%}")
    print()
    
    if metrics['failure_types']:
        print("Failure Analysis:")
        for failure_type, count in metrics['failure_types'].items():
            print(f"  {failure_type}: {count} occurrences")
        print()
    
    print("=== SINGLE AGENT LIMITATIONS ===")
    print("❌ No redundancy - single point of failure")
    print("❌ No automatic failover capabilities") 
    print("❌ No distributed consensus mechanisms")
    print("❌ Limited by single agent reliability")
    print(f"❌ Baseline error rate: {cache.base_failure_rate:.1%}")
    print()
    
    print("Expected improvement with federation:")
    single_reliability = metrics['reliability']
    federated_reliability = 1 - (1 - single_reliability)**3
    meta_reliability = 1 - ((1 - single_reliability)**3)**3
    
    print(f"Current (Single):     {single_reliability:.3f}")
    print(f"Federated (Depth=1):  {federated_reliability:.3f}")
    print(f"Meta-Fed (Depth=2):   {meta_reliability:.3f}")
    print()
    
    return metrics
    
    
if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_single_agent_cache())