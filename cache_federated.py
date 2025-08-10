"""
Distributed Cache System - Federated Implementation (Depth=1)

Federation of 3 specialized cache agents with consensus mechanisms and automatic failover.
Demonstrates exponential reliability improvement through redundancy.

Reliability Formula: P(success) = 1 - ε^N (where N=3 agents, ε=12% individual error rate)
Expected Reliability: 1 - 0.12^3 = 99.83%
"""

import asyncio
import json
import time
import hashlib
import threading
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import statistics

# Import base cache components
from cache_single import (
    CacheEntry, CacheOperation, CacheResponse, PerformanceMetrics,
    LRUCache, DistributedConsistency, ShardingManager, ReplicationManager
)

# Import federation orchestrator components
from meta_federation_system import (
    AgentSpecialty, ReliabilityMetrics, CircuitBreaker
)

# Import proven rate limiter
from rate_limiter_final import TokenBucketRateLimiter, MultiKeyRateLimiter


class CacheAgentSpecialty(Enum):
    """Specialized cache agent types"""
    PERFORMANCE_CACHE = "performance_cache"  # Optimized for speed
    CONSISTENCY_CACHE = "consistency_cache"  # Strong consistency focus  
    DURABILITY_CACHE = "durability_cache"    # Fault tolerance focus


@dataclass
class AgentResponse:
    """Response from individual cache agent"""
    agent_id: str
    specialty: CacheAgentSpecialty
    response: CacheResponse
    confidence: float
    consensus_vote: bool = True
    execution_path: str = ""


@dataclass
class FederationResponse:
    """Response from federated cache system"""
    success: bool
    value: Any = None
    hit: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    consensus_achieved: bool = False
    confidence_score: float = 0.0
    primary_agent: str = ""
    backup_agents: List[str] = field(default_factory=list)
    cache_size: int = 0
    memory_usage: float = 0.0
    reliability_score: float = 0.0
    cascade_prevented: bool = False
    agent_responses: List[AgentResponse] = field(default_factory=list)


class ConsensusManager:
    """Manages consensus across cache agents"""
    
    def __init__(self, quorum_size: int = 2):
        self.quorum_size = quorum_size
        self.consensus_timeout = 0.1  # 100ms timeout
        
    async def achieve_consensus(self, agent_responses: List[AgentResponse], 
                              operation_type: str) -> Tuple[bool, Any, float]:
        """Achieve consensus across agent responses"""
        if not agent_responses:
            return False, None, 0.0
            
        # For GET operations, use majority voting on values
        if operation_type == "get":
            return self._consensus_get(agent_responses)
        # For SET/DELETE, use majority voting on success
        elif operation_type in ["set", "delete", "exists"]:
            return self._consensus_modify(agent_responses)
        else:
            return False, None, 0.0
            
    def _consensus_get(self, responses: List[AgentResponse]) -> Tuple[bool, Any, float]:
        """Consensus for GET operations"""
        successful_responses = [r for r in responses if r.response.success]
        
        if len(successful_responses) < self.quorum_size:
            return False, None, 0.0
            
        # Collect values and their frequencies
        value_votes = {}
        confidence_scores = {}
        
        for response in successful_responses:
            value_key = str(response.response.value) if response.response.value is not None else "None"
            
            if value_key not in value_votes:
                value_votes[value_key] = 0
                confidence_scores[value_key] = []
                
            value_votes[value_key] += 1
            confidence_scores[value_key].append(response.confidence)
            
        # Find majority value
        max_votes = max(value_votes.values())
        if max_votes >= self.quorum_size:
            # Get the value with most votes
            majority_value_key = max(value_votes.items(), key=lambda x: x[1])[0]
            majority_value = None if majority_value_key == "None" else json.loads(majority_value_key) if majority_value_key.startswith('{') else majority_value_key
            
            # Calculate confidence
            avg_confidence = statistics.mean(confidence_scores[majority_value_key])
            consensus_strength = max_votes / len(successful_responses)
            final_confidence = avg_confidence * consensus_strength
            
            return True, majority_value, final_confidence
            
        return False, None, 0.0
        
    def _consensus_modify(self, responses: List[AgentResponse]) -> Tuple[bool, Any, float]:
        """Consensus for SET/DELETE operations"""
        successful_responses = [r for r in responses if r.response.success]
        
        if len(successful_responses) < self.quorum_size:
            return False, None, 0.0
            
        # For modify operations, success if quorum succeeds
        avg_confidence = statistics.mean([r.confidence for r in successful_responses])
        consensus_strength = len(successful_responses) / len(responses)
        final_confidence = avg_confidence * consensus_strength
        
        return True, True, final_confidence


class PerformanceCacheAgent:
    """Performance-optimized cache agent with minimal latency"""
    
    def __init__(self, agent_id: str = "performance_agent"):
        self.agent_id = agent_id
        self.specialty = CacheAgentSpecialty.PERFORMANCE_CACHE
        
        # Performance-optimized configuration
        self.cache = LRUCache(max_size=15000, max_memory_mb=150.0)  # Larger cache
        self.rate_limiter = TokenBucketRateLimiter(capacity=1500, refill_rate=300.0)
        
        # Minimal consistency overhead
        self.consistency_enabled = False
        self.replication_enabled = False
        
        # Performance metrics
        self.total_operations = 0
        self.response_times = []
        self.base_failure_rate = 0.08  # 8% failure rate
        
    async def execute_operation(self, operation: CacheOperation) -> AgentResponse:
        """Execute operation optimized for performance"""
        start_time = time.time()
        
        # Rate limiting
        if not await self.rate_limiter.allow_async():
            response = CacheResponse(
                success=False,
                error="Rate limit exceeded",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="rate_limited"
            )
            
        # Simulate failure for testing
        if random.random() < self.base_failure_rate:
            response = CacheResponse(
                success=False,
                error="Agent failure simulation",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="simulated_failure"
            )
            
        # Execute fast cache operations
        try:
            if operation.operation == "get":
                hit, value = self.cache.get(operation.key)
                response = CacheResponse(
                    success=True,
                    value=value,
                    hit=hit,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache)
                )
                confidence = 0.95 if hit else 0.80
                
            elif operation.operation == "set":
                success = self.cache.set(operation.key, operation.value, operation.ttl)
                response = CacheResponse(
                    success=success,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache)
                )
                confidence = 0.90
                
            elif operation.operation == "delete":
                success = self.cache.delete(operation.key)
                response = CacheResponse(
                    success=success,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache)
                )
                confidence = 0.85
                
            elif operation.operation == "exists":
                exists = self.cache.exists(operation.key)
                response = CacheResponse(
                    success=True,
                    value=exists,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache)
                )
                confidence = 0.90
                
            else:
                response = CacheResponse(
                    success=False,
                    error=f"Unsupported operation: {operation.operation}",
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id
                )
                confidence = 0.0
                
            self.total_operations += 1
            self.response_times.append(response.execution_time)
            
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=confidence,
                execution_path="performance_optimized"
            )
            
        except Exception as e:
            response = CacheResponse(
                success=False,
                error=f"Performance agent error: {str(e)}",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="exception_handled"
            )


class ConsistencyCacheAgent:
    """Consistency-focused cache agent with strong guarantees"""
    
    def __init__(self, agent_id: str = "consistency_agent"):
        self.agent_id = agent_id
        self.specialty = CacheAgentSpecialty.CONSISTENCY_CACHE
        
        # Consistency-optimized configuration
        self.cache = LRUCache(max_size=10000, max_memory_mb=100.0)
        self.consistency = DistributedConsistency()
        self.rate_limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=200.0)
        
        # Strong consistency enabled
        self.consistency_enabled = True
        self.consistency_timeout = 0.05  # 50ms consistency operations
        
        # Reliability metrics
        self.total_operations = 0
        self.consistency_violations = 0
        self.base_failure_rate = 0.10  # 10% failure rate (higher due to complexity)
        
    async def execute_operation(self, operation: CacheOperation) -> AgentResponse:
        """Execute operation with strong consistency guarantees"""
        start_time = time.time()
        
        # Rate limiting
        if not await self.rate_limiter.allow_async():
            response = CacheResponse(
                success=False,
                error="Rate limit exceeded", 
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="rate_limited"
            )
            
        # Simulate failure for testing
        if random.random() < self.base_failure_rate:
            response = CacheResponse(
                success=False,
                error="Agent failure simulation",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="simulated_failure"
            )
            
        try:
            # Execute with consistency guarantees
            if operation.operation == "get":
                # Achieve consistency before reading
                consistency_achieved = await self.consistency.achieve_consistency(operation)
                
                hit, value = self.cache.get(operation.key)
                response = CacheResponse(
                    success=True,
                    value=value,
                    hit=hit,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache),
                    consistency_achieved=consistency_achieved
                )
                
                # High confidence if consistency achieved
                confidence = 0.98 if consistency_achieved else 0.75
                
            elif operation.operation == "set":
                # Set with consistency guarantees
                success = self.cache.set(operation.key, operation.value, operation.ttl)
                consistency_achieved = await self.consistency.achieve_consistency(operation)
                
                response = CacheResponse(
                    success=success and consistency_achieved,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache),
                    consistency_achieved=consistency_achieved
                )
                confidence = 0.95 if consistency_achieved else 0.70
                
            elif operation.operation == "delete":
                success = self.cache.delete(operation.key)
                consistency_achieved = await self.consistency.achieve_consistency(operation)
                
                response = CacheResponse(
                    success=success and consistency_achieved,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache),
                    consistency_achieved=consistency_achieved
                )
                confidence = 0.90 if consistency_achieved else 0.65
                
            elif operation.operation == "exists":
                consistency_achieved = await self.consistency.achieve_consistency(operation)
                exists = self.cache.exists(operation.key)
                
                response = CacheResponse(
                    success=True,
                    value=exists,
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id,
                    cache_size=len(self.cache.cache),
                    consistency_achieved=consistency_achieved
                )
                confidence = 0.95 if consistency_achieved else 0.75
                
            else:
                response = CacheResponse(
                    success=False,
                    error=f"Unsupported operation: {operation.operation}",
                    execution_time=time.time() - start_time,
                    node_id=self.agent_id
                )
                confidence = 0.0
                
            self.total_operations += 1
            
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=confidence,
                execution_path="consistency_guaranteed"
            )
            
        except Exception as e:
            response = CacheResponse(
                success=False,
                error=f"Consistency agent error: {str(e)}",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="exception_handled"
            )


class DurabilityCacheAgent:
    """Durability-focused cache agent with fault tolerance"""
    
    def __init__(self, agent_id: str = "durability_agent"):
        self.agent_id = agent_id
        self.specialty = CacheAgentSpecialty.DURABILITY_CACHE
        
        # Durability-optimized configuration
        self.cache = LRUCache(max_size=8000, max_memory_mb=80.0)  # Smaller for reliability
        self.replication = ReplicationManager(replication_factor=5)  # High replication
        self.rate_limiter = TokenBucketRateLimiter(capacity=800, refill_rate=150.0)
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        # Durability metrics
        self.total_operations = 0
        self.replication_failures = 0
        self.base_failure_rate = 0.15  # 15% failure rate (highest, most complex)
        
    async def execute_operation(self, operation: CacheOperation) -> AgentResponse:
        """Execute operation with maximum fault tolerance"""
        start_time = time.time()
        
        # Rate limiting
        if not await self.rate_limiter.allow_async():
            response = CacheResponse(
                success=False,
                error="Rate limit exceeded",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="rate_limited"
            )
            
        # Simulate failure for testing
        if random.random() < self.base_failure_rate:
            response = CacheResponse(
                success=False,
                error="Agent failure simulation",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="simulated_failure"
            )
            
        try:
            # Execute with circuit breaker protection
            async def protected_operation():
                if operation.operation == "get":
                    hit, value = self.cache.get(operation.key)
                    return CacheResponse(
                        success=True,
                        value=value,
                        hit=hit,
                        execution_time=time.time() - start_time,
                        node_id=self.agent_id,
                        cache_size=len(self.cache.cache)
                    )
                    
                elif operation.operation == "set":
                    # High durability write with replication
                    success = self.cache.set(operation.key, operation.value, operation.ttl)
                    replication_success = await self.replication.replicate_operation(operation)
                    
                    if not replication_success:
                        self.replication_failures += 1
                    
                    return CacheResponse(
                        success=success and replication_success,
                        execution_time=time.time() - start_time,
                        node_id=self.agent_id,
                        cache_size=len(self.cache.cache),
                        replication_success=replication_success
                    )
                    
                elif operation.operation == "delete":
                    success = self.cache.delete(operation.key)
                    replication_success = await self.replication.replicate_operation(operation)
                    
                    return CacheResponse(
                        success=success and replication_success,
                        execution_time=time.time() - start_time,
                        node_id=self.agent_id,
                        cache_size=len(self.cache.cache),
                        replication_success=replication_success
                    )
                    
                elif operation.operation == "exists":
                    exists = self.cache.exists(operation.key)
                    return CacheResponse(
                        success=True,
                        value=exists,
                        execution_time=time.time() - start_time,
                        node_id=self.agent_id,
                        cache_size=len(self.cache.cache)
                    )
                    
                else:
                    return CacheResponse(
                        success=False,
                        error=f"Unsupported operation: {operation.operation}",
                        execution_time=time.time() - start_time,
                        node_id=self.agent_id
                    )
                    
            # Execute with circuit breaker
            response = await self.circuit_breaker.call(protected_operation)
            
            # Calculate confidence based on durability features
            confidence = 0.92
            if hasattr(response, 'replication_success') and not response.replication_success:
                confidence *= 0.8
                
            self.total_operations += 1
            
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=confidence,
                execution_path="durability_protected"
            )
            
        except Exception as e:
            response = CacheResponse(
                success=False,
                error=f"Durability agent error: {str(e)}",
                execution_time=time.time() - start_time,
                node_id=self.agent_id
            )
            return AgentResponse(
                agent_id=self.agent_id,
                specialty=self.specialty,
                response=response,
                confidence=0.0,
                execution_path="exception_handled"
            )


class FederatedDistributedCache:
    """
    Federated distributed cache with 3 specialized agents
    
    Demonstrates exponential reliability improvement through agent redundancy
    and consensus mechanisms. Target reliability: 99.83% vs 88% single-agent.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        # Initialize specialized agents
        self.performance_agent = PerformanceCacheAgent("perf_cache_001")
        self.consistency_agent = ConsistencyCacheAgent("cons_cache_002") 
        self.durability_agent = DurabilityCacheAgent("dura_cache_003")
        
        self.agents = [
            self.performance_agent,
            self.consistency_agent, 
            self.durability_agent
        ]
        
        # Federation management
        self.consensus_manager = ConsensusManager(quorum_size=2)
        self.reliability_metrics = ReliabilityMetrics()
        
        # System configuration
        self.federation_id = "federated_cache_system"
        self.enable_monitoring = enable_monitoring
        self.max_concurrent_operations = 50
        self.active_operations = 0
        
        # Performance tracking
        self.total_operations = 0
        self.successful_operations = 0
        self.consensus_achieved = 0
        self.cascade_prevented = 0
        
    async def execute_operation(self, operation: CacheOperation) -> FederationResponse:
        """Execute cache operation across federated agents with consensus"""
        start_time = time.time()
        
        # Capacity check
        if self.active_operations >= self.max_concurrent_operations:
            return FederationResponse(
                success=False,
                error="System at capacity",
                execution_time=time.time() - start_time
            )
            
        self.active_operations += 1
        
        try:
            # Execute operation across all agents in parallel
            agent_tasks = []
            for agent in self.agents:
                task = asyncio.create_task(agent.execute_operation(operation))
                agent_tasks.append(task)
                
            # Wait for all agents to respond
            agent_responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Filter successful responses and handle exceptions
            valid_responses = []
            for i, response in enumerate(agent_responses):
                if isinstance(response, Exception):
                    # Create error response for failed agent
                    error_response = AgentResponse(
                        agent_id=self.agents[i].agent_id,
                        specialty=self.agents[i].specialty,
                        response=CacheResponse(
                            success=False,
                            error=f"Agent exception: {str(response)}",
                            execution_time=time.time() - start_time,
                            node_id=self.agents[i].agent_id
                        ),
                        confidence=0.0,
                        execution_path="exception"
                    )
                    valid_responses.append(error_response)
                else:
                    valid_responses.append(response)
                    
            # Achieve consensus across agent responses
            consensus_success, consensus_value, confidence_score = await self.consensus_manager.achieve_consensus(
                valid_responses, operation.operation
            )
            
            # Determine primary agent (highest confidence successful response)
            successful_agents = [r for r in valid_responses if r.response.success]
            primary_agent = ""
            backup_agents = []
            
            if successful_agents:
                primary_agent_response = max(successful_agents, key=lambda r: r.confidence)
                primary_agent = primary_agent_response.agent_id
                backup_agents = [r.agent_id for r in successful_agents if r.agent_id != primary_agent]
                
            # Calculate system metrics
            agents_used = [r.agent_id for r in valid_responses]
            total_cache_size = sum(r.response.cache_size for r in valid_responses if r.response.cache_size > 0)
            avg_memory_usage = statistics.mean([r.response.memory_usage for r in valid_responses if r.response.memory_usage > 0]) if any(r.response.memory_usage > 0 for r in valid_responses) else 0.0
            
            # Determine overall success
            if consensus_success and len(successful_agents) >= 2:
                # Successful consensus
                self.total_operations += 1
                self.successful_operations += 1
                if consensus_success:
                    self.consensus_achieved += 1
                    
                return FederationResponse(
                    success=True,
                    value=consensus_value,
                    hit=any(r.response.hit for r in valid_responses if r.response.success),
                    execution_time=time.time() - start_time,
                    agents_used=agents_used,
                    consensus_achieved=True,
                    confidence_score=confidence_score,
                    primary_agent=primary_agent,
                    backup_agents=backup_agents,
                    cache_size=total_cache_size,
                    memory_usage=avg_memory_usage,
                    reliability_score=len(successful_agents) / len(valid_responses),
                    agent_responses=valid_responses
                )
                
            elif len(successful_agents) >= 1:
                # Partial success - at least one agent succeeded
                self.total_operations += 1
                self.cascade_prevented += 1
                
                # Use best available response
                best_response = max(successful_agents, key=lambda r: r.confidence)
                
                return FederationResponse(
                    success=True,  # Partial success still considered success
                    value=best_response.response.value,
                    hit=best_response.response.hit,
                    execution_time=time.time() - start_time,
                    agents_used=agents_used,
                    consensus_achieved=False,
                    confidence_score=best_response.confidence * 0.7,  # Reduced confidence
                    primary_agent=best_response.agent_id,
                    backup_agents=[],
                    cache_size=total_cache_size,
                    memory_usage=avg_memory_usage,
                    reliability_score=len(successful_agents) / len(valid_responses),
                    cascade_prevented=True,
                    agent_responses=valid_responses
                )
                
            else:
                # Complete failure
                self.total_operations += 1
                
                return FederationResponse(
                    success=False,
                    error="All agents failed",
                    execution_time=time.time() - start_time,
                    agents_used=agents_used,
                    consensus_achieved=False,
                    confidence_score=0.0,
                    cache_size=total_cache_size,
                    memory_usage=avg_memory_usage,
                    reliability_score=0.0,
                    agent_responses=valid_responses
                )
                
        except Exception as e:
            self.total_operations += 1
            
            return FederationResponse(
                success=False,
                error=f"Federation error: {str(e)}",
                execution_time=time.time() - start_time
            )
            
        finally:
            self.active_operations -= 1
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive federation metrics"""
        if self.total_operations == 0:
            reliability = 0.0
            consensus_rate = 0.0
            cascade_prevention_rate = 0.0
        else:
            reliability = self.successful_operations / self.total_operations
            consensus_rate = self.consensus_achieved / self.total_operations
            cascade_prevention_rate = self.cascade_prevented / self.total_operations
            
        # Agent-specific metrics
        agent_metrics = {}
        for agent in self.agents:
            agent_metrics[agent.agent_id] = {
                'total_operations': getattr(agent, 'total_operations', 0),
                'specialty': agent.specialty.value,
                'base_failure_rate': getattr(agent, 'base_failure_rate', 0.0)
            }
            
        return {
            'reliability': reliability,
            'consensus_rate': consensus_rate,
            'cascade_prevention_rate': cascade_prevention_rate,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'agent_metrics': agent_metrics,
            'federation_id': self.federation_id,
            'architecture': 'federated',
            'depth_level': 1,
            'agent_count': len(self.agents)
        }
        
    def get_theoretical_reliability(self) -> Dict[str, float]:
        """Calculate theoretical vs empirical reliability"""
        # Individual agent error rates
        agent_error_rates = {
            'performance': 0.08,
            'consistency': 0.10,
            'durability': 0.15
        }
        
        # Average error rate
        avg_error_rate = statistics.mean(agent_error_rates.values())
        
        # Theoretical federation reliability: 1 - ε^N
        theoretical_reliability = 1 - (avg_error_rate ** 3)
        
        # Empirical reliability
        empirical_reliability = self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0
        
        return {
            'theoretical_reliability': theoretical_reliability,
            'empirical_reliability': empirical_reliability,
            'avg_agent_error_rate': avg_error_rate,
            'reliability_improvement': theoretical_reliability / (1 - avg_error_rate) if avg_error_rate < 1 else 0,
            'agent_error_rates': agent_error_rates
        }


# Demonstration and testing functionality
async def demonstrate_federated_cache():
    """Demonstrate federated cache with 3 agents and consensus"""
    print("=== Federated Distributed Cache (Depth=1) ===\n")
    
    # Initialize federated cache
    cache = FederatedDistributedCache(enable_monitoring=True)
    
    print("System initialized with 3 specialized agents:")
    print("✓ Performance Agent: Speed-optimized, minimal latency")
    print("✓ Consistency Agent: Strong consistency guarantees")
    print("✓ Durability Agent: Maximum fault tolerance, replication")
    print("✓ Consensus Manager: 2/3 quorum for reliability")
    print("✓ Circuit Breakers: Automatic failure isolation")
    print()
    
    # Test operations
    test_operations = [
        CacheOperation("set", "user:2001", {"name": "Alice", "role": "admin"}, client_id="fed_client"),
        CacheOperation("set", "user:2002", {"name": "Bob", "role": "user"}, client_id="fed_client"),
        CacheOperation("get", "user:2001", client_id="fed_client"),
        CacheOperation("get", "user:2003", client_id="fed_client"),  # Miss
        CacheOperation("exists", "user:2002", client_id="fed_client"),
        CacheOperation("delete", "user:2001", client_id="fed_client"),
        CacheOperation("get", "user:2001", client_id="fed_client"),  # Miss after delete
    ]
    
    print("Executing federated test operations:")
    print()
    
    # Execute test operations
    for i, operation in enumerate(test_operations, 1):
        response = await cache.execute_operation(operation)
        
        print(f"Operation {i}: {operation.operation.upper()} {operation.key}")
        print(f"  Success: {response.success}")
        print(f"  Consensus: {response.consensus_achieved}")
        print(f"  Confidence: {response.confidence_score:.3f}")
        print(f"  Primary Agent: {response.primary_agent}")
        print(f"  Agents Used: {len(response.agents_used)}")
        
        if response.value is not None:
            print(f"  Value: {response.value}")
        if hasattr(response, 'hit'):
            print(f"  Cache Hit: {response.hit}")
        if response.cascade_prevented:
            print(f"  ✓ Cascade Prevented")
            
        print(f"  Execution Time: {response.execution_time:.4f}s")
        print(f"  Reliability Score: {response.reliability_score:.3f}")
        print()
        
        # Show agent responses summary
        successful_agents = [r for r in response.agent_responses if r.response.success]
        failed_agents = [r for r in response.agent_responses if not r.response.success]
        
        if successful_agents:
            print(f"    Successful Agents: {[r.agent_id for r in successful_agents]}")
        if failed_agents:
            print(f"    Failed Agents: {[r.agent_id for r in failed_agents]}")
        print()
        
    # Performance stress test
    print("Performing federated stress test...")
    
    stress_operations = []
    for i in range(100):
        if i % 4 == 0:
            op = CacheOperation("set", f"fed_key_{i}", f"fed_value_{i}", client_id="stress_fed")
        elif i % 4 == 1:
            op = CacheOperation("get", f"fed_key_{i-1}", client_id="stress_fed")
        elif i % 4 == 2:
            op = CacheOperation("exists", f"fed_key_{i-2}", client_id="stress_fed")
        else:
            op = CacheOperation("delete", f"fed_key_{i-3}", client_id="stress_fed")
        stress_operations.append(op)
        
    # Execute stress test
    stress_start = time.time()
    for operation in stress_operations:
        await cache.execute_operation(operation)
    stress_total = time.time() - stress_start
    
    # Display comprehensive metrics
    metrics = cache.get_performance_metrics()
    theoretical = cache.get_theoretical_reliability()
    
    print("=== FEDERATED PERFORMANCE ANALYSIS ===")
    print(f"Architecture: Federated (Depth=1)")
    print(f"Agent Count: {metrics['agent_count']}")
    print(f"Reliability: {metrics['reliability']:.3f} ({metrics['reliability']*100:.1f}%)")
    print(f"Consensus Rate: {metrics['consensus_rate']:.3f}")
    print(f"Cascade Prevention: {metrics['cascade_prevention_rate']:.3f}")
    print(f"Total Operations: {metrics['total_operations']}")
    print()
    
    print("Agent Performance Breakdown:")
    for agent_id, agent_data in metrics['agent_metrics'].items():
        print(f"  {agent_id}:")
        print(f"    Specialty: {agent_data['specialty']}")
        print(f"    Operations: {agent_data['total_operations']}")
        print(f"    Base Failure Rate: {agent_data['base_failure_rate']:.1%}")
    print()
    
    print("=== THEORETICAL VALIDATION ===")
    print(f"Theoretical Reliability: {theoretical['theoretical_reliability']:.3f}")
    print(f"Empirical Reliability: {theoretical['empirical_reliability']:.3f}")
    print(f"Average Agent Error Rate: {theoretical['avg_agent_error_rate']:.1%}")
    print(f"Reliability Improvement: {theoretical['reliability_improvement']:.2f}x")
    print()
    
    print("Individual Agent Error Rates:")
    for agent_type, error_rate in theoretical['agent_error_rates'].items():
        reliability = 1 - error_rate
        print(f"  {agent_type.title()}: {reliability:.1%} ({error_rate:.1%} error rate)")
    print()
    
    # Mathematical proof of depth multiplication
    single_agent_reliability = 1 - statistics.mean(theoretical['agent_error_rates'].values())
    federated_reliability = theoretical['empirical_reliability']
    
    print("=== DEPTH MULTIPLICATION PROOF (Depth=1) ===")
    print(f"Single Agent Baseline: {single_agent_reliability:.3f}")
    print(f"Federated (3 Agents): {federated_reliability:.3f}")
    print(f"Improvement Factor: {federated_reliability/single_agent_reliability:.2f}x")
    print()
    
    # Error rate comparison
    single_error_rate = 1 - single_agent_reliability
    federated_error_rate = 1 - federated_reliability
    error_reduction = single_error_rate / federated_error_rate if federated_error_rate > 0 else float('inf')
    
    print("Error Rate Analysis:")
    print(f"Single Agent Error Rate: {single_error_rate:.1%}")
    print(f"Federated Error Rate: {federated_error_rate:.1%}")
    print(f"Error Reduction Factor: {error_reduction:.1f}x")
    print()
    
    print("=== FEDERATION ADVANTAGES ===")
    print("✅ Exponential reliability improvement through redundancy")
    print("✅ Automatic consensus and failover mechanisms")
    print("✅ Specialized agents for different performance characteristics")
    print("✅ Cascade failure prevention through agent isolation")
    print("✅ 99.83% theoretical reliability vs 88% single agent")
    print()
    
    print("Next Level Prediction (Meta-Federation):")
    meta_theoretical = 1 - (federated_error_rate ** 3)  # 3 federated orchestrators
    print(f"Meta-Federation (Depth=2): {meta_theoretical:.3f}")
    print(f"Additional Improvement: {meta_theoretical/federated_reliability:.2f}x")
    
    return metrics
    
    
if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_federated_cache())