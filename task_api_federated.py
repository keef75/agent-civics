"""
Level 1 Federated Task Management API - 3 Agent Implementation

This implementation demonstrates Level 1 federation with 3 specialized agents
working together to achieve higher reliability through redundancy and specialization.

Expected Reliability: ~95-98% (significant improvement over single agent)
Architecture: 3 specialized agents with intelligent routing and failover
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import random
import statistics

# Reuse common structures from single agent
from task_api_single import TaskStatus, TaskPriority, Task, APIResponse
from rate_limiter_final import TokenBucketRateLimiter


class AgentType(Enum):
    """Specialized agent types for Level 1 federation"""
    CRUD_AGENT = "crud"           # Create, Read, Update, Delete operations
    VALIDATION_AGENT = "validation"  # Data validation and business rules
    QUERY_AGENT = "query"         # Complex queries and filtering


@dataclass
class AgentResponse:
    """Response from individual agent"""
    agent_type: AgentType
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 1.0  # Agent's confidence in result


class FederationReliabilityMetrics:
    """Reliability metrics for federated system"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.partial_success_requests = 0
        
        # Agent-specific metrics
        self.agent_metrics = {
            AgentType.CRUD_AGENT: {'success': 0, 'failure': 0, 'execution_times': []},
            AgentType.VALIDATION_AGENT: {'success': 0, 'failure': 0, 'execution_times': []},
            AgentType.QUERY_AGENT: {'success': 0, 'failure': 0, 'execution_times': []}
        }
        
        # Federation metrics
        self.failover_events = 0
        self.consensus_failures = 0
        self.execution_times = []
        self.start_time = time.time()
        
    def record_agent_result(self, agent_type: AgentType, success: bool, execution_time: float):
        """Record individual agent result"""
        if success:
            self.agent_metrics[agent_type]['success'] += 1
        else:
            self.agent_metrics[agent_type]['failure'] += 1
        
        self.agent_metrics[agent_type]['execution_times'].append(execution_time)
    
    def record_request_result(self, success: bool, partial: bool, execution_time: float,
                            failover_occurred: bool = False):
        """Record overall request result"""
        self.total_requests += 1
        self.execution_times.append(execution_time)
        
        if success:
            self.successful_requests += 1
        elif partial:
            self.partial_success_requests += 1
        else:
            self.failed_requests += 1
        
        if failover_occurred:
            self.failover_events += 1
    
    def get_overall_reliability(self) -> float:
        """Calculate overall system reliability"""
        if self.total_requests == 0:
            return 0.0
        
        # Count partial successes as 70% of full success
        effective_successes = self.successful_requests + (self.partial_success_requests * 0.7)
        return effective_successes / self.total_requests
    
    def get_agent_reliability(self, agent_type: AgentType) -> float:
        """Calculate individual agent reliability"""
        metrics = self.agent_metrics[agent_type]
        total = metrics['success'] + metrics['failure']
        
        if total == 0:
            return 0.0
        
        return metrics['success'] / total
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for analysis"""
        agent_reliabilities = {}
        agent_performance = {}
        
        for agent_type in AgentType:
            agent_reliabilities[agent_type.value] = self.get_agent_reliability(agent_type)
            
            times = self.agent_metrics[agent_type]['execution_times']
            if times:
                agent_performance[agent_type.value] = {
                    'mean_time': statistics.mean(times),
                    'p99_time': sorted(times)[int(0.99 * len(times))] if len(times) > 1 else times[0]
                }
            else:
                agent_performance[agent_type.value] = {'mean_time': 0.0, 'p99_time': 0.0}
        
        overall_times = self.execution_times
        
        return {
            'architecture': 'level1_federation',
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'partial_success_requests': self.partial_success_requests,
            'failed_requests': self.failed_requests,
            'overall_reliability': self.get_overall_reliability(),
            'agent_reliabilities': agent_reliabilities,
            'agent_performance': agent_performance,
            'failover_events': self.failover_events,
            'failover_rate': self.failover_events / max(1, self.total_requests),
            'consensus_failures': self.consensus_failures,
            'mean_execution_time': statistics.mean(overall_times) if overall_times else 0.0,
            'p99_execution_time': sorted(overall_times)[int(0.99 * len(overall_times))] if len(overall_times) > 1 else (overall_times[0] if overall_times else 0.0),
            'uptime_seconds': time.time() - self.start_time,
            'federation_efficiency': self._calculate_federation_efficiency()
        }
    
    def _calculate_federation_efficiency(self) -> float:
        """Calculate how efficiently the federation is working"""
        if self.total_requests == 0:
            return 0.0
        
        # Efficiency = (successful + partial) / total - failover_penalty
        base_efficiency = (self.successful_requests + self.partial_success_requests) / self.total_requests
        failover_penalty = (self.failover_events / self.total_requests) * 0.1  # 10% penalty per failover
        
        return max(0.0, base_efficiency - failover_penalty)


class BaseAgent:
    """Base class for federation agents"""
    
    def __init__(self, agent_type: AgentType, reliability: float = 0.90):
        self.agent_type = agent_type
        self.base_reliability = reliability
        self.tasks = {}  # Agent's view of tasks
        
    async def execute(self, operation: str, *args, **kwargs) -> AgentResponse:
        """Execute operation with simulated reliability"""
        start_time = time.time()
        
        try:
            # Simulate processing time based on agent type
            processing_times = {
                AgentType.CRUD_AGENT: (0.01, 0.03),      # 10-30ms
                AgentType.VALIDATION_AGENT: (0.005, 0.015),  # 5-15ms
                AgentType.QUERY_AGENT: (0.02, 0.05)      # 20-50ms
            }
            
            min_time, max_time = processing_times[self.agent_type]
            await asyncio.sleep(min_time + random.uniform(0, max_time - min_time))
            
            # Simulate agent reliability
            if random.random() <= self.base_reliability:
                # Success path
                if operation == 'create':
                    return await self._create_task(*args, **kwargs)
                elif operation == 'read':
                    return await self._read_task(*args, **kwargs)
                elif operation == 'update':
                    return await self._update_task(*args, **kwargs)
                elif operation == 'delete':
                    return await self._delete_task(*args, **kwargs)
                elif operation == 'list':
                    return await self._list_tasks(*args, **kwargs)
                elif operation == 'validate':
                    return await self._validate_data(*args, **kwargs)
                else:
                    return AgentResponse(
                        agent_type=self.agent_type,
                        success=False,
                        error=f"Unknown operation: {operation}",
                        execution_time=time.time() - start_time
                    )
            else:
                # Failure path
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    error=f"{self.agent_type.value} agent processing failure",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
                
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"Agent error: {str(e)}",
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    
    async def _create_task(self, task_data: Dict[str, Any]) -> AgentResponse:
        """Create task operation"""
        task_id = str(uuid.uuid4())
        
        if self.agent_type == AgentType.CRUD_AGENT:
            # CRUD agent creates and stores the task
            task = Task(
                id=task_id,
                title=task_data['title'],
                description=task_data.get('description', ''),
                status=TaskStatus.PENDING,
                priority=TaskPriority(task_data.get('priority', TaskPriority.MEDIUM.value)),
                due_date=datetime.fromisoformat(task_data['due_date']) if task_data.get('due_date') else None,
                assigned_to=task_data.get('assigned_to'),
                tags=task_data.get('tags', []),
                metadata=task_data.get('metadata', {})
            )
            
            self.tasks[task_id] = task
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=task.to_dict(),
                confidence=0.95
            )
        else:
            # Other agents don't handle creation directly
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"{self.agent_type.value} agent cannot create tasks",
                confidence=0.0
            )
    
    async def _read_task(self, task_id: str) -> AgentResponse:
        """Read task operation"""
        task = self.tasks.get(task_id)
        
        if task:
            confidence = 0.98 if self.agent_type == AgentType.CRUD_AGENT else 0.85
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=task.to_dict(),
                confidence=confidence
            )
        else:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error="Task not found",
                confidence=0.9  # High confidence in "not found"
            )
    
    async def _update_task(self, task_id: str, update_data: Dict[str, Any]) -> AgentResponse:
        """Update task operation"""
        if self.agent_type != AgentType.CRUD_AGENT:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"{self.agent_type.value} agent cannot update tasks",
                confidence=0.0
            )
        
        task = self.tasks.get(task_id)
        if not task:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error="Task not found",
                confidence=0.9
            )
        
        # Update task fields
        if 'title' in update_data:
            task.title = update_data['title']
        if 'description' in update_data:
            task.description = update_data['description']
        if 'status' in update_data:
            task.status = TaskStatus(update_data['status'])
        if 'priority' in update_data:
            task.priority = TaskPriority(update_data['priority'])
        # ... other fields
        
        task.updated_at = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data=task.to_dict(),
            confidence=0.95
        )
    
    async def _delete_task(self, task_id: str) -> AgentResponse:
        """Delete task operation"""
        if self.agent_type != AgentType.CRUD_AGENT:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"{self.agent_type.value} agent cannot delete tasks",
                confidence=0.0
            )
        
        if task_id in self.tasks:
            deleted_task = self.tasks.pop(task_id)
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data={"deleted_task_id": task_id},
                confidence=0.95
            )
        else:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error="Task not found",
                confidence=0.9
            )
    
    async def _list_tasks(self, filters: Dict[str, Any] = None) -> AgentResponse:
        """List tasks operation"""
        if self.agent_type == AgentType.QUERY_AGENT:
            # Query agent is best at complex filtering
            tasks = list(self.tasks.values())
            
            # Apply filters
            if filters:
                if 'status' in filters:
                    tasks = [t for t in tasks if t.status.value == filters['status']]
                if 'priority' in filters:
                    tasks = [t for t in tasks if t.priority.value == filters['priority']]
                # ... other filters
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=[task.to_dict() for task in tasks],
                confidence=0.92
            )
        elif self.agent_type == AgentType.CRUD_AGENT:
            # CRUD agent can do basic listing
            tasks = [task.to_dict() for task in self.tasks.values()]
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=tasks,
                confidence=0.88
            )
        else:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"{self.agent_type.value} agent cannot list tasks",
                confidence=0.0
            )
    
    async def _validate_data(self, data: Dict[str, Any]) -> AgentResponse:
        """Validate data operation"""
        if self.agent_type != AgentType.VALIDATION_AGENT:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"{self.agent_type.value} agent cannot validate data",
                confidence=0.0
            )
        
        # Validation logic
        errors = []
        
        if 'title' in data:
            if not data['title'] or not data['title'].strip():
                errors.append("Title is required")
            elif len(data['title']) > 200:
                errors.append("Title must be less than 200 characters")
        
        if 'priority' in data:
            try:
                TaskPriority(data['priority'])
            except ValueError:
                errors.append("Invalid priority value")
        
        if 'due_date' in data and data['due_date']:
            try:
                datetime.fromisoformat(data['due_date'])
            except ValueError:
                errors.append("Invalid due_date format")
        
        if errors:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=f"Validation errors: {', '.join(errors)}",
                confidence=0.95
            )
        else:
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data={"validation": "passed"},
                confidence=0.98
            )


class FederationOrchestrator:
    """Level 1 Federation Orchestrator managing 3 specialized agents"""
    
    def __init__(self):
        # Create 3 specialized agents with different reliability characteristics
        self.agents = {
            AgentType.CRUD_AGENT: BaseAgent(AgentType.CRUD_AGENT, reliability=0.92),
            AgentType.VALIDATION_AGENT: BaseAgent(AgentType.VALIDATION_AGENT, reliability=0.95),
            AgentType.QUERY_AGENT: BaseAgent(AgentType.QUERY_AGENT, reliability=0.90)
        }
        
        # Shared task storage across agents (eventual consistency)
        self.shared_tasks = {}
        
        # Rate limiting
        self.rate_limiter = TokenBucketRateLimiter(
            capacity=150,     # Higher capacity than single agent
            refill_rate=15.0  # Higher refill rate
        )
        
        # Metrics
        self.metrics = FederationReliabilityMetrics()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def create_task(self, task_data: Dict[str, Any], 
                         client_id: str = "default") -> APIResponse:
        """Create task using federated approach"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Step 1: Validate data with validation agent
            validation_response = await self.agents[AgentType.VALIDATION_AGENT].execute(
                'validate', task_data
            )
            
            self.metrics.record_agent_result(
                AgentType.VALIDATION_AGENT, 
                validation_response.success, 
                validation_response.execution_time
            )
            
            if not validation_response.success:
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error=f"Validation failed: {validation_response.error}",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Step 2: Create task with CRUD agent
            crud_response = await self.agents[AgentType.CRUD_AGENT].execute(
                'create', task_data
            )
            
            self.metrics.record_agent_result(
                AgentType.CRUD_AGENT,
                crud_response.success,
                crud_response.execution_time
            )
            
            failover_occurred = False
            
            if not crud_response.success:
                # Failover: try a different approach or graceful degradation
                self.logger.warning(f"CRUD agent failed, implementing failover: {crud_response.error}")
                
                # Simple failover: create task directly in shared storage
                task_id = str(uuid.uuid4())
                task = Task(
                    id=task_id,
                    title=task_data['title'],
                    description=task_data.get('description', ''),
                    status=TaskStatus.PENDING,
                    priority=TaskPriority(task_data.get('priority', TaskPriority.MEDIUM.value)),
                    due_date=datetime.fromisoformat(task_data['due_date']) if task_data.get('due_date') else None,
                    assigned_to=task_data.get('assigned_to'),
                    tags=task_data.get('tags', []),
                    metadata=task_data.get('metadata', {})
                )
                
                self.shared_tasks[task_id] = task
                
                # Sync to agents
                await self._sync_task_to_agents(task)
                
                crud_response = AgentResponse(
                    agent_type=AgentType.CRUD_AGENT,
                    success=True,
                    data=task.to_dict(),
                    confidence=0.85  # Lower confidence due to failover
                )
                
                failover_occurred = True
            else:
                # Success: sync to shared storage and other agents
                task = Task(**crud_response.data)  # Reconstruct task object
                self.shared_tasks[task.id] = task
                await self._sync_task_to_agents(task)
            
            execution_time = time.time() - start_time
            self.metrics.record_request_result(True, False, execution_time, failover_occurred)
            
            return APIResponse(
                success=True,
                data=crud_response.data,
                message="Task created successfully" + (" (with failover)" if failover_occurred else ""),
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            self.logger.error(f"Federation error in create_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Federation processing error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def get_task(self, task_id: str, client_id: str = "default") -> APIResponse:
        """Get task using federated approach with consensus"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Query multiple agents for consensus
            agent_responses = []
            
            # Try CRUD and Query agents (both can read)
            read_agents = [AgentType.CRUD_AGENT, AgentType.QUERY_AGENT]
            
            for agent_type in read_agents:
                response = await self.agents[agent_type].execute('read', task_id)
                agent_responses.append(response)
                
                self.metrics.record_agent_result(
                    agent_type,
                    response.success,
                    response.execution_time
                )
            
            # Consensus logic: prefer successful responses with higher confidence
            successful_responses = [r for r in agent_responses if r.success]
            
            if successful_responses:
                # Choose response with highest confidence
                best_response = max(successful_responses, key=lambda r: r.confidence)
                
                execution_time = time.time() - start_time
                self.metrics.record_request_result(True, False, execution_time)
                
                return APIResponse(
                    success=True,
                    data=best_response.data,
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Failover: check shared storage
            task = self.shared_tasks.get(task_id)
            if task:
                execution_time = time.time() - start_time
                self.metrics.record_request_result(True, False, execution_time, failover_occurred=True)
                
                return APIResponse(
                    success=True,
                    data=task.to_dict(),
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Complete failure
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            return APIResponse(
                success=False,
                error="Task not found",
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            self.logger.error(f"Federation error in get_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Federation processing error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def update_task(self, task_id: str, update_data: Dict[str, Any],
                         client_id: str = "default") -> APIResponse:
        """Update task using federated approach"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Step 1: Validate update data
            validation_response = await self.agents[AgentType.VALIDATION_AGENT].execute(
                'validate', update_data
            )
            
            self.metrics.record_agent_result(
                AgentType.VALIDATION_AGENT,
                validation_response.success,
                validation_response.execution_time
            )
            
            if not validation_response.success:
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error=f"Validation failed: {validation_response.error}",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Step 2: Update with CRUD agent
            crud_response = await self.agents[AgentType.CRUD_AGENT].execute(
                'update', task_id, update_data
            )
            
            self.metrics.record_agent_result(
                AgentType.CRUD_AGENT,
                crud_response.success,
                crud_response.execution_time
            )
            
            failover_occurred = False
            
            if crud_response.success:
                # Update shared storage
                updated_task = Task(**crud_response.data)
                self.shared_tasks[task_id] = updated_task
                await self._sync_task_to_agents(updated_task)
                
            else:
                # Failover: try to update shared storage directly
                task = self.shared_tasks.get(task_id)
                if task:
                    # Apply updates manually
                    if 'title' in update_data:
                        task.title = update_data['title']
                    if 'description' in update_data:
                        task.description = update_data['description']
                    if 'status' in update_data:
                        task.status = TaskStatus(update_data['status'])
                    # ... other fields
                    
                    task.updated_at = datetime.now()
                    
                    await self._sync_task_to_agents(task)
                    
                    crud_response = AgentResponse(
                        agent_type=AgentType.CRUD_AGENT,
                        success=True,
                        data=task.to_dict(),
                        confidence=0.80
                    )
                    
                    failover_occurred = True
                else:
                    execution_time = time.time() - start_time
                    self.metrics.record_request_result(False, False, execution_time)
                    
                    return APIResponse(
                        success=False,
                        error="Task not found",
                        execution_time=execution_time,
                        request_id=request_id
                    )
            
            execution_time = time.time() - start_time
            self.metrics.record_request_result(True, False, execution_time, failover_occurred)
            
            return APIResponse(
                success=True,
                data=crud_response.data,
                message="Task updated successfully" + (" (with failover)" if failover_occurred else ""),
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            self.logger.error(f"Federation error in update_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Federation processing error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def delete_task(self, task_id: str, client_id: str = "default") -> APIResponse:
        """Delete task using federated approach"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Delete with CRUD agent
            crud_response = await self.agents[AgentType.CRUD_AGENT].execute(
                'delete', task_id
            )
            
            self.metrics.record_agent_result(
                AgentType.CRUD_AGENT,
                crud_response.success,
                crud_response.execution_time
            )
            
            failover_occurred = False
            
            if crud_response.success:
                # Remove from shared storage
                self.shared_tasks.pop(task_id, None)
                
                # Remove from other agents
                for agent_type, agent in self.agents.items():
                    if agent_type != AgentType.CRUD_AGENT:
                        agent.tasks.pop(task_id, None)
            else:
                # Failover: delete from shared storage
                if task_id in self.shared_tasks:
                    deleted_task = self.shared_tasks.pop(task_id)
                    
                    # Remove from agents
                    for agent in self.agents.values():
                        agent.tasks.pop(task_id, None)
                    
                    crud_response = AgentResponse(
                        agent_type=AgentType.CRUD_AGENT,
                        success=True,
                        data={"deleted_task_id": task_id},
                        confidence=0.80
                    )
                    
                    failover_occurred = True
                else:
                    execution_time = time.time() - start_time
                    self.metrics.record_request_result(False, False, execution_time)
                    
                    return APIResponse(
                        success=False,
                        error="Task not found",
                        execution_time=execution_time,
                        request_id=request_id
                    )
            
            execution_time = time.time() - start_time
            self.metrics.record_request_result(True, False, execution_time, failover_occurred)
            
            return APIResponse(
                success=True,
                data=crud_response.data,
                message="Task deleted successfully" + (" (with failover)" if failover_occurred else ""),
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            return APIResponse(
                success=False,
                error="Federation processing error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def list_tasks(self, filters: Dict[str, Any] = None,
                        client_id: str = "default") -> APIResponse:
        """List tasks using federated approach"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request_result(False, False, execution_time)
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Prefer Query agent for listing (it's specialized for this)
            query_response = await self.agents[AgentType.QUERY_AGENT].execute(
                'list', filters
            )
            
            self.metrics.record_agent_result(
                AgentType.QUERY_AGENT,
                query_response.success,
                query_response.execution_time
            )
            
            if query_response.success:
                execution_time = time.time() - start_time
                self.metrics.record_request_result(True, False, execution_time)
                
                return APIResponse(
                    success=True,
                    data={
                        "tasks": query_response.data,
                        "total_count": len(query_response.data),
                        "filters_applied": filters or {}
                    },
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Failover to CRUD agent
            crud_response = await self.agents[AgentType.CRUD_AGENT].execute(
                'list', filters
            )
            
            self.metrics.record_agent_result(
                AgentType.CRUD_AGENT,
                crud_response.success,
                crud_response.execution_time
            )
            
            if crud_response.success:
                execution_time = time.time() - start_time
                self.metrics.record_request_result(True, False, execution_time, failover_occurred=True)
                
                return APIResponse(
                    success=True,
                    data={
                        "tasks": crud_response.data,
                        "total_count": len(crud_response.data),
                        "filters_applied": filters or {}
                    },
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Final failover: use shared storage
            tasks = list(self.shared_tasks.values())
            
            # Apply filters manually
            if filters:
                if 'status' in filters:
                    tasks = [t for t in tasks if t.status.value == filters['status']]
                if 'priority' in filters:
                    tasks = [t for t in tasks if t.priority.value == filters['priority']]
                # ... other filters
            
            task_list = [task.to_dict() for task in tasks]
            
            execution_time = time.time() - start_time
            self.metrics.record_request_result(True, False, execution_time, failover_occurred=True)
            
            return APIResponse(
                success=True,
                data={
                    "tasks": task_list,
                    "total_count": len(task_list),
                    "filters_applied": filters or {}
                },
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request_result(False, False, execution_time)
            
            return APIResponse(
                success=False,
                error="Federation processing error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def _sync_task_to_agents(self, task: Task) -> None:
        """Synchronize task data across all agents (eventual consistency)"""
        for agent in self.agents.values():
            agent.tasks[task.id] = task
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        base_metrics = self.metrics.get_comprehensive_metrics()
        
        return {
            **base_metrics,
            "agent_count": len(self.agents),
            "federation_level": 1,
            "redundancy_enabled": True,
            "consensus_enabled": True,
            "total_tasks": len(self.shared_tasks),
            "rate_limiter_metrics": self.rate_limiter.get_metrics()
        }


# Demonstration and testing functions
async def demonstrate_federated_api():
    """Demonstrate federated API with reliability measurement"""
    
    print("=== LEVEL 1 FEDERATED TASK API DEMONSTRATION ===\n")
    
    api = FederationOrchestrator()
    
    print("Testing Level 1 federation with 3 specialized agents:")
    print("- CRUD Agent (92% reliability)")
    print("- Validation Agent (95% reliability)")  
    print("- Query Agent (90% reliability)")
    print("- Consensus and failover mechanisms enabled")
    print()
    
    # Test scenarios
    test_tasks = [
        {
            "title": "Implement federation system",
            "description": "Build multi-agent coordination system",
            "priority": TaskPriority.CRITICAL.value,
            "due_date": (datetime.now() + timedelta(days=5)).isoformat(),
            "tags": ["federation", "architecture", "reliability"]
        },
        {
            "title": "Performance optimization",
            "description": "Optimize system for high throughput",
            "priority": TaskPriority.HIGH.value,
            "tags": ["performance", "optimization", "scalability"]
        },
        {
            "title": "Monitoring dashboard",
            "description": "Create real-time system monitoring",
            "priority": TaskPriority.MEDIUM.value,
            "tags": ["monitoring", "dashboard", "observability"]
        }
    ]
    
    # Execute test operations
    created_tasks = []
    
    print("Creating tasks with federation...")
    for i, task_data in enumerate(test_tasks):
        response = await api.create_task(task_data)
        
        if response.success:
            created_tasks.append(response.data['id'])
            message = response.message or "Success"
            print(f"✓ Created task {i+1}: {response.data['id']} ({response.execution_time*1000:.1f}ms) - {message}")
        else:
            print(f"✗ Failed to create task {i+1}: {response.error}")
    
    print()
    
    # Test consensus reading
    print("Reading tasks with consensus...")
    for i, task_id in enumerate(created_tasks):
        response = await api.get_task(task_id)
        
        if response.success:
            print(f"✓ Retrieved task {i+1}: {response.data['title']} ({response.execution_time*1000:.1f}ms)")
        else:
            print(f"✗ Failed to retrieve task {i+1}: {response.error}")
    
    print()
    
    # Test federated listing
    print("Listing tasks with federated query...")
    response = await api.list_tasks()
    if response.success:
        print(f"✓ Listed {response.data['total_count']} tasks ({response.execution_time*1000:.1f}ms)")
    else:
        print(f"✗ Failed to list tasks: {response.error}")
    
    print()
    
    # Test updates with validation
    print("Updating tasks with validation...")
    for i, task_id in enumerate(created_tasks[:2]):
        update_data = {
            "status": TaskStatus.IN_PROGRESS.value,
            "metadata": {"updated_by": "federation", "validation_passed": True}
        }
        response = await api.update_task(task_id, update_data)
        
        if response.success:
            message = response.message or "Success"
            print(f"✓ Updated task {i+1} ({response.execution_time*1000:.1f}ms) - {message}")
        else:
            print(f"✗ Failed to update task {i+1}: {response.error}")
    
    print()
    
    # Display metrics
    print("=== FEDERATION RELIABILITY METRICS ===")
    metrics = api.get_system_metrics()
    
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Successful Requests: {metrics['successful_requests']}")
    print(f"Partial Success Requests: {metrics['partial_success_requests']}")
    print(f"Failed Requests: {metrics['failed_requests']}")
    print(f"Overall Reliability: {metrics['overall_reliability']:.3f}")
    print(f"Failover Events: {metrics['failover_events']}")
    print(f"Failover Rate: {metrics['failover_rate']:.3f}")
    print(f"Federation Efficiency: {metrics['federation_efficiency']:.3f}")
    print()
    
    print("Agent-Specific Reliabilities:")
    for agent_type, reliability in metrics['agent_reliabilities'].items():
        print(f"  {agent_type}: {reliability:.3f}")
    
    print()
    print("Agent-Specific Performance:")
    for agent_type, perf in metrics['agent_performance'].items():
        print(f"  {agent_type}: {perf['mean_time']*1000:.1f}ms mean, {perf['p99_time']*1000:.1f}ms P99")
    
    print()
    
    return metrics


async def stress_test_federated_system(request_count: int = 200):
    """Stress test federated system to measure reliability under load"""
    
    print(f"=== FEDERATED SYSTEM STRESS TEST ({request_count} requests) ===\n")
    
    api = FederationOrchestrator()
    
    # Generate test data
    test_requests = []
    for i in range(request_count):
        test_requests.append({
            "title": f"Federation Test Task {i+1}",
            "description": f"Automated federation test task number {i+1}",
            "priority": random.choice(list(TaskPriority)).value,
            "tags": [f"federation", f"test", f"batch_{i//20}"]
        })
    
    # Execute requests
    start_time = time.time()
    results = []
    
    print("Executing federated stress test...")
    
    for i, task_data in enumerate(test_requests):
        response = await api.create_task(task_data, client_id=f"client_{i % 20}")
        results.append(response.success)
        
        if (i + 1) % 40 == 0:
            current_reliability = sum(results) / len(results)
            print(f"Progress: {i+1}/{request_count} requests, reliability: {current_reliability:.3f}")
    
    total_time = time.time() - start_time
    
    # Analysis
    successful = sum(results)
    failed = len(results) - successful
    reliability = successful / len(results)
    throughput = len(results) / total_time
    
    print(f"\n=== FEDERATED STRESS TEST RESULTS ===")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Reliability: {reliability:.3f}")
    print(f"Throughput: {throughput:.1f} req/sec")
    print(f"Total Time: {total_time:.2f}s")
    
    # System metrics
    metrics = api.get_system_metrics()
    print(f"Mean Response Time: {metrics['mean_execution_time']*1000:.1f}ms")
    print(f"P99 Response Time: {metrics['p99_execution_time']*1000:.1f}ms")
    print(f"Failover Rate: {metrics['failover_rate']:.3f}")
    print(f"Federation Efficiency: {metrics['federation_efficiency']:.3f}")
    
    return {
        'architecture': 'level1_federation',
        'agent_count': 3,
        'total_requests': len(results),
        'successful_requests': successful,
        'failed_requests': failed,
        'reliability': reliability,
        'throughput': throughput,
        'mean_execution_time': metrics['mean_execution_time'],
        'p99_execution_time': metrics['p99_execution_time'],
        'failover_rate': metrics['failover_rate'],
        'federation_efficiency': metrics['federation_efficiency'],
        'agent_reliabilities': metrics['agent_reliabilities']
    }


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demonstrate_federated_api())
    print("\n" + "="*80 + "\n")
    asyncio.run(stress_test_federated_system(200))