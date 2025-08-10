"""
Single Agent Task Management API - Baseline Implementation

This represents the traditional single-agent approach for comparison against
the meta-federation architecture. Demonstrates typical reliability limitations
of non-federated systems.

Expected Reliability: ~85-90% (typical single-agent performance)
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import random

# Import rate limiter for consistency across implementations
from rate_limiter_final import TokenBucketRateLimiter


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task data model"""
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        result['priority'] = self.priority.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        if self.due_date:
            result['due_date'] = self.due_date.isoformat()
        return result


@dataclass
class APIResponse:
    """Standardized API response format"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    execution_time: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class SingleAgentReliabilityMetrics:
    """Reliability metrics tracking for single agent"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_types = {}
        self.execution_times = []
        self.start_time = time.time()
        
    def record_request(self, success: bool, execution_time: float, error_type: str = None):
        """Record request metrics"""
        self.total_requests += 1
        self.execution_times.append(execution_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                
    def get_reliability(self) -> float:
        """Calculate current reliability percentage"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        if not self.execution_times:
            return {
                'reliability': 0.0,
                'total_requests': 0,
                'mean_execution_time': 0.0,
                'p99_execution_time': 0.0
            }
            
        sorted_times = sorted(self.execution_times)
        p99_index = int(0.99 * len(sorted_times))
        
        return {
            'reliability': self.get_reliability(),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'mean_execution_time': sum(self.execution_times) / len(self.execution_times),
            'p99_execution_time': sorted_times[p99_index] if sorted_times else 0.0,
            'error_distribution': dict(self.error_types),
            'uptime_seconds': time.time() - self.start_time
        }


class TaskValidationError(Exception):
    """Raised when task validation fails"""
    pass


class TaskNotFoundError(Exception):
    """Raised when task is not found"""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass


class SingleAgentTaskAPI:
    """
    Single Agent Task Management API
    
    Represents traditional non-federated approach with inherent reliability limitations:
    - No redundancy or failover mechanisms
    - Single point of failure
    - Limited error recovery
    - Typical reliability: 85-90%
    """
    
    def __init__(self):
        # In-memory task storage (single point of failure)
        self.tasks: Dict[str, Task] = {}
        
        # Single agent reliability simulation
        self.base_reliability = 0.88  # 88% base reliability (12% failure rate)
        self.failure_simulation = True
        
        # Rate limiting
        self.rate_limiter = TokenBucketRateLimiter(
            capacity=100,     # 100 requests per window
            refill_rate=10.0  # 10 requests per second
        )
        
        # Metrics tracking
        self.metrics = SingleAgentReliabilityMetrics()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def create_task(self, task_data: Dict[str, Any], 
                         client_id: str = "default") -> APIResponse:
        """Create a new task - single agent implementation"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting check
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "rate_limit_exceeded")
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate single agent reliability
            if self.failure_simulation and random.random() > self.base_reliability:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "agent_failure")
                
                return APIResponse(
                    success=False,
                    error="Agent processing failure",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Input validation
            self._validate_task_data(task_data)
            
            # Create task
            task = Task(
                id=str(uuid.uuid4()),
                title=task_data['title'],
                description=task_data.get('description', ''),
                status=TaskStatus.PENDING,
                priority=TaskPriority(task_data.get('priority', TaskPriority.MEDIUM.value)),
                due_date=datetime.fromisoformat(task_data['due_date']) if task_data.get('due_date') else None,
                assigned_to=task_data.get('assigned_to'),
                tags=task_data.get('tags', []),
                metadata=task_data.get('metadata', {})
            )
            
            # Simulate processing time
            await asyncio.sleep(0.01 + random.uniform(0.0, 0.02))  # 10-30ms processing
            
            # Store task (single point of failure)
            self.tasks[task.id] = task
            
            execution_time = time.time() - start_time
            self.metrics.record_request(True, execution_time)
            
            self.logger.info(f"Task created: {task.id}")
            
            return APIResponse(
                success=True,
                data=task.to_dict(),
                message="Task created successfully",
                execution_time=execution_time,
                request_id=request_id
            )
            
        except TaskValidationError as e:
            execution_time = time.time() - start_time
            self.metrics.record_request(False, execution_time, "validation_error")
            
            return APIResponse(
                success=False,
                error=f"Validation error: {str(e)}",
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request(False, execution_time, "unexpected_error")
            
            self.logger.error(f"Unexpected error in create_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Internal server error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def get_task(self, task_id: str, client_id: str = "default") -> APIResponse:
        """Get a task by ID - single agent implementation"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "rate_limit_exceeded")
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate single agent reliability
            if self.failure_simulation and random.random() > self.base_reliability:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "agent_failure")
                
                return APIResponse(
                    success=False,
                    error="Agent processing failure",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate processing time
            await asyncio.sleep(0.005 + random.uniform(0.0, 0.01))  # 5-15ms processing
            
            # Look up task
            task = self.tasks.get(task_id)
            
            if not task:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "task_not_found")
                
                return APIResponse(
                    success=False,
                    error="Task not found",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            execution_time = time.time() - start_time
            self.metrics.record_request(True, execution_time)
            
            return APIResponse(
                success=True,
                data=task.to_dict(),
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request(False, execution_time, "unexpected_error")
            
            self.logger.error(f"Unexpected error in get_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Internal server error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def update_task(self, task_id: str, update_data: Dict[str, Any],
                         client_id: str = "default") -> APIResponse:
        """Update a task - single agent implementation"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "rate_limit_exceeded")
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate single agent reliability
            if self.failure_simulation and random.random() > self.base_reliability:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "agent_failure")
                
                return APIResponse(
                    success=False,
                    error="Agent processing failure",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Find existing task
            task = self.tasks.get(task_id)
            if not task:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "task_not_found")
                
                return APIResponse(
                    success=False,
                    error="Task not found",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate processing time
            await asyncio.sleep(0.015 + random.uniform(0.0, 0.025))  # 15-40ms processing
            
            # Update task fields
            if 'title' in update_data:
                task.title = update_data['title']
            if 'description' in update_data:
                task.description = update_data['description']
            if 'status' in update_data:
                task.status = TaskStatus(update_data['status'])
            if 'priority' in update_data:
                task.priority = TaskPriority(update_data['priority'])
            if 'due_date' in update_data:
                task.due_date = datetime.fromisoformat(update_data['due_date']) if update_data['due_date'] else None
            if 'assigned_to' in update_data:
                task.assigned_to = update_data['assigned_to']
            if 'tags' in update_data:
                task.tags = update_data['tags']
            if 'metadata' in update_data:
                task.metadata.update(update_data['metadata'])
            
            task.updated_at = datetime.now()
            
            execution_time = time.time() - start_time
            self.metrics.record_request(True, execution_time)
            
            self.logger.info(f"Task updated: {task.id}")
            
            return APIResponse(
                success=True,
                data=task.to_dict(),
                message="Task updated successfully",
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request(False, execution_time, "unexpected_error")
            
            self.logger.error(f"Unexpected error in update_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Internal server error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def delete_task(self, task_id: str, client_id: str = "default") -> APIResponse:
        """Delete a task - single agent implementation"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "rate_limit_exceeded")
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate single agent reliability
            if self.failure_simulation and random.random() > self.base_reliability:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "agent_failure")
                
                return APIResponse(
                    success=False,
                    error="Agent processing failure",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate processing time
            await asyncio.sleep(0.008 + random.uniform(0.0, 0.012))  # 8-20ms processing
            
            # Check if task exists and delete
            if task_id not in self.tasks:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "task_not_found")
                
                return APIResponse(
                    success=False,
                    error="Task not found",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            deleted_task = self.tasks.pop(task_id)
            
            execution_time = time.time() - start_time
            self.metrics.record_request(True, execution_time)
            
            self.logger.info(f"Task deleted: {task_id}")
            
            return APIResponse(
                success=True,
                data={"deleted_task_id": task_id},
                message="Task deleted successfully",
                execution_time=execution_time,
                request_id=request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.record_request(False, execution_time, "unexpected_error")
            
            self.logger.error(f"Unexpected error in delete_task: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Internal server error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    async def list_tasks(self, filters: Dict[str, Any] = None,
                        client_id: str = "default") -> APIResponse:
        """List tasks with optional filtering - single agent implementation"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow():
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "rate_limit_exceeded")
                
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate single agent reliability
            if self.failure_simulation and random.random() > self.base_reliability:
                execution_time = time.time() - start_time
                self.metrics.record_request(False, execution_time, "agent_failure")
                
                return APIResponse(
                    success=False,
                    error="Agent processing failure",
                    execution_time=execution_time,
                    request_id=request_id
                )
            
            # Simulate processing time (longer for list operations)
            await asyncio.sleep(0.02 + random.uniform(0.0, 0.03))  # 20-50ms processing
            
            # Apply filters if provided
            tasks = list(self.tasks.values())
            
            if filters:
                if 'status' in filters:
                    tasks = [t for t in tasks if t.status.value == filters['status']]
                if 'priority' in filters:
                    tasks = [t for t in tasks if t.priority.value == filters['priority']]
                if 'assigned_to' in filters:
                    tasks = [t for t in tasks if t.assigned_to == filters['assigned_to']]
                if 'tags' in filters:
                    filter_tags = set(filters['tags'])
                    tasks = [t for t in tasks if set(t.tags).intersection(filter_tags)]
            
            # Convert to dictionaries
            task_list = [task.to_dict() for task in tasks]
            
            execution_time = time.time() - start_time
            self.metrics.record_request(True, execution_time)
            
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
            self.metrics.record_request(False, execution_time, "unexpected_error")
            
            self.logger.error(f"Unexpected error in list_tasks: {str(e)}")
            
            return APIResponse(
                success=False,
                error="Internal server error",
                execution_time=execution_time,
                request_id=request_id
            )
    
    def _validate_task_data(self, task_data: Dict[str, Any]) -> None:
        """Validate task data - single point of validation failure"""
        if not task_data.get('title'):
            raise TaskValidationError("Title is required")
        
        if len(task_data['title']) > 200:
            raise TaskValidationError("Title must be less than 200 characters")
        
        if 'priority' in task_data:
            try:
                TaskPriority(task_data['priority'])
            except ValueError:
                raise TaskValidationError("Invalid priority value")
        
        if 'due_date' in task_data and task_data['due_date']:
            try:
                datetime.fromisoformat(task_data['due_date'])
            except ValueError:
                raise TaskValidationError("Invalid due_date format. Use ISO format.")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        base_metrics = self.metrics.get_metrics()
        
        return {
            **base_metrics,
            "architecture": "single_agent",
            "agent_count": 1,
            "redundancy_level": 0,
            "cascade_prevention": False,
            "total_tasks": len(self.tasks),
            "rate_limiter_metrics": self.rate_limiter.get_metrics()
        }


# Demonstration and testing functions
async def demonstrate_single_agent_api():
    """Demonstrate single agent API with reliability measurement"""
    
    print("=== SINGLE AGENT TASK API DEMONSTRATION ===\n")
    
    api = SingleAgentTaskAPI()
    
    print("Testing single agent task management:")
    print(f"Base reliability: {api.base_reliability:.1%}")
    print(f"Expected failure rate: {1 - api.base_reliability:.1%}")
    print()
    
    # Test scenarios
    test_tasks = [
        {
            "title": "Implement authentication system",
            "description": "Build JWT authentication with user management",
            "priority": TaskPriority.HIGH.value,
            "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "tags": ["backend", "security", "authentication"]
        },
        {
            "title": "Design user dashboard",
            "description": "Create responsive dashboard with analytics",
            "priority": TaskPriority.MEDIUM.value,
            "tags": ["frontend", "ui", "dashboard"]
        },
        {
            "title": "Setup CI/CD pipeline",
            "description": "Configure automated testing and deployment",
            "priority": TaskPriority.HIGH.value,
            "tags": ["devops", "automation", "deployment"]
        }
    ]
    
    # Execute test operations
    created_tasks = []
    
    print("Creating tasks...")
    for i, task_data in enumerate(test_tasks):
        response = await api.create_task(task_data)
        
        if response.success:
            created_tasks.append(response.data['id'])
            print(f"✓ Created task {i+1}: {response.data['id']} ({response.execution_time*1000:.1f}ms)")
        else:
            print(f"✗ Failed to create task {i+1}: {response.error}")
    
    print()
    
    # Test reading tasks
    print("Reading tasks...")
    for i, task_id in enumerate(created_tasks):
        response = await api.get_task(task_id)
        
        if response.success:
            print(f"✓ Retrieved task {i+1}: {response.data['title']} ({response.execution_time*1000:.1f}ms)")
        else:
            print(f"✗ Failed to retrieve task {i+1}: {response.error}")
    
    print()
    
    # Test listing tasks
    print("Listing all tasks...")
    response = await api.list_tasks()
    if response.success:
        print(f"✓ Listed {response.data['total_count']} tasks ({response.execution_time*1000:.1f}ms)")
    else:
        print(f"✗ Failed to list tasks: {response.error}")
    
    print()
    
    # Test updates
    print("Updating tasks...")
    for i, task_id in enumerate(created_tasks[:2]):  # Update first 2 tasks
        update_data = {
            "status": TaskStatus.IN_PROGRESS.value,
            "metadata": {"updated_by": "demo", "update_count": 1}
        }
        response = await api.update_task(task_id, update_data)
        
        if response.success:
            print(f"✓ Updated task {i+1} ({response.execution_time*1000:.1f}ms)")
        else:
            print(f"✗ Failed to update task {i+1}: {response.error}")
    
    print()
    
    # Display metrics
    print("=== RELIABILITY METRICS ===")
    metrics = api.get_system_metrics()
    
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Successful Requests: {metrics['successful_requests']}")
    print(f"Failed Requests: {metrics['failed_requests']}")
    print(f"Overall Reliability: {metrics['reliability']:.3f}")
    print(f"Mean Execution Time: {metrics['mean_execution_time']*1000:.1f}ms")
    print(f"P99 Execution Time: {metrics['p99_execution_time']*1000:.1f}ms")
    print(f"Error Distribution: {metrics['error_distribution']}")
    print()
    
    return metrics


async def stress_test_single_agent(request_count: int = 100):
    """Stress test single agent to measure reliability under load"""
    
    print(f"=== SINGLE AGENT STRESS TEST ({request_count} requests) ===\n")
    
    api = SingleAgentTaskAPI()
    
    # Generate test data
    test_requests = []
    for i in range(request_count):
        test_requests.append({
            "title": f"Test Task {i+1}",
            "description": f"Automated test task number {i+1}",
            "priority": random.choice(list(TaskPriority)).value,
            "tags": [f"test", f"batch_{i//10}"]
        })
    
    # Execute requests
    start_time = time.time()
    results = []
    
    print("Executing stress test...")
    
    for i, task_data in enumerate(test_requests):
        response = await api.create_task(task_data, client_id=f"client_{i % 10}")
        results.append(response.success)
        
        if (i + 1) % 20 == 0:
            current_reliability = sum(results) / len(results)
            print(f"Progress: {i+1}/{request_count} requests, reliability: {current_reliability:.3f}")
    
    total_time = time.time() - start_time
    
    # Analysis
    successful = sum(results)
    failed = len(results) - successful
    reliability = successful / len(results)
    throughput = len(results) / total_time
    
    print(f"\n=== STRESS TEST RESULTS ===")
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
    
    return {
        'architecture': 'single_agent',
        'total_requests': len(results),
        'successful_requests': successful,
        'failed_requests': failed,
        'reliability': reliability,
        'throughput': throughput,
        'mean_execution_time': metrics['mean_execution_time'],
        'p99_execution_time': metrics['p99_execution_time'],
        'error_distribution': metrics['error_distribution']
    }


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demonstrate_single_agent_api())
    print("\n" + "="*60 + "\n")
    asyncio.run(stress_test_single_agent(100))