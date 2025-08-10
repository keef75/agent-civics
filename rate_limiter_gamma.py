#!/usr/bin/env python3
"""
Defensive Rate Limiter Implementation using Token Bucket Algorithm

This implementation prioritizes defensive programming, extensive error handling,
and fault tolerance over performance optimization. Every possible failure mode
is anticipated and handled gracefully.

Author: Claude (Defensive Programming Specialist)
License: MIT
"""

import asyncio
import logging
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import sys
import traceback
from contextlib import contextmanager
import signal
import os


class RateLimiterError(Exception):
    """Base exception for all rate limiter errors."""
    pass


class ConfigurationError(RateLimiterError):
    """Raised when configuration parameters are invalid."""
    pass


class ResourceExhaustionError(RateLimiterError):
    """Raised when system resources are exhausted."""
    pass


class TimeoutError(RateLimiterError):
    """Raised when operations timeout."""
    pass


class CorruptionError(RateLimiterError):
    """Raised when internal state corruption is detected."""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateLimiterMetrics:
    """Comprehensive metrics for observability."""
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    errors_handled: int = 0
    timeouts: int = 0
    circuit_breaker_trips: int = 0
    average_response_time_ms: float = 0.0
    peak_concurrent_requests: int = 0
    current_concurrent_requests: int = 0
    memory_usage_bytes: int = 0
    corruption_events: int = 0
    recovery_events: int = 0
    last_reset_time: float = field(default_factory=time.time)
    
    def reset(self) -> None:
        """Reset all metrics to zero with defensive checks."""
        try:
            self.total_requests = 0
            self.allowed_requests = 0
            self.rejected_requests = 0
            self.errors_handled = 0
            self.timeouts = 0
            self.circuit_breaker_trips = 0
            self.average_response_time_ms = 0.0
            self.peak_concurrent_requests = 0
            self.current_concurrent_requests = 0
            self.memory_usage_bytes = 0
            self.corruption_events = 0
            self.recovery_events = 0
            self.last_reset_time = time.time()
        except Exception as e:
            # Even metric reset can fail - log but don't propagate
            logging.error(f"Failed to reset metrics: {e}")


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    
    state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED)
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    def __post_init__(self):
        """Validate circuit breaker configuration."""
        if not isinstance(self.failure_threshold, int) or self.failure_threshold <= 0:
            raise ConfigurationError("failure_threshold must be a positive integer")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        if not isinstance(self.half_open_max_calls, int) or self.half_open_max_calls <= 0:
            raise ConfigurationError("half_open_max_calls must be a positive integer")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        try:
            current_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if current_time - self.last_failure_time >= self.timeout_seconds:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return self.half_open_calls < self.half_open_max_calls
            
            return False
        except Exception:
            # Defensive: if circuit breaker fails, allow execution
            return True
    
    def record_success(self) -> None:
        """Record successful execution."""
        try:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
        except Exception as e:
            logging.error(f"Circuit breaker success recording failed: {e}")
    
    def record_failure(self) -> None:
        """Record failed execution."""
        try:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        except Exception as e:
            logging.error(f"Circuit breaker failure recording failed: {e}")


class DefensiveRateLimiter:
    """
    Defensive token bucket rate limiter with extensive error handling.
    
    This implementation assumes hostile inputs, system failures, and worst-case
    scenarios. Every operation includes comprehensive validation and recovery.
    """
    
    # Class-level constants for defensive bounds
    MIN_CAPACITY = 1
    MAX_CAPACITY = 1_000_000
    MIN_REFILL_RATE = 0.001  # tokens per second
    MAX_REFILL_RATE = 100_000.0  # tokens per second
    MIN_TIMEOUT = 0.001  # seconds
    MAX_TIMEOUT = 300.0  # seconds
    MAX_CONCURRENT_REQUESTS = 50_000
    CORRUPTION_CHECK_INTERVAL = 100  # operations
    
    def __init__(
        self,
        capacity: Union[int, float],
        refill_rate: Union[int, float],
        initial_tokens: Optional[Union[int, float]] = None,
        enable_metrics: bool = True,
        enable_circuit_breaker: bool = True,
        max_concurrent_requests: int = 10_000,
        corruption_detection: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize defensive rate limiter with extensive validation.
        
        Args:
            capacity: Maximum tokens in bucket (1 to 1,000,000)
            refill_rate: Tokens added per second (0.001 to 100,000)
            initial_tokens: Starting tokens (defaults to capacity)
            enable_metrics: Enable comprehensive metrics collection
            enable_circuit_breaker: Enable circuit breaker for fault tolerance
            max_concurrent_requests: Maximum concurrent requests allowed
            corruption_detection: Enable internal state corruption detection
            logger: Custom logger instance
            
        Raises:
            ConfigurationError: If parameters are invalid
            ResourceExhaustionError: If system resources insufficient
        """
        
        # Initialize logger first for error reporting
        self._logger = logger or self._create_defensive_logger()
        
        try:
            # Comprehensive input validation with defensive bounds
            self._validate_and_set_capacity(capacity)
            self._validate_and_set_refill_rate(refill_rate)
            self._validate_and_set_initial_tokens(initial_tokens)
            self._validate_and_set_concurrent_limit(max_concurrent_requests)
            
            # Initialize core state with defensive defaults
            self._tokens = float(self._initial_tokens)
            self._last_refill_time = time.time()
            self._operation_count = 0
            self._is_shutdown = False
            
            # Thread safety primitives with timeout protection
            self._lock = threading.RLock(timeout=30.0) if hasattr(threading.RLock, 'timeout') else threading.RLock()
            self._condition = threading.Condition(self._lock)
            
            # Concurrent request tracking
            self._concurrent_requests = 0
            self._active_requests: Set[int] = set()
            
            # Feature flags with defensive defaults
            self._enable_metrics = bool(enable_metrics)
            self._enable_circuit_breaker = bool(enable_circuit_breaker)
            self._corruption_detection = bool(corruption_detection)
            
            # Initialize subsystems with error handling
            self._initialize_metrics()
            self._initialize_circuit_breaker()
            self._initialize_corruption_detection()
            
            # Register cleanup handlers
            self._register_cleanup_handlers()
            
            # Validate initial state
            self._validate_internal_state()
            
            self._logger.info(f"DefensiveRateLimiter initialized: capacity={self._capacity}, "
                            f"refill_rate={self._refill_rate}, concurrent_limit={self._max_concurrent_requests}")
            
        except Exception as e:
            self._logger.error(f"Rate limiter initialization failed: {e}")
            self._logger.error(f"Traceback: {traceback.format_exc()}")
            if isinstance(e, (ConfigurationError, ResourceExhaustionError)):
                raise
            raise ConfigurationError(f"Initialization failed: {e}") from e
    
    def _create_defensive_logger(self) -> logging.Logger:
        """Create a defensive logger with comprehensive error handling."""
        try:
            logger = logging.getLogger(f"{__name__}.{id(self)}")
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.WARNING)
            return logger
        except Exception:
            # Fallback to basic logger if creation fails
            return logging.getLogger(__name__)
    
    def _validate_and_set_capacity(self, capacity: Union[int, float]) -> None:
        """Validate and set bucket capacity with defensive checks."""
        try:
            if capacity is None:
                raise ConfigurationError("Capacity cannot be None")
            
            if not isinstance(capacity, (int, float)):
                try:
                    capacity = float(capacity)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Capacity must be numeric, got {type(capacity)}") from e
            
            if not (self.MIN_CAPACITY <= capacity <= self.MAX_CAPACITY):
                raise ConfigurationError(
                    f"Capacity must be between {self.MIN_CAPACITY} and {self.MAX_CAPACITY}, got {capacity}"
                )
            
            if not isinstance(capacity, (int, float)) or capacity != capacity:  # NaN check
                raise ConfigurationError(f"Capacity must be a valid number, got {capacity}")
            
            self._capacity = float(capacity)
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Capacity validation failed: {e}") from e
    
    def _validate_and_set_refill_rate(self, refill_rate: Union[int, float]) -> None:
        """Validate and set refill rate with defensive checks."""
        try:
            if refill_rate is None:
                raise ConfigurationError("Refill rate cannot be None")
            
            if not isinstance(refill_rate, (int, float)):
                try:
                    refill_rate = float(refill_rate)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Refill rate must be numeric, got {type(refill_rate)}") from e
            
            if not (self.MIN_REFILL_RATE <= refill_rate <= self.MAX_REFILL_RATE):
                raise ConfigurationError(
                    f"Refill rate must be between {self.MIN_REFILL_RATE} and {self.MAX_REFILL_RATE}, got {refill_rate}"
                )
            
            if not isinstance(refill_rate, (int, float)) or refill_rate != refill_rate:  # NaN check
                raise ConfigurationError(f"Refill rate must be a valid number, got {refill_rate}")
            
            self._refill_rate = float(refill_rate)
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Refill rate validation failed: {e}") from e
    
    def _validate_and_set_initial_tokens(self, initial_tokens: Optional[Union[int, float]]) -> None:
        """Validate and set initial token count with defensive checks."""
        try:
            if initial_tokens is None:
                self._initial_tokens = self._capacity
                return
            
            if not isinstance(initial_tokens, (int, float)):
                try:
                    initial_tokens = float(initial_tokens)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Initial tokens must be numeric, got {type(initial_tokens)}") from e
            
            if not (0 <= initial_tokens <= self._capacity):
                raise ConfigurationError(
                    f"Initial tokens must be between 0 and capacity ({self._capacity}), got {initial_tokens}"
                )
            
            if not isinstance(initial_tokens, (int, float)) or initial_tokens != initial_tokens:  # NaN check
                raise ConfigurationError(f"Initial tokens must be a valid number, got {initial_tokens}")
            
            self._initial_tokens = float(initial_tokens)
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Initial tokens validation failed: {e}") from e
    
    def _validate_and_set_concurrent_limit(self, max_concurrent: int) -> None:
        """Validate and set concurrent request limit."""
        try:
            if not isinstance(max_concurrent, int):
                try:
                    max_concurrent = int(max_concurrent)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Max concurrent requests must be integer, got {type(max_concurrent)}") from e
            
            if not (1 <= max_concurrent <= self.MAX_CONCURRENT_REQUESTS):
                raise ConfigurationError(
                    f"Max concurrent requests must be between 1 and {self.MAX_CONCURRENT_REQUESTS}, got {max_concurrent}"
                )
            
            self._max_concurrent_requests = max_concurrent
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Concurrent limit validation failed: {e}") from e
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics collection with error handling."""
        try:
            if self._enable_metrics:
                self._metrics = RateLimiterMetrics()
            else:
                self._metrics = None
        except Exception as e:
            self._logger.error(f"Metrics initialization failed: {e}")
            self._metrics = None
            self._enable_metrics = False
    
    def _initialize_circuit_breaker(self) -> None:
        """Initialize circuit breaker with error handling."""
        try:
            if self._enable_circuit_breaker:
                self._circuit_breaker = CircuitBreaker()
            else:
                self._circuit_breaker = None
        except Exception as e:
            self._logger.error(f"Circuit breaker initialization failed: {e}")
            self._circuit_breaker = None
            self._enable_circuit_breaker = False
    
    def _initialize_corruption_detection(self) -> None:
        """Initialize corruption detection mechanisms."""
        try:
            if self._corruption_detection:
                self._state_checksum = self._calculate_state_checksum()
                self._corruption_checks = 0
            else:
                self._state_checksum = None
                self._corruption_checks = 0
        except Exception as e:
            self._logger.error(f"Corruption detection initialization failed: {e}")
            self._corruption_detection = False
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for graceful shutdown."""
        try:
            # Register weakref callback for garbage collection
            weakref.finalize(self, self._cleanup_resources)
            
            # Register signal handlers for graceful shutdown
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, self._signal_handler)
                
        except Exception as e:
            self._logger.error(f"Cleanup handler registration failed: {e}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        try:
            self._logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
        except Exception as e:
            self._logger.error(f"Signal handler failed: {e}")
    
    def _cleanup_resources(self) -> None:
        """Clean up resources during shutdown."""
        try:
            self._is_shutdown = True
            if hasattr(self, '_condition'):
                with self._condition:
                    self._condition.notify_all()
        except Exception as e:
            self._logger.error(f"Resource cleanup failed: {e}")
    
    def _calculate_state_checksum(self) -> int:
        """Calculate checksum of internal state for corruption detection."""
        try:
            state_data = (
                self._capacity,
                self._refill_rate,
                self._initial_tokens,
                self._max_concurrent_requests,
                int(self._tokens * 1000),  # Avoid floating point precision issues
                self._concurrent_requests
            )
            return hash(state_data)
        except Exception:
            return 0  # Default checksum on failure
    
    def _validate_internal_state(self) -> None:
        """Validate internal state consistency."""
        try:
            if self._tokens < 0:
                self._logger.error("State corruption: negative tokens detected")
                self._tokens = 0.0
                self._record_corruption_event()
            
            if self._tokens > self._capacity:
                self._logger.error("State corruption: tokens exceed capacity")
                self._tokens = float(self._capacity)
                self._record_corruption_event()
            
            if self._concurrent_requests < 0:
                self._logger.error("State corruption: negative concurrent requests")
                self._concurrent_requests = 0
                self._record_corruption_event()
            
            if self._concurrent_requests > self._max_concurrent_requests:
                self._logger.error("State corruption: concurrent requests exceed limit")
                self._concurrent_requests = self._max_concurrent_requests
                self._record_corruption_event()
            
            # Verify configuration integrity
            if not (self.MIN_CAPACITY <= self._capacity <= self.MAX_CAPACITY):
                raise CorruptionError(f"Capacity corruption detected: {self._capacity}")
            
            if not (self.MIN_REFILL_RATE <= self._refill_rate <= self.MAX_REFILL_RATE):
                raise CorruptionError(f"Refill rate corruption detected: {self._refill_rate}")
            
        except CorruptionError:
            raise
        except Exception as e:
            self._logger.error(f"State validation failed: {e}")
            raise CorruptionError(f"State validation error: {e}") from e
    
    def _record_corruption_event(self) -> None:
        """Record corruption event in metrics."""
        try:
            if self._metrics:
                self._metrics.corruption_events += 1
        except Exception:
            pass  # Don't fail on metric recording
    
    def _record_recovery_event(self) -> None:
        """Record recovery event in metrics."""
        try:
            if self._metrics:
                self._metrics.recovery_events += 1
        except Exception:
            pass  # Don't fail on metric recording
    
    def _periodic_corruption_check(self) -> None:
        """Perform periodic corruption checks."""
        try:
            if not self._corruption_detection:
                return
            
            self._corruption_checks += 1
            if self._corruption_checks % self.CORRUPTION_CHECK_INTERVAL == 0:
                current_checksum = self._calculate_state_checksum()
                if self._state_checksum and current_checksum != self._state_checksum:
                    self._logger.warning("Potential state corruption detected via checksum")
                    self._validate_internal_state()
                self._state_checksum = current_checksum
                
        except Exception as e:
            self._logger.error(f"Corruption check failed: {e}")
    
    def _refill_tokens(self, current_time: float) -> None:
        """
        Refill tokens based on elapsed time with defensive checks.
        
        Args:
            current_time: Current timestamp
            
        Raises:
            CorruptionError: If time corruption is detected
        """
        try:
            # Defensive time validation
            if not isinstance(current_time, (int, float)) or current_time != current_time:
                raise ValueError(f"Invalid current_time: {current_time}")
            
            if current_time < 0:
                raise ValueError(f"Negative timestamp: {current_time}")
            
            # Handle time going backwards (system clock adjustment)
            if current_time < self._last_refill_time:
                self._logger.warning(f"Time went backwards: {current_time} < {self._last_refill_time}")
                self._last_refill_time = current_time
                return
            
            # Calculate time elapsed with overflow protection
            time_elapsed = min(current_time - self._last_refill_time, 3600.0)  # Max 1 hour
            
            if time_elapsed < 0:
                self._logger.error(f"Negative time elapsed: {time_elapsed}")
                time_elapsed = 0
            
            # Calculate tokens to add with overflow protection
            tokens_to_add = min(time_elapsed * self._refill_rate, self._capacity * 2)
            
            if tokens_to_add < 0:
                self._logger.error(f"Negative tokens to add: {tokens_to_add}")
                tokens_to_add = 0
            
            # Update token count with capacity limit
            old_tokens = self._tokens
            self._tokens = min(self._tokens + tokens_to_add, self._capacity)
            
            # Defensive bounds checking
            if self._tokens < 0:
                self._logger.error(f"Token underflow detected: {self._tokens}")
                self._tokens = 0.0
                self._record_corruption_event()
            
            if self._tokens > self._capacity:
                self._logger.error(f"Token overflow detected: {self._tokens}")
                self._tokens = float(self._capacity)
                self._record_corruption_event()
            
            # Update refill time
            self._last_refill_time = current_time
            
            # Log significant refill events
            if tokens_to_add > 0 and self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(f"Refilled {tokens_to_add:.3f} tokens: {old_tokens:.3f} -> {self._tokens:.3f}")
            
        except Exception as e:
            self._logger.error(f"Token refill failed: {e}")
            # Defensive recovery: reset to safe state
            try:
                self._tokens = min(self._tokens, self._capacity) if self._tokens >= 0 else 0.0
                self._last_refill_time = current_time
                self._record_recovery_event()
            except Exception as recovery_error:
                self._logger.critical(f"Token refill recovery failed: {recovery_error}")
                raise CorruptionError(f"Unrecoverable refill error: {e}") from e
    
    def _update_metrics(self, allowed: bool, response_time_ms: float = 0.0) -> None:
        """Update metrics with defensive error handling."""
        try:
            if not self._metrics:
                return
            
            # Validate inputs
            if not isinstance(allowed, bool):
                allowed = bool(allowed)
            
            if not isinstance(response_time_ms, (int, float)) or response_time_ms < 0:
                response_time_ms = 0.0
            
            # Update counters
            self._metrics.total_requests += 1
            
            if allowed:
                self._metrics.allowed_requests += 1
            else:
                self._metrics.rejected_requests += 1
            
            # Update response time average
            if response_time_ms > 0:
                total_requests = self._metrics.total_requests
                if total_requests > 1:
                    # Rolling average calculation
                    old_avg = self._metrics.average_response_time_ms
                    self._metrics.average_response_time_ms = (
                        (old_avg * (total_requests - 1) + response_time_ms) / total_requests
                    )
                else:
                    self._metrics.average_response_time_ms = response_time_ms
            
            # Update concurrent request metrics
            self._metrics.current_concurrent_requests = self._concurrent_requests
            if self._concurrent_requests > self._metrics.peak_concurrent_requests:
                self._metrics.peak_concurrent_requests = self._concurrent_requests
            
            # Defensive bounds checking on metrics
            if self._metrics.total_requests < 0:
                self._logger.error("Metrics corruption: negative total requests")
                self._metrics.total_requests = 0
            
            if self._metrics.allowed_requests > self._metrics.total_requests:
                self._logger.error("Metrics corruption: allowed > total requests")
                self._metrics.allowed_requests = self._metrics.total_requests
            
        except Exception as e:
            self._logger.error(f"Metrics update failed: {e}")
            # Don't propagate metrics errors
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows execution."""
        try:
            if not self._circuit_breaker:
                return True
            
            can_execute = self._circuit_breaker.can_execute()
            
            if not can_execute and self._metrics:
                self._metrics.circuit_breaker_trips += 1
            
            return can_execute
            
        except Exception as e:
            self._logger.error(f"Circuit breaker check failed: {e}")
            # Defensive: allow execution if check fails
            return True
    
    def _record_circuit_breaker_result(self, success: bool) -> None:
        """Record circuit breaker execution result."""
        try:
            if not self._circuit_breaker:
                return
            
            if success:
                self._circuit_breaker.record_success()
            else:
                self._circuit_breaker.record_failure()
                
        except Exception as e:
            self._logger.error(f"Circuit breaker result recording failed: {e}")
    
    @contextmanager
    def _concurrent_request_tracking(self, request_id: int):
        """Context manager for tracking concurrent requests."""
        try:
            # Increment concurrent requests
            self._concurrent_requests += 1
            self._active_requests.add(request_id)
            
            # Check concurrent limit
            if self._concurrent_requests > self._max_concurrent_requests:
                self._concurrent_requests -= 1
                self._active_requests.discard(request_id)
                raise ResourceExhaustionError(
                    f"Concurrent request limit exceeded: {self._concurrent_requests} > {self._max_concurrent_requests}"
                )
            
            yield
            
        except Exception:
            # Ensure cleanup happens even on exception
            raise
        finally:
            # Defensive cleanup
            try:
                if request_id in self._active_requests:
                    self._active_requests.discard(request_id)
                    self._concurrent_requests = max(0, self._concurrent_requests - 1)
            except Exception as cleanup_error:
                self._logger.error(f"Request tracking cleanup failed: {cleanup_error}")
    
    def _acquire_lock_with_timeout(self, timeout: float) -> bool:
        """Acquire lock with timeout and defensive error handling."""
        try:
            # Validate timeout
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                timeout = 1.0  # Default timeout
            
            if timeout > self.MAX_TIMEOUT:
                timeout = self.MAX_TIMEOUT
            
            # Try to acquire lock with timeout
            if hasattr(self._lock, 'acquire'):
                return self._lock.acquire(timeout=timeout)
            else:
                # Fallback for locks without timeout support
                return self._lock.acquire(blocking=False)
                
        except Exception as e:
            self._logger.error(f"Lock acquisition failed: {e}")
            return False
    
    def allow_request(
        self,
        tokens_requested: Union[int, float] = 1,
        timeout_seconds: float = 1.0,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Check if request should be allowed based on token availability.
        
        This is the core synchronous method with comprehensive defensive programming.
        
        Args:
            tokens_requested: Number of tokens requested (default: 1)
            timeout_seconds: Maximum time to wait for tokens (default: 1.0)
            request_id: Optional request identifier for tracking
            
        Returns:
            bool: True if request is allowed, False if rate limited
            
        Raises:
            ConfigurationError: If parameters are invalid
            ResourceExhaustionError: If system resources are exhausted
            TimeoutError: If operation times out
            CorruptionError: If internal state corruption is detected
        """
        if self._is_shutdown:
            raise RateLimiterError("Rate limiter is shutdown")
        
        start_time = time.time()
        request_hash = hash((request_id, start_time, os.getpid(), threading.get_ident()))
        
        try:
            # Comprehensive input validation
            self._validate_allow_request_inputs(tokens_requested, timeout_seconds, request_id)
            
            # Circuit breaker check
            if not self._check_circuit_breaker():
                self._update_metrics(allowed=False)
                return False
            
            # Concurrent request tracking
            with self._concurrent_request_tracking(request_hash):
                
                # Acquire lock with timeout
                if not self._acquire_lock_with_timeout(timeout_seconds):
                    if self._metrics:
                        self._metrics.timeouts += 1
                    raise TimeoutError(f"Lock acquisition timeout: {timeout_seconds}s")
                
                try:
                    # Perform periodic corruption check
                    self._periodic_corruption_check()
                    
                    # Refill tokens
                    current_time = time.time()
                    self._refill_tokens(current_time)
                    
                    # Check token availability
                    if self._tokens >= tokens_requested:
                        # Allow request
                        self._tokens -= tokens_requested
                        self._validate_internal_state()
                        
                        # Record success
                        response_time = (time.time() - start_time) * 1000
                        self._update_metrics(allowed=True, response_time_ms=response_time)
                        self._record_circuit_breaker_result(success=True)
                        
                        return True
                    else:
                        # Rate limit request
                        self._update_metrics(allowed=False)
                        self._record_circuit_breaker_result(success=False)
                        
                        return False
                
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        self._logger.error(f"Lock release failed: {release_error}")
        
        except (ConfigurationError, ResourceExhaustionError, TimeoutError, CorruptionError):
            # Re-raise expected exceptions
            self._record_circuit_breaker_result(success=False)
            if self._metrics:
                self._metrics.errors_handled += 1
            raise
        
        except Exception as e:
            # Handle unexpected exceptions defensively
            self._logger.error(f"Unexpected error in allow_request: {e}")
            self._logger.error(f"Traceback: {traceback.format_exc()}")
            
            self._record_circuit_breaker_result(success=False)
            if self._metrics:
                self._metrics.errors_handled += 1
            
            # Attempt recovery
            try:
                self._validate_internal_state()
                self._record_recovery_event()
            except Exception as recovery_error:
                self._logger.critical(f"Recovery failed: {recovery_error}")
                raise CorruptionError(f"Unrecoverable error: {e}") from e
            
            # Re-raise as RateLimiterError
            raise RateLimiterError(f"Request processing failed: {e}") from e
    
    def _validate_allow_request_inputs(
        self,
        tokens_requested: Union[int, float],
        timeout_seconds: float,
        request_id: Optional[str]
    ) -> None:
        """Validate inputs to allow_request method."""
        try:
            # Validate tokens_requested
            if not isinstance(tokens_requested, (int, float)):
                try:
                    tokens_requested = float(tokens_requested)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"tokens_requested must be numeric, got {type(tokens_requested)}") from e
            
            if tokens_requested <= 0:
                raise ConfigurationError(f"tokens_requested must be positive, got {tokens_requested}")
            
            if tokens_requested > self._capacity:
                raise ConfigurationError(
                    f"tokens_requested ({tokens_requested}) exceeds capacity ({self._capacity})"
                )
            
            if tokens_requested != tokens_requested:  # NaN check
                raise ConfigurationError(f"tokens_requested must be a valid number, got {tokens_requested}")
            
            # Validate timeout_seconds
            if not isinstance(timeout_seconds, (int, float)):
                try:
                    timeout_seconds = float(timeout_seconds)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"timeout_seconds must be numeric, got {type(timeout_seconds)}") from e
            
            if not (self.MIN_TIMEOUT <= timeout_seconds <= self.MAX_TIMEOUT):
                raise ConfigurationError(
                    f"timeout_seconds must be between {self.MIN_TIMEOUT} and {self.MAX_TIMEOUT}, got {timeout_seconds}"
                )
            
            if timeout_seconds != timeout_seconds:  # NaN check
                raise ConfigurationError(f"timeout_seconds must be a valid number, got {timeout_seconds}")
            
            # Validate request_id
            if request_id is not None and not isinstance(request_id, str):
                try:
                    request_id = str(request_id)
                except Exception as e:
                    raise ConfigurationError(f"request_id must be string-convertible, got {type(request_id)}") from e
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Input validation failed: {e}") from e
    
    async def allow_request_async(
        self,
        tokens_requested: Union[int, float] = 1,
        timeout_seconds: float = 1.0,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Asynchronous version of allow_request with backpressure handling.
        
        Args:
            tokens_requested: Number of tokens requested (default: 1)
            timeout_seconds: Maximum time to wait for tokens (default: 1.0)
            request_id: Optional request identifier for tracking
            
        Returns:
            bool: True if request is allowed, False if rate limited
            
        Raises:
            ConfigurationError: If parameters are invalid
            ResourceExhaustionError: If system resources are exhausted
            TimeoutError: If operation times out
            CorruptionError: If internal state corruption is detected
        """
        if self._is_shutdown:
            raise RateLimiterError("Rate limiter is shutdown")
        
        start_time = time.time()
        
        try:
            # Input validation (same as synchronous version)
            self._validate_allow_request_inputs(tokens_requested, timeout_seconds, request_id)
            
            # Run synchronous method in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Use asyncio timeout for defensive timeout handling
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self.allow_request,
                        tokens_requested,
                        timeout_seconds,
                        request_id
                    ),
                    timeout=timeout_seconds + 1.0  # Add buffer for thread switching
                )
                return result
                
            except asyncio.TimeoutError:
                if self._metrics:
                    self._metrics.timeouts += 1
                raise TimeoutError(f"Async operation timeout: {timeout_seconds}s")
        
        except (ConfigurationError, ResourceExhaustionError, TimeoutError, CorruptionError, RateLimiterError):
            # Re-raise expected exceptions
            raise
        
        except Exception as e:
            # Handle unexpected exceptions
            self._logger.error(f"Unexpected error in allow_request_async: {e}")
            if self._metrics:
                self._metrics.errors_handled += 1
            raise RateLimiterError(f"Async request processing failed: {e}") from e
    
    def wait_for_tokens(
        self,
        tokens_requested: Union[int, float] = 1,
        timeout_seconds: float = 10.0,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Wait for tokens to become available with backpressure handling.
        
        Args:
            tokens_requested: Number of tokens requested
            timeout_seconds: Maximum time to wait
            request_id: Optional request identifier
            
        Returns:
            bool: True if tokens acquired, False if timeout
            
        Raises:
            ConfigurationError: If parameters are invalid
            ResourceExhaustionError: If system resources are exhausted
            CorruptionError: If internal state corruption is detected
        """
        if self._is_shutdown:
            raise RateLimiterError("Rate limiter is shutdown")
        
        start_time = time.time()
        end_time = start_time + timeout_seconds
        
        try:
            # Input validation
            self._validate_allow_request_inputs(tokens_requested, timeout_seconds, request_id)
            
            while time.time() < end_time and not self._is_shutdown:
                try:
                    if self.allow_request(tokens_requested, timeout_seconds=1.0, request_id=request_id):
                        return True
                    
                    # Calculate wait time based on token deficit
                    remaining_timeout = end_time - time.time()
                    if remaining_timeout <= 0:
                        break
                    
                    # Calculate optimal wait time
                    token_deficit = tokens_requested - self._tokens
                    wait_time = min(
                        token_deficit / self._refill_rate,
                        remaining_timeout,
                        1.0  # Max 1 second wait
                    )
                    
                    if wait_time > 0:
                        time.sleep(max(wait_time, 0.001))  # Minimum 1ms sleep
                
                except (ConfigurationError, ResourceExhaustionError, CorruptionError):
                    raise
                except Exception as e:
                    self._logger.error(f"Error during token wait: {e}")
                    time.sleep(0.1)  # Brief pause on error
            
            # Timeout reached
            if self._metrics:
                self._metrics.timeouts += 1
            return False
        
        except Exception as e:
            self._logger.error(f"wait_for_tokens failed: {e}")
            if self._metrics:
                self._metrics.errors_handled += 1
            raise
    
    async def wait_for_tokens_async(
        self,
        tokens_requested: Union[int, float] = 1,
        timeout_seconds: float = 10.0,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Asynchronously wait for tokens to become available.
        
        Args:
            tokens_requested: Number of tokens requested
            timeout_seconds: Maximum time to wait
            request_id: Optional request identifier
            
        Returns:
            bool: True if tokens acquired, False if timeout
        """
        if self._is_shutdown:
            raise RateLimiterError("Rate limiter is shutdown")
        
        start_time = time.time()
        end_time = start_time + timeout_seconds
        
        try:
            # Input validation
            self._validate_allow_request_inputs(tokens_requested, timeout_seconds, request_id)
            
            while time.time() < end_time and not self._is_shutdown:
                try:
                    if await self.allow_request_async(tokens_requested, timeout_seconds=1.0, request_id=request_id):
                        return True
                    
                    # Calculate wait time
                    remaining_timeout = end_time - time.time()
                    if remaining_timeout <= 0:
                        break
                    
                    token_deficit = max(0, tokens_requested - self._tokens)
                    wait_time = min(
                        token_deficit / self._refill_rate,
                        remaining_timeout,
                        1.0
                    )
                    
                    if wait_time > 0:
                        await asyncio.sleep(max(wait_time, 0.001))
                
                except (ConfigurationError, ResourceExhaustionError, CorruptionError):
                    raise
                except Exception as e:
                    self._logger.error(f"Error during async token wait: {e}")
                    await asyncio.sleep(0.1)
            
            return False
        
        except Exception as e:
            self._logger.error(f"wait_for_tokens_async failed: {e}")
            if self._metrics:
                self._metrics.errors_handled += 1
            raise
    
    def get_metrics(self) -> Optional[RateLimiterMetrics]:
        """
        Get comprehensive metrics with defensive copying.
        
        Returns:
            RateLimiterMetrics: Copy of current metrics or None if disabled
        """
        try:
            if not self._metrics:
                return None
            
            # Create defensive copy
            import copy
            return copy.deepcopy(self._metrics)
        
        except Exception as e:
            self._logger.error(f"Metrics retrieval failed: {e}")
            return None
    
    def reset_metrics(self) -> None:
        """Reset metrics with error handling."""
        try:
            if self._metrics:
                self._metrics.reset()
                self._logger.info("Metrics reset successfully")
        except Exception as e:
            self._logger.error(f"Metrics reset failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Dict containing current status and configuration
        """
        try:
            with self._lock:
                current_time = time.time()
                self._refill_tokens(current_time)
                
                status = {
                    'capacity': self._capacity,
                    'refill_rate': self._refill_rate,
                    'current_tokens': round(self._tokens, 3),
                    'tokens_percentage': round((self._tokens / self._capacity) * 100, 1),
                    'concurrent_requests': self._concurrent_requests,
                    'max_concurrent_requests': self._max_concurrent_requests,
                    'is_shutdown': self._is_shutdown,
                    'metrics_enabled': self._enable_metrics,
                    'circuit_breaker_enabled': self._enable_circuit_breaker,
                    'corruption_detection_enabled': self._corruption_detection,
                    'last_refill_time': self._last_refill_time,
                    'operation_count': self._operation_count,
                }
                
                # Add circuit breaker status
                if self._circuit_breaker:
                    status['circuit_breaker_state'] = self._circuit_breaker.state.value
                    status['circuit_breaker_failures'] = self._circuit_breaker.failure_count
                
                # Add metrics summary
                if self._metrics:
                    status['total_requests'] = self._metrics.total_requests
                    status['success_rate'] = (
                        (self._metrics.allowed_requests / self._metrics.total_requests * 100)
                        if self._metrics.total_requests > 0 else 0
                    )
                    status['error_count'] = self._metrics.errors_handled
                    status['corruption_events'] = self._metrics.corruption_events
                    status['recovery_events'] = self._metrics.recovery_events
                
                return status
        
        except Exception as e:
            self._logger.error(f"Status retrieval failed: {e}")
            return {
                'error': str(e),
                'is_shutdown': getattr(self, '_is_shutdown', True),
                'capacity': getattr(self, '_capacity', 0),
                'refill_rate': getattr(self, '_refill_rate', 0),
            }
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the rate limiter.
        
        This method ensures all resources are properly cleaned up and
        prevents new requests from being processed.
        """
        try:
            self._logger.info("Initiating graceful shutdown")
            self._is_shutdown = True
            
            # Wake up any waiting threads
            try:
                with self._condition:
                    self._condition.notify_all()
            except Exception as e:
                self._logger.error(f"Failed to notify waiting threads: {e}")
            
            # Wait for active requests to complete (with timeout)
            shutdown_timeout = 30.0  # 30 seconds
            shutdown_start = time.time()
            
            while (self._concurrent_requests > 0 and 
                   time.time() - shutdown_start < shutdown_timeout):
                time.sleep(0.1)
            
            if self._concurrent_requests > 0:
                self._logger.warning(f"Shutdown timeout: {self._concurrent_requests} requests still active")
            
            # Clean up resources
            self._cleanup_resources()
            
            self._logger.info("Shutdown completed")
        
        except Exception as e:
            self._logger.error(f"Shutdown failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        try:
            return (f"DefensiveRateLimiter(capacity={self._capacity}, "
                   f"refill_rate={self._refill_rate}, "
                   f"tokens={self._tokens:.2f}, "
                   f"concurrent={self._concurrent_requests})")
        except Exception:
            return "DefensiveRateLimiter(corrupted_state)"