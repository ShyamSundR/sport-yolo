import logging
import os
import sys
from typing import Optional
import structlog
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def setup_logger(service_name: str = "sports-video-api") -> structlog.stdlib.BoundLogger:
    """
    Setup structured logging with optional CloudWatch integration
    
    Args:
        service_name: Name of the service for logging
        
    Returns:
        Configured logger instance
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Try to setup CloudWatch logging if AWS credentials are available
    try:
        setup_cloudwatch_logging(service_name)
    except (NoCredentialsError, ClientError) as e:
        print(f"CloudWatch logging not available: {e}")
        print("Falling back to local logging only")
    
    # Get logger
    logger = structlog.get_logger(service_name)
    logger.info("Logger initialized", service=service_name)
    
    return logger

def setup_cloudwatch_logging(service_name: str):
    """
    Setup CloudWatch logging handler
    
    Args:
        service_name: Service name for log group
    """
    
    # Only setup CloudWatch in production
    if os.getenv("AWS_REGION") and os.getenv("ENVIRONMENT", "development") == "production":
        try:
            import watchtower
            
            # Create CloudWatch logs client
            cloudwatch_client = boto3.client(
                'logs',
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            
            # Create CloudWatch handler
            cw_handler = watchtower.CloudWatchLogsHandler(
                boto3_client=cloudwatch_client,
                log_group=f"/aws/ec2/{service_name}",
                stream_name=f"{service_name}-{os.getenv('HOSTNAME', 'unknown')}",
                create_log_group=True
            )
            
            # Add CloudWatch handler to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(cw_handler)
            
            print(f"CloudWatch logging enabled for {service_name}")
            
        except ImportError:
            print("watchtower not available, skipping CloudWatch logging")
        except Exception as e:
            print(f"Failed to setup CloudWatch logging: {e}")

class MetricsLogger:
    """Custom metrics logger for performance tracking"""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self.logger = logger
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'total_detections': 0
        }
    
    def log_request_start(self, endpoint: str, file_size: Optional[int] = None):
        """Log request start"""
        self.logger.info(
            "Request started",
            endpoint=endpoint,
            file_size=file_size
        )
        self.metrics['total_requests'] += 1
    
    def log_request_success(self, endpoint: str, processing_time: float, 
                          detection_count: int):
        """Log successful request"""
        self.logger.info(
            "Request completed successfully",
            endpoint=endpoint,
            processing_time=processing_time,
            detection_count=detection_count,
            fps=1.0/processing_time if processing_time > 0 else 0
        )
        
        self.metrics['successful_requests'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['total_detections'] += detection_count
    
    def log_request_failure(self, endpoint: str, error: str, 
                          processing_time: Optional[float] = None):
        """Log failed request"""
        self.logger.error(
            "Request failed",
            endpoint=endpoint,
            error=error,
            processing_time=processing_time
        )
        self.metrics['failed_requests'] += 1
    
    def log_system_metrics(self):
        """Log overall system metrics"""
        avg_processing_time = (
            self.metrics['total_processing_time'] / self.metrics['successful_requests']
            if self.metrics['successful_requests'] > 0 else 0
        )
        
        success_rate = (
            self.metrics['successful_requests'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0 else 0
        )
        
        self.logger.info(
            "System metrics",
            total_requests=self.metrics['total_requests'],
            success_rate=success_rate,
            avg_processing_time=avg_processing_time,
            total_detections=self.metrics['total_detections'],
            avg_detections_per_request=(
                self.metrics['total_detections'] / self.metrics['successful_requests']
                if self.metrics['successful_requests'] > 0 else 0
            )
        )
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.copy()
