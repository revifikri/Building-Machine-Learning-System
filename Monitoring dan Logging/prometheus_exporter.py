#!/usr/bin/env python3
"""
Prometheus Exporter untuk MLflow Model Monitoring
Disesuaikan dengan structure dataset real: data forecasting_processed.csv
Compatible dengan modelling.py dan mlflow_serve.py
"""

import time
import threading
import requests
import psutil
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    start_http_server, CollectorRegistry, REGISTRY
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowSalesModelExporter:
    """Prometheus exporter untuk monitoring MLflow Sales Forecasting model"""
    
    def __init__(self, 
                 model_endpoint: str = "http://127.0.0.1:1234",
                 mlflow_tracking_uri: str = "http://localhost:5000",
                 port: int = 8000):
        self.model_endpoint = model_endpoint
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.port = port
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Monitoring state
        self.prediction_errors = 0
        self.total_predictions = 0
        self.start_time = time.time()
        
        # Sample features yang sesuai dengan data forecasting_processed.csv (37 features)
        self.sample_features = self._create_realistic_sample_features()
        
        logger.info(f"MLflow Sales Model Exporter initialized")
        logger.info(f"Model endpoint: {model_endpoint}")
        logger.info(f"MLflow tracking: {mlflow_tracking_uri}")
        logger.info(f"Prometheus port: {port}")
        logger.info(f"Sample features count: {len(self.sample_features)}")
    
    def _create_realistic_sample_features(self) -> List[float]:
        """
        Create sample features based on real data structure from dataset
        Compatible dengan create_time_series_features() dari modelling.py
        """
        import numpy as np
        
        # Simulate realistic sales transaction data
        quantity = 5.0
        unit_price = 15.50
        year = 2024.0
        month = 6.0
        day = 15.0
        day_of_week = 3.0  # Wednesday
        hour = 14.0  # 2 PM
        is_weekend = 0.0
        
        # Encoded features (dari dataset original)
        invoice_no_encoded = 12345.0
        stock_code_encoded = 6789.0
        customer_id_encoded = 1001.0
        country_encoded = 5.0
        
        # Time features yang dibuat oleh create_time_series_features()
        day_of_month = 15.0
        week_of_year = 24.0
        quarter = 2.0
        days_from_start = 165.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_year_sin = np.sin(2 * np.pi * 166 / 365)  # June 15th ≈ day 166
        day_of_year_cos = np.cos(2 * np.pi * 166 / 365)
        
        # Lag features (historical sales)
        base_sales = quantity * unit_price  # 77.5
        total_sales_lag_1 = base_sales * 0.95   # 73.625
        total_sales_lag_2 = base_sales * 1.05   # 81.375  
        total_sales_lag_3 = base_sales * 0.98   # 75.95
        total_sales_lag_7 = base_sales * 1.02   # 79.05
        total_sales_lag_14 = base_sales * 0.92  # 71.3
        
        # Rolling statistics features
        # Window 3
        rolling_mean_3 = base_sales * 0.99    # 76.725
        rolling_std_3 = base_sales * 0.15     # 11.625
        rolling_min_3 = base_sales * 0.85     # 65.875
        rolling_max_3 = base_sales * 1.15     # 89.125
        
        # Window 7  
        rolling_mean_7 = base_sales * 1.01    # 78.275
        rolling_std_7 = base_sales * 0.18     # 13.95
        rolling_min_7 = base_sales * 0.80     # 62.0
        rolling_max_7 = base_sales * 1.20     # 93.0
        
        # Window 14
        rolling_mean_14 = base_sales * 0.97   # 75.175
        rolling_std_14 = base_sales * 0.22    # 17.05
        rolling_min_14 = base_sales * 0.75    # 58.125
        rolling_max_14 = base_sales * 1.25    # 96.875
        
        # Construct feature vector in exact order expected by model
        # Based on prepare_features() function in modelling.py
        features = [
            # Basic features (8)
            quantity, unit_price, year, month, day, day_of_week, hour, is_weekend,
            
            # Encoded features (4) 
            invoice_no_encoded, stock_code_encoded, customer_id_encoded, country_encoded,
            
            # Time features (8)
            day_of_month, week_of_year, quarter, days_from_start,
            float(hour_sin), float(hour_cos), float(day_of_year_sin), float(day_of_year_cos),
            
            # Lag features (5)
            total_sales_lag_1, total_sales_lag_2, total_sales_lag_3, total_sales_lag_7, total_sales_lag_14,
            
            # Rolling features (12)
            rolling_mean_3, rolling_std_3, rolling_min_3, rolling_max_3,      # window 3
            rolling_mean_7, rolling_std_7, rolling_min_7, rolling_max_7,      # window 7
            rolling_mean_14, rolling_std_14, rolling_min_14, rolling_max_14   # window 14
        ]
        
        # Verify feature count
        if len(features) != 37:
            logger.error(f"Feature count mismatch! Expected 37, got {len(features)}")
            logger.error("Check feature construction logic")
        
        return features
    
    def _init_metrics(self):
        """Initialize 3+ core Prometheus metrics sesuai requirement"""
        
        # === METRIK 1: PREDICTION REQUESTS (Counter) ===
        self.prediction_requests_total = Counter(
            'mlflow_prediction_requests_total',
            'Total number of prediction requests to Sales Forecasting model',
            ['status', 'model_name']
        )
        
        self.prediction_duration_seconds = Histogram(
            'mlflow_prediction_duration_seconds',
            'Time spent processing Sales Forecasting predictions',
            ['model_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # === METRIK 2: MODEL HEALTH & PERFORMANCE (Gauge) ===
        self.model_server_up = Gauge(
            'mlflow_model_server_up',
            'Whether Sales Forecasting model server is up (1) or down (0)'
        )
        
        self.model_prediction_value = Gauge(
            'mlflow_model_latest_prediction_value',
            'Latest prediction value from Sales Forecasting model',
            ['model_name']
        )
        
        self.model_response_time = Gauge(
            'mlflow_model_response_time_seconds',
            'Latest response time for model predictions'
        )
        
        # === METRIK 3: SYSTEM RESOURCES (Gauge) ===
        self.system_cpu_usage = Gauge(
            'mlflow_system_cpu_usage_percent',
            'System CPU usage percentage where MLflow model is running'
        )
        
        self.system_memory_usage = Gauge(
            'mlflow_system_memory_usage_percent',
            'System memory usage percentage where MLflow model is running'
        )
        
        self.model_uptime_seconds = Gauge(
            'mlflow_model_uptime_seconds',
            'Model server uptime in seconds since monitoring started'
        )
        
        # === METRIK TAMBAHAN ===
        self.mlflow_tracking_server_up = Gauge(
            'mlflow_tracking_server_up',
            'Whether MLflow tracking server is up (1) or down (0)'
        )
        
        self.prediction_error_rate = Gauge(
            'mlflow_prediction_error_rate',
            'Prediction error rate (0-1)'
        )
        
        # Model info untuk metadata
        self.model_info = Info(
            'mlflow_model_info',
            'Information about the deployed Sales Forecasting model'
        )
        
        logger.info("✅ Prometheus metrics initialized dengan 3+ metrik utama")
    
    def check_model_server_health(self) -> bool:
        """Check if model server (port 1234) is healthy"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.model_endpoint}/ping", timeout=10)
            response_time = time.time() - start_time
            
            is_healthy = response.status_code == 200
            self.model_server_up.set(1 if is_healthy else 0)
            
            if is_healthy:
                self.model_response_time.set(response_time)
                logger.debug(f"Model server health: OK ({response_time:.3f}s)")
            else:
                logger.warning(f"Model server health: FAILED (status {response.status_code})")
                
            return is_healthy
        except Exception as e:
            logger.warning(f"Model server health check failed: {e}")
            self.model_server_up.set(0)
            return False
    
    def check_mlflow_tracking_health(self) -> bool:
        """Check if MLflow tracking server (port 5000) is healthy"""
        try:
            response = requests.get(f"{self.mlflow_tracking_uri}", timeout=10)
            is_healthy = response.status_code == 200
            self.mlflow_tracking_server_up.set(1 if is_healthy else 0)
            
            if is_healthy:
                logger.debug("MLflow tracking server: OK")
            else:
                logger.warning(f"MLflow tracking server: FAILED (status {response.status_code})")
                
            return is_healthy
        except Exception as e:
            logger.warning(f"MLflow tracking health check failed: {e}")
            self.mlflow_tracking_server_up.set(0)
            return False
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def test_model_prediction(self):
        """Test model dengan prediction real untuk monitoring"""
        try:
            # Verify feature count (critical untuk compatibility)
            if len(self.sample_features) != 37:
                logger.error(f"Feature count mismatch: expected 37, got {len(self.sample_features)}")
                logger.error("Cannot test prediction with wrong feature count")
                return False
            
            # Test prediction
            start_time = time.time()
            response = requests.post(
                f"{self.model_endpoint}/invocations",
                json={"instances": [self.sample_features]},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                # Success
                self.prediction_requests_total.labels(
                    status="success", 
                    model_name="sales_forecasting"
                ).inc()
                
                self.prediction_duration_seconds.labels(
                    model_name="sales_forecasting"
                ).observe(duration)
                
                # Parse response and update prediction value
                result = response.json()
                
                # Handle different MLflow response formats
                prediction_value = None
                if isinstance(result, list) and len(result) > 0:
                    prediction_value = float(result[0])
                elif isinstance(result, dict):
                    if "predictions" in result:
                        predictions = result["predictions"]
                        if isinstance(predictions, list) and len(predictions) > 0:
                            prediction_value = float(predictions[0])
                        else:
                            prediction_value = float(predictions)
                    else:
                        # Sometimes MLflow returns direct value
                        prediction_value = float(list(result.values())[0])
                else:
                    prediction_value = float(result)
                
                if prediction_value is not None:
                    self.model_prediction_value.labels(
                        model_name="sales_forecasting"
                    ).set(prediction_value)
                
                self.total_predictions += 1
                
                # Update error rate
                error_rate = self.prediction_errors / self.total_predictions if self.total_predictions > 0 else 0
                self.prediction_error_rate.set(error_rate)
                
                logger.info(f"✅ Test prediction: ${prediction_value:.2f} in {duration:.3f}s")
                return True
            else:
                # Error
                self.prediction_requests_total.labels(
                    status="error", 
                    model_name="sales_forecasting"
                ).inc()
                
                self.prediction_errors += 1
                self.total_predictions += 1
                
                error_rate = self.prediction_errors / self.total_predictions
                self.prediction_error_rate.set(error_rate)
                
                logger.error(f"❌ Test prediction failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            # Exception
            self.prediction_requests_total.labels(
                status="error", 
                model_name="sales_forecasting"
            ).inc()
            
            self.prediction_errors += 1
            self.total_predictions += 1
            
            error_rate = self.prediction_errors / self.total_predictions
            self.prediction_error_rate.set(error_rate)
            
            logger.error(f"❌ Test prediction exception: {e}")
            return False
    
    def start_monitoring(self):
        """Start monitoring loop"""
        start_time = time.time()
        
        def monitoring_loop():
            while True:
                try:
                    logger.info("🔄 Running monitoring cycle...")
                    
                    # 1. Health checks
                    model_healthy = self.check_model_server_health()
                    mlflow_healthy = self.check_mlflow_tracking_health()
                    
                    # 2. System metrics
                    self.update_system_metrics()
                    
                    # 3. Test prediction (hanya jika model server healthy)
                    if model_healthy:
                        self.test_model_prediction()
                    else:
                        logger.warning("⚠️ Skipping prediction test - model server not healthy")
                    
                    # 4. Update uptime
                    uptime = time.time() - start_time
                    self.model_uptime_seconds.set(uptime)
                    
                    # Log summary
                    success_rate = ((self.total_predictions - self.prediction_errors) / self.total_predictions * 100) if self.total_predictions > 0 else 0
                    logger.info(f"📊 Monitoring summary:")
                    logger.info(f"   Model healthy: {model_healthy}")
                    logger.info(f"   MLflow healthy: {mlflow_healthy}")
                    logger.info(f"   Total predictions: {self.total_predictions}")
                    logger.info(f"   Success rate: {success_rate:.1f}%")
                    logger.info(f"   Uptime: {uptime:.0f}s")
                    
                    # Sleep for monitoring interval
                    time.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(10)
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("✅ Monitoring loop started")
    
    def start_server(self):
        """Start Prometheus HTTP server"""
        try:
            # Set initial model info
            self.model_info.info({
                'model_name': 'sales_forecasting_model',
                'model_version': '1.0.0',
                'framework': 'scikit-learn',
                'features': '37',
                'dataset': 'data_forecasting_processed.csv',
                'deployment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_endpoint': self.model_endpoint,
                'mlflow_tracking': self.mlflow_tracking_uri,
                'experiment_name': 'Sales_Monitoring_Experiment'
            })
            
            # Start monitoring
            self.start_monitoring()
            
            # Start HTTP server
            start_http_server(self.port)
            
            logger.info(f"🚀 Prometheus exporter started on port {self.port}")
            logger.info(f"📊 Metrics available at: http://localhost:{self.port}/metrics")
            logger.info(f"🎯 Monitoring MLflow model at: {self.model_endpoint}")
            logger.info(f"📋 Compatible with dataset: data forecasting_processed.csv")
            
            # Keep server running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⏹️  Prometheus exporter stopped")
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow Prometheus Exporter untuk Sales Forecasting Model')
    parser.add_argument('--model-endpoint', default='http://127.0.0.1:1234',
                       help='MLflow model endpoint (default: http://127.0.0.1:1234)')
    parser.add_argument('--mlflow-uri', default='http://localhost:5000',
                       help='MLflow tracking URI (default: http://localhost:5000)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Prometheus exporter port (default: 8000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start exporter
    exporter = MLflowSalesModelExporter(
        model_endpoint=args.model_endpoint,
        mlflow_tracking_uri=args.mlflow_uri,
        port=args.port
    )
    
    print(f"""
🔥 MLflow Prometheus Exporter untuk Sales Forecasting
===================================================
Model Endpoint: {args.model_endpoint}
MLflow URI: {args.mlflow_uri}
Prometheus Port: {args.port}
Metrics URL: http://localhost:{args.port}/metrics

📊 Metrik yang dimonitor:
1. Prediction Requests (Counter): mlflow_prediction_requests_total
2. Model Health & Performance (Gauge): mlflow_model_server_up, mlflow_model_latest_prediction_value
3. System Resources (Gauge): mlflow_system_cpu_usage_percent, mlflow_system_memory_usage_percent

📋 Dataset Compatibility: data forecasting_processed.csv (37 features)
🧪 Compatible dengan Sales_Monitoring_Experiment dari modelling.py

Press Ctrl+C to stop
    """)
    
    exporter.start_server()

if __name__ == "__main__":
    main()