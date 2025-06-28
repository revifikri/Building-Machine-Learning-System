#!/usr/bin/env python3
"""
MLflow Model Inference Service untuk Load Testing & Monitoring
Disesuaikan dengan dataset structure: data forecasting_processed.csv
Compatible dengan modelling.py feature engineering
"""

import time
import requests
import json
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesForecastingInferenceService:
    """Service untuk inference Sales Forecasting model dengan monitoring"""
    
    def __init__(self,
                 model_endpoint: str = "http://127.0.0.1:1234",
                 prometheus_endpoint: str = "http://localhost:8000",
                 mlflow_tracking_uri: str = "http://localhost:5000"):
        
        self.model_endpoint = model_endpoint
        self.prometheus_endpoint = prometheus_endpoint
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Inference statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        logger.info(f"Sales Forecasting Inference Service initialized")
        logger.info(f"Model endpoint: {model_endpoint}")
        logger.info(f"Compatible dengan: data forecasting_processed.csv structure")
    
    def create_realistic_sales_features(self, num_samples: int = 1) -> List[List[float]]:
        """
        Create realistic sales forecasting features (37 features) 
        matching data forecasting_processed.csv structure dan modelling.py processing
        """
        samples = []
        
        for _ in range(num_samples):
            # Generate realistic transaction data based on retail dataset
            quantity = random.randint(1, 50)  # Realistic quantity range
            unit_price = random.uniform(0.85, 85.0)  # Realistic price range dari retail
            
            # Date/time features - current realistic values
            year = 2024
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            day_of_week = random.randint(0, 6)
            hour = random.randint(8, 20)  # Business hours 8AM-8PM
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Encoded features (from original dataset structure)
            # Based on range analysis of typical retail data
            invoice_no_encoded = random.randint(536365, 581587)  # Typical invoice range
            stock_code_encoded = random.randint(10002, 23843)   # Stock code range
            customer_id_encoded = random.randint(12346, 18287)  # Customer ID range  
            country_encoded = random.randint(0, 36)             # Country encoding range
            
            # Time features yang akan dibuat oleh create_time_series_features()
            day_of_month = day
            week_of_year = random.randint(1, 52)
            quarter = (month - 1) // 3 + 1
            days_from_start = random.randint(0, 731)  # ~2 years range
            
            # Cyclical features (matching modelling.py)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_of_year = random.randint(1, 365)
            day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
            
            # Create base feature vector (exact order from prepare_features in modelling.py)
            sample = [
                # Basic features (8) - dari kolom dataset original
                float(quantity),           # 0: Quantity
                float(unit_price),         # 1: UnitPrice
                float(year),               # 2: Year
                float(month),              # 3: Month
                float(day),                # 4: Day
                float(day_of_week),        # 5: DayOfWeek
                float(hour),               # 6: Hour
                float(is_weekend),         # 7: IsWeekend
                
                # Encoded features (4) - dari dataset original
                float(invoice_no_encoded), # 8: InvoiceNo_encoded
                float(stock_code_encoded), # 9: StockCode_encoded
                float(customer_id_encoded),# 10: CustomerID_encoded
                float(country_encoded),    # 11: Country_encoded
                
                # Time features (8) - dibuat oleh create_time_series_features()
                float(day_of_month),       # 12: DayOfMonth
                float(week_of_year),       # 13: WeekOfYear
                float(quarter),            # 14: Quarter
                float(days_from_start),    # 15: DaysFromStart
                float(hour_sin),           # 16: HourSin
                float(hour_cos),           # 17: HourCos
                float(day_of_year_sin),    # 18: DayOfYearSin
                float(day_of_year_cos),    # 19: DayOfYearCos
            ]
            
            # Add lag features (TotalSales_lag_1, 2, 3, 7, 14) - 5 features
            base_sales = quantity * unit_price
            lag_multipliers = [0.95, 1.05, 0.98, 1.02, 0.92]  # Realistic variations
            for multiplier in lag_multipliers:
                lag_sales = base_sales * multiplier * random.uniform(0.9, 1.1)
                sample.append(float(lag_sales))
            
            # Add rolling features (12 features: mean, std, min, max for windows 3, 7, 14)
            rolling_configs = [
                # window 3
                (0.99, 0.15, 0.85, 1.15),
                # window 7  
                (1.01, 0.18, 0.80, 1.20),
                # window 14
                (0.97, 0.22, 0.75, 1.25)
            ]
            
            for mean_mult, std_mult, min_mult, max_mult in rolling_configs:
                rolling_mean = base_sales * mean_mult * random.uniform(0.95, 1.05)
                rolling_std = base_sales * std_mult * random.uniform(0.8, 1.2)
                rolling_min = base_sales * min_mult * random.uniform(0.9, 1.0)
                rolling_max = base_sales * max_mult * random.uniform(1.0, 1.1)
                
                sample.extend([
                    float(rolling_mean),
                    float(rolling_std),
                    float(rolling_min),
                    float(rolling_max)
                ])
            
            samples.append(sample)
        
        # Verify feature count (harus 37 sesuai modelling.py)
        if samples and len(samples[0]) != 37:
            logger.error(f"❌ Feature count mismatch: expected 37, got {len(samples[0])}")
            logger.error("Feature structure doesn't match modelling.py")
            # Print detailed breakdown for debugging
            logger.error("Expected feature breakdown:")
            logger.error("  Basic: 8, Encoded: 4, Time: 8, Lag: 5, Rolling: 12 = 37 total")
            logger.error(f"  Actual feature count: {len(samples[0])}")
        else:
            logger.debug(f"✅ Generated {len(samples)} samples with 37 features each")
        
        return samples
    
    def check_model_health(self) -> bool:
        """Check if model server is healthy"""
        try:
            response = requests.get(f"{self.model_endpoint}/ping", timeout=10)
            is_healthy = response.status_code == 200
            if is_healthy:
                logger.debug("Model server health: OK")
            else:
                logger.warning(f"Model server health: FAILED (status {response.status_code})")
            return is_healthy
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False
    
    def make_prediction(self, features: List[List[float]]) -> Dict[str, Any]:
        """Make prediction using MLflow Sales Forecasting model"""
        start_time = time.time()
        
        try:
            # Verify feature structure (critical untuk compatibility)
            if features and len(features[0]) != 37:
                error_msg = f"Invalid feature count: expected 37, got {len(features[0])}"
                logger.error(error_msg)
                logger.error("Feature mismatch dengan data forecasting_processed.csv structure")
                return {
                    "status": "error",
                    "error": error_msg,
                    "duration": 0,
                    "batch_size": len(features),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare request data
            payload = {
                "instances": features
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Make prediction request
            logger.info(f"Making Sales Forecasting prediction for {len(features)} samples...")
            
            response = requests.post(
                f"{self.model_endpoint}/invocations",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                predictions = response.json()
                
                result = {
                    "status": "success",
                    "predictions": predictions,
                    "duration": duration,
                    "batch_size": len(features),
                    "timestamp": datetime.now().isoformat(),
                    "model_endpoint": self.model_endpoint,
                    "model_type": "sales_forecasting",
                    "dataset_compatible": "data_forecasting_processed.csv"
                }
                
                self.total_requests += 1
                self.successful_requests += 1
                
                # Extract prediction values for logging
                pred_values = []
                if isinstance(predictions, list):
                    pred_values = predictions
                elif isinstance(predictions, dict) and "predictions" in predictions:
                    pred_values = predictions["predictions"]
                    if not isinstance(pred_values, list):
                        pred_values = [pred_values]
                else:
                    pred_values = [predictions]
                
                if pred_values:
                    avg_prediction = np.mean(pred_values)
                    max_prediction = np.max(pred_values)
                    min_prediction = np.min(pred_values)
                    logger.info(f"✅ Prediction successful: {len(pred_values)} results")
                    logger.info(f"   Sales range: ${min_prediction:.2f} - ${max_prediction:.2f}, avg: ${avg_prediction:.2f}")
                    logger.info(f"   Duration: {duration:.3f}s")
                else:
                    logger.info(f"✅ Prediction successful in {duration:.3f}s")
                
                return result
            
            else:
                error_result = {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration,
                    "batch_size": len(features),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.total_requests += 1
                self.failed_requests += 1
                
                logger.error(f"❌ Prediction failed: {error_result['error']}")
                
                return error_result
        
        except Exception as e:
            duration = time.time() - start_time
            
            error_result = {
                "status": "error",
                "error": str(e),
                "duration": duration,
                "batch_size": len(features),
                "timestamp": datetime.now().isoformat()
            }
            
            self.total_requests += 1
            self.failed_requests += 1
            
            logger.error(f"❌ Prediction exception: {e}")
            
            return error_result
    
    def run_continuous_inference(self, 
                               interval: int = 60,
                               batch_size: int = 1,
                               max_iterations: Optional[int] = None):
        """Run continuous inference untuk load testing"""
        
        logger.info(f"🚀 Starting continuous Sales Forecasting inference...")
        logger.info(f"   Interval: {interval} seconds")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Max iterations: {max_iterations or 'unlimited'}")
        logger.info(f"   Dataset compatible: data forecasting_processed.csv")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Check if we should stop
                if max_iterations and iteration > max_iterations:
                    logger.info(f"🏁 Reached max iterations ({max_iterations})")
                    break
                
                logger.info(f"\n--- Sales Forecasting Iteration {iteration} ---")
                
                # Check model health first
                if not self.check_model_health():
                    logger.warning("⚠️ Model server is not healthy, skipping prediction")
                    time.sleep(interval)
                    continue
                
                # Create realistic sales features
                features = self.create_realistic_sales_features(batch_size)
                
                # Make prediction
                result = self.make_prediction(features)
                
                # Print current statistics
                uptime = time.time() - self.start_time
                success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
                
                logger.info(f"📈 Statistics:")
                logger.info(f"   Total requests: {self.total_requests}")
                logger.info(f"   Successful: {self.successful_requests}")
                logger.info(f"   Failed: {self.failed_requests}")
                logger.info(f"   Success rate: {success_rate:.1f}%")
                logger.info(f"   Uptime: {uptime:.0f}s")
                
                # Show latest prediction
                if result['status'] == 'success':
                    predictions = result['predictions']
                    if isinstance(predictions, list) and predictions:
                        logger.info(f"   Latest prediction: ${predictions[0]:.2f}")
                    elif isinstance(predictions, dict) and "predictions" in predictions:
                        pred_vals = predictions["predictions"]
                        if isinstance(pred_vals, list) and pred_vals:
                            logger.info(f"   Latest prediction: ${pred_vals[0]:.2f}")
                        else:
                            logger.info(f"   Latest prediction: ${pred_vals:.2f}")
                
                # Wait for next iteration
                logger.info(f"⏳ Waiting {interval} seconds...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("⏹️ Continuous inference stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in continuous inference: {e}")
            raise
    
    def run_single_prediction(self, batch_size: int = 1):
        """Run a single prediction"""
        logger.info(f"🎯 Running single Sales Forecasting prediction (batch size: {batch_size})")
        logger.info(f"📋 Compatible dengan: data forecasting_processed.csv")
        
        # Check model health
        if not self.check_model_health():
            logger.error("❌ Model server is not healthy")
            return None
        
        # Create realistic features
        features = self.create_realistic_sales_features(batch_size)
        
        # Log feature sample for debugging
        if features:
            logger.info(f"📊 Sample feature summary (first sample):")
            logger.info(f"   Quantity: {features[0][0]:.2f}")
            logger.info(f"   UnitPrice: {features[0][1]:.2f}")
            logger.info(f"   Expected TotalSales: {features[0][0] * features[0][1]:.2f}")
            logger.info(f"   Total features: {len(features[0])}")
        
        # Make prediction
        result = self.make_prediction(features)
        
        return result
    
    def run_load_test(self, num_requests: int = 20, batch_size: int = 1, delay: float = 1.0):
        """Run load test untuk stress testing model"""
        logger.info(f"🏃 Running Sales Forecasting load test: {num_requests} requests, batch size {batch_size}")
        logger.info(f"📋 Using features compatible dengan: data forecasting_processed.csv")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            logger.info(f"Request {i+1}/{num_requests}")
            
            features = self.create_realistic_sales_features(batch_size)
            result = self.make_prediction(features)
            results.append(result)
            
            # Delay between requests
            time.sleep(delay)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = [r for r in results if r['status'] == 'success']
        durations = [r['duration'] for r in successful]
        
        logger.info(f"\n📊 Load Test Results:")
        logger.info(f"   Total requests: {num_requests}")
        logger.info(f"   Successful: {len(successful)}")
        logger.info(f"   Failed: {len(results) - len(successful)}")
        logger.info(f"   Success rate: {len(successful)/num_requests*100:.1f}%")
        logger.info(f"   Total time: {total_time:.2f}s")
        
        if durations:
            logger.info(f"   Average duration: {np.mean(durations):.3f}s")
            logger.info(f"   Min duration: {np.min(durations):.3f}s")
            logger.info(f"   Max duration: {np.max(durations):.3f}s")
            logger.info(f"   Requests per second: {len(successful)/total_time:.2f}")
            
            # Show sample predictions and statistics
            successful_with_preds = [r for r in successful if 'predictions' in r]
            if successful_with_preds:
                sample_result = successful_with_preds[0]
                pred_data = sample_result['predictions']
                
                # Extract prediction values
                all_predictions = []
                for result in successful_with_preds:
                    preds = result['predictions']
                    if isinstance(preds, list):
                        all_predictions.extend(preds)
                    elif isinstance(preds, dict) and "predictions" in preds:
                        pred_vals = preds["predictions"]
                        if isinstance(pred_vals, list):
                            all_predictions.extend(pred_vals)
                        else:
                            all_predictions.append(pred_vals)
                    else:
                        all_predictions.append(preds)
                
                if all_predictions:
                    logger.info(f"   Prediction range: ${np.min(all_predictions):.2f} - ${np.max(all_predictions):.2f}")
                    logger.info(f"   Average prediction: ${np.mean(all_predictions):.2f}")
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sales Forecasting Inference Service - Dataset Compatible')
    parser.add_argument('--model-endpoint', default='http://127.0.0.1:1234',
                       help='MLflow model endpoint')
    parser.add_argument('--prometheus-endpoint', default='http://localhost:8000',
                       help='Prometheus exporter endpoint')
    parser.add_argument('--mlflow-uri', default='http://localhost:5000',
                       help='MLflow tracking URI')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single prediction command
    single_parser = subparsers.add_parser('single', help='Run single prediction')
    single_parser.add_argument('--batch-size', type=int, default=1,
                              help='Batch size for prediction')
    
    # Continuous inference command
    continuous_parser = subparsers.add_parser('continuous', help='Run continuous inference')
    continuous_parser.add_argument('--interval', type=int, default=60,
                                  help='Interval between predictions (seconds)')
    continuous_parser.add_argument('--batch-size', type=int, default=1,
                                  help='Batch size for each prediction')
    continuous_parser.add_argument('--max-iterations', type=int, default=None,
                                  help='Maximum number of iterations')
    
    # Load test command
    loadtest_parser = subparsers.add_parser('loadtest', help='Run load test')
    loadtest_parser.add_argument('--num-requests', type=int, default=20,
                                 help='Number of requests to send')
    loadtest_parser.add_argument('--batch-size', type=int, default=1,
                                 help='Batch size for each request')
    loadtest_parser.add_argument('--delay', type=float, default=1.0,
                                 help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    # Create inference service
    service = SalesForecastingInferenceService(
        model_endpoint=args.model_endpoint,
        prometheus_endpoint=args.prometheus_endpoint,
        mlflow_tracking_uri=args.mlflow_uri
    )
    
    print(f"""
🚀 Sales Forecasting Inference Service
======================================
Model Endpoint: {args.model_endpoint}
Prometheus: {args.prometheus_endpoint}
MLflow URI: {args.mlflow_uri}

📋 Dataset Compatibility: data forecasting_processed.csv
🔧 Features: 37 (sesuai struktur dari create_time_series_features)
🎯 Compatible dengan modelling.py feature engineering
    """)
    
    # Execute commands
    if args.command == 'single':
        result = service.run_single_prediction(batch_size=args.batch_size)
        if result:
            print(f"\n✅ Prediction result:")
            print(json.dumps(result, indent=2))
    
    elif args.command == 'continuous':
        service.run_continuous_inference(
            interval=args.interval,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations
        )
    
    elif args.command == 'loadtest':
        service.run_load_test(
            num_requests=args.num_requests,
            batch_size=args.batch_size,
            delay=args.delay
        )
    
    else:
        print("❌ Please specify a command: single, continuous, or loadtest")
        print("\nExamples:")
        print("  python inference.py single")
        print("  python inference.py continuous --interval 30")
        print("  python inference.py loadtest --num-requests 10")
        parser.print_help()

if __name__ == "__main__":
    main()