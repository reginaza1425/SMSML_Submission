import time
import random
from prometheus_client import start_http_server, Counter, Histogram
import joblib

# ==========================
# Load Model
# ==========================
model = joblib.load("model_fraud_detection/model.pkl")

# ==========================
# Define Metrics
# ==========================
REQUEST_COUNT = Counter(
    'model_request_total',
    'Total number of prediction requests'
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency of prediction requests'
)

# ==========================
# Dummy Inference Function
# ==========================
@REQUEST_LATENCY.time()
def predict(data):
    REQUEST_COUNT.inc()
    time.sleep(random.uniform(0.1, 0.5))  # simulasi latency
    return model.predict(data)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    print("Starting inference server on port 8000...")
    start_http_server(8000)

    while True:
        dummy_data = [[random.random() for _ in range(10)]]
        predict(dummy_data)
