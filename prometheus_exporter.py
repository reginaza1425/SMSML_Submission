from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'model_request_total',
    'Total request model'
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency model inference'
)
