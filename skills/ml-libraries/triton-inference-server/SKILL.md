---
name: triton-inference-server
description: NVIDIA Triton Inference Server — production model serving with dynamic batching, multi-framework support, concurrent execution, model ensembles, and GPU optimization. Use when deploying ML models at scale beyond what Flask/FastAPI can handle.
---

# NVIDIA Triton Inference Server

## Why This Exists

**Problem:** Serving ML models in production requires handling concurrent requests, batching for GPU efficiency, supporting multiple frameworks, orchestrating model pipelines, and monitoring performance — none of which Flask/FastAPI endpoints handle well. A naive `model.predict()` behind an HTTP server wastes GPU cycles (no batching), can't run multiple models concurrently, and offers no observability.

**Key insight:** Triton treats model serving as an infrastructure problem. It manages model lifecycle, automatically batches requests across clients, runs multiple model instances in parallel, and exposes standardized protocols (KServe) with built-in metrics — turning GPUs from idle-between-requests to saturated-at-peak-throughput.

**Reach for this when:**
- You need to serve multiple models (different frameworks) from the same infrastructure
- Single-request latency is acceptable but throughput is too low (dynamic batching helps)
- You have a model pipeline (preprocessing → model → postprocessing) that needs orchestration
- You need production observability: Prometheus metrics, health checks, model versioning
- You're deploying on NVIDIA GPUs and want to maximize utilization

---

## 1. Architecture & Key Concepts

### Model Repository

The filesystem layout that Triton reads models from:

```
model_repository/
├── text_classifier/
│   ├── config.pbtxt          # Model configuration
│   ├── 1/                    # Version 1
│   │   └── model.onnx
│   └── 2/                    # Version 2
│       └── model.onnx
├── image_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── preprocessing/
    ├── config.pbtxt
    └── 1/
        └── model.py          # Python backend
```

Rules:
- Each model gets a directory named after the model
- Versions are numbered subdirectories (1/, 2/, ...)
- `config.pbtxt` defines inputs, outputs, batching, and instance configuration
- Model files are named by convention per backend (`model.onnx`, `model.pt`, `model.plan`, `model.py`)

### Backends

| Backend | File Format | Use Case |
|---------|-------------|----------|
| TensorRT | `model.plan` | Maximum GPU inference speed (compiled) |
| ONNX Runtime | `model.onnx` | Cross-framework portability |
| PyTorch (TorchScript) | `model.pt` | Direct PyTorch model serving |
| Python | `model.py` | Custom logic, preprocessing, BLS |
| vLLM | config-driven | LLM serving with PagedAttention |
| OpenVINO | `model.xml` + `model.bin` | Intel CPU/GPU optimization |
| TensorFlow | `model.savedmodel/` | TF SavedModel serving |

### Dynamic Batching

Triton accumulates individual client requests and combines them into a batch before GPU execution. This converts N sequential inferences into 1 batched inference — dramatically improving throughput.

```
Client A (1 image)  ─┐
Client B (1 image)  ─┼─→ [Dynamic Batcher] → Batch of 4 → GPU → Split results
Client C (1 image)  ─┤
Client D (1 image)  ─┘
```

### Concurrent Model Execution

Triton can run multiple instances of the same model (on same or different GPUs) and run different models simultaneously. Instance groups control parallelism.

### Model Ensembles & BLS

- **Ensemble:** Declarative DAG pipeline defined in config.pbtxt (no code)
- **Business Logic Scripting (BLS):** Imperative pipeline in Python backend — supports loops, conditionals, data-dependent routing

---

## 2. Model Configuration (`config.pbtxt`)

### PyTorch Model

```protobuf
name: "text_classifier"
backend: "pytorch"
max_batch_size: 32

input [
  {
    name: "INPUT__0"
    data_type: TYPE_INT64
    dims: [ 512 ]
  }
]
output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
```

### ONNX Model

```protobuf
name: "image_encoder"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 200
}

# Model warmup — avoid cold start on first request
model_warmup [
  {
    name: "warmup"
    batch_size: 1
    inputs {
      key: "pixel_values"
      value: {
        data_type: TYPE_FP32
        dims: [ 3, 224, 224 ]
        zero_data: true
      }
    }
  }
]
```

### TensorRT Model (Maximum Performance)

```protobuf
name: "detector"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "detections"
    data_type: TYPE_FP32
    dims: [ 100, 6 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 50
}

response_cache {
  enable: true
}
```

---

## 3. Python Backend — Custom Models

### Basic Model (Preprocessing)

```python
# model_repository/preprocessing/1/model.py
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import json


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 512

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input text
            input_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = input_text.as_numpy()[0].decode("utf-8")

            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            # Create output tensors
            input_ids = pb_utils.Tensor("INPUT_IDS", encoded["input_ids"].astype(np.int64))
            attention_mask = pb_utils.Tensor("ATTENTION_MASK", encoded["attention_mask"].astype(np.int64))

            response = pb_utils.InferenceResponse(output_tensors=[input_ids, attention_mask])
            responses.append(response)

        return responses

    def finalize(self):
        print("Preprocessing model unloaded.")
```

Config for Python backend:
```protobuf
name: "preprocessing"
backend: "python"
max_batch_size: 0

input [
  {
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
output [
  {
    name: "INPUT_IDS"
    data_type: TYPE_INT64
    dims: [ 1, 512 ]
  },
  {
    name: "ATTENTION_MASK"
    data_type: TYPE_INT64
    dims: [ 1, 512 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
```

### BLS Model (Pipeline Orchestration)

```python
# model_repository/pipeline/1/model.py
import numpy as np
import triton_python_backend_utils as pb_utils
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

    def execute(self, requests):
        responses = []
        for request in requests:
            # Step 1: Call preprocessing model
            text_input = pb_utils.get_input_tensor_by_name(request, "RAW_TEXT")

            preprocess_request = pb_utils.InferenceRequest(
                model_name="preprocessing",
                requested_output_names=["INPUT_IDS", "ATTENTION_MASK"],
                inputs=[pb_utils.Tensor("TEXT", text_input.as_numpy())]
            )
            preprocess_response = preprocess_request.exec()

            if preprocess_response.has_error():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(preprocess_response.error().message())))
                continue

            # Step 2: Call the classifier model
            input_ids = pb_utils.get_output_tensor_by_name(preprocess_response, "INPUT_IDS")
            attention_mask = pb_utils.get_output_tensor_by_name(preprocess_response, "ATTENTION_MASK")

            classify_request = pb_utils.InferenceRequest(
                model_name="text_classifier",
                requested_output_names=["OUTPUT__0"],
                inputs=[
                    pb_utils.Tensor("INPUT__0", input_ids.as_numpy())
                ]
            )
            classify_response = classify_request.exec()

            if classify_response.has_error():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(classify_response.error().message())))
                continue

            # Step 3: Post-process (argmax)
            logits = pb_utils.get_output_tensor_by_name(classify_response, "OUTPUT__0").as_numpy()
            prediction = np.argmax(logits, axis=-1).astype(np.int32)

            output = pb_utils.Tensor("PREDICTION", prediction)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))

        return responses
```

### Ensemble (Declarative Pipeline)

```protobuf
name: "ensemble_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "RAW_IMAGE"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]
  }
]
output [
  {
    name: "CLASSIFICATION"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "image_preprocess"
      model_version: -1
      input_map {
        key: "RAW_INPUT"
        value: "RAW_IMAGE"
      }
      output_map {
        key: "PROCESSED"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "resnet50"
      model_version: -1
      input_map {
        key: "input"
        value: "preprocessed_image"
      }
      output_map {
        key: "output"
        value: "CLASSIFICATION"
      }
    }
  ]
}
```

---

## 4. Client Inference

### Python Client (tritonclient)

```python
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np

# --- HTTP Client ---
client = httpclient.InferenceServerClient(url="localhost:8000")

# Check server/model health
assert client.is_server_ready()
assert client.is_model_ready("text_classifier")

# Prepare input
input_data = np.random.randint(0, 30000, size=(1, 512), dtype=np.int64)
inputs = [httpclient.InferInput("INPUT__0", input_data.shape, "INT64")]
inputs[0].set_data_from_numpy(input_data)

# Request specific outputs
outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]

# Infer
result = client.infer(model_name="text_classifier", inputs=inputs, outputs=outputs)
prediction = result.as_numpy("OUTPUT__0")
print(f"Prediction shape: {prediction.shape}")  # (1, 10)

# --- gRPC Client (higher throughput) ---
grpc_client = grpcclient.InferenceServerClient(url="localhost:8001")

inputs = [grpcclient.InferInput("INPUT__0", [1, 512], "INT64")]
inputs[0].set_data_from_numpy(input_data)
outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

result = grpc_client.infer(model_name="text_classifier", inputs=inputs, outputs=outputs)
print(result.as_numpy("OUTPUT__0"))
```

### Async Client (High Throughput)

```python
import tritonclient.grpc.aio as aio_grpc
import asyncio
import numpy as np


async def infer_batch(client, model_name, batch):
    inputs = [aio_grpc.InferInput("INPUT__0", batch.shape, "INT64")]
    inputs[0].set_data_from_numpy(batch)
    outputs = [aio_grpc.InferRequestedOutput("OUTPUT__0")]
    result = await client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return result.as_numpy("OUTPUT__0")


async def main():
    client = aio_grpc.InferenceServerClient(url="localhost:8001")

    # Fire multiple concurrent requests
    batches = [np.random.randint(0, 30000, (4, 512), dtype=np.int64) for _ in range(10)]
    tasks = [infer_batch(client, "text_classifier", b) for b in batches]
    results = await asyncio.gather(*tasks)

    for i, r in enumerate(results):
        print(f"Batch {i}: {r.shape}")

asyncio.run(main())
```

---

## 5. Performance Tuning

### Instance Groups — Concurrent Execution

```protobuf
# 2 instances on GPU 0, 1 instance on GPU 1
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0 ] },
  { count: 1, kind: KIND_GPU, gpus: [ 1 ] }
]
```

**Tuning guideline:** Start with 1 instance per GPU. Add more instances when GPU utilization < 80% and latency allows. Each instance consumes GPU memory — profile with `nvidia-smi`.

### Dynamic Batching Parameters

```protobuf
dynamic_batching {
  # Maximum requests to batch together (set by max_batch_size at top level)

  # Preferred sizes — Triton will try to form these batch sizes
  preferred_batch_size: [ 8, 16, 32 ]

  # Maximum time to wait for more requests to form a batch (microseconds)
  # Lower = less latency, Higher = better throughput
  max_queue_delay_microseconds: 100

  # Priority levels (optional)
  priority_levels: 3
  default_priority_level: 2

  # Queue policy per priority
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 5000000  # 5 seconds
    allow_timeout_override: true
    max_queue_size: 100
  }
}
```

| Parameter | Low Latency | High Throughput |
|-----------|-------------|-----------------|
| `max_queue_delay_microseconds` | 50–100 | 500–5000 |
| `preferred_batch_size` | [1, 2, 4] | [16, 32, 64] |
| `max_batch_size` | 8 | 64–128 |
| Instance count | 1 | 2–4 per GPU |

### Model Warmup

Prevents cold-start latency on first request:

```protobuf
model_warmup [
  {
    name: "warmup_batch"
    batch_size: 8
    inputs {
      key: "INPUT__0"
      value: {
        data_type: TYPE_INT64
        dims: [ 512 ]
        zero_data: true
      }
    }
  }
]
```

### Response Cache

Cache identical requests to avoid recomputation:

```protobuf
response_cache {
  enable: true
}
```

Server must be started with cache config:
```bash
tritonserver --model-repository=/models \
  --cache-config local,size=1048576  # 1MB cache
```

---

## 6. Deployment

### Docker

```bash
# Basic GPU deployment
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.05-py3 \
  tritonserver --model-repository=/models

# With explicit model loading (don't auto-load all models)
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.05-py3 \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model=text_classifier \
    --load-model=image_encoder
```

### Kubernetes (Full Manifest)

```yaml
# triton-deployment.yaml — complete Deployment + Service + HPA
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
  labels:
    app: triton
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
    spec:
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.05-py3
          args:
            - tritonserver
            - --model-repository=s3://my-bucket/model_repository
            - --log-verbose=0
            - --strict-model-config=false
          ports:
            - name: http
              containerPort: 8000
            - name: grpc
              containerPort: 8001
            - name: metrics
              containerPort: 8002
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
            requests:
              nvidia.com/gpu: 1
              memory: "8Gi"
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          volumeMounts:
            - name: model-cache
              mountPath: /models
      volumes:
        - name: model-cache
          emptyDir: {}
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: triton-inference-server
spec:
  selector:
    app: triton
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 8001
      targetPort: 8001
    - name: metrics
      port: 8002
      targetPort: 8002
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference-server
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: nv_inference_queue_duration_us  # Triton's queue wait metric
        target:
          type: AverageValue
          averageValue: "5000"  # scale up if avg queue > 5ms
```

### Helm Chart (Alternative)

```bash
# NVIDIA official Helm chart
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install triton nvidia/triton-inference-server \
  --set image.tag=24.05-py3 \
  --set modelRepository.path=s3://my-bucket/model_repository \
  --set resources.limits.nvidia\\.com/gpu=1 \
  --set replicaCount=2
```

Helm chart reference: https://github.com/triton-inference-server/server/tree/main/deploy

### Local Model Repository (PVC Instead of S3)

```yaml
# For serving your own fine-tuned models from persistent storage
volumes:
  - name: model-repo
    persistentVolumeClaim:
      claimName: triton-models  # pre-populated with model_repository/
containers:
  - name: triton
    args:
      - tritonserver
      - --model-repository=/models
    volumeMounts:
      - name: model-repo
        mountPath: /models
```

### Health Checks & Model Management API

```bash
# Server health
curl localhost:8000/v2/health/live     # → 200 if server is alive
curl localhost:8000/v2/health/ready    # → 200 if all models loaded

# Model status
curl localhost:8000/v2/models/text_classifier/ready  # → 200 if model ready

# Model metadata (inputs/outputs/config)
curl localhost:8000/v2/models/text_classifier

# Load/unload models (requires --model-control-mode=explicit)
curl -X POST localhost:8000/v2/repository/models/text_classifier/load
curl -X POST localhost:8000/v2/repository/models/text_classifier/unload

# List all models
curl localhost:8000/v2/repository/index
```

### Prometheus Metrics

Triton exposes metrics on port 8002:

```bash
curl localhost:8002/metrics
```

Key metrics:
| Metric | Description |
|--------|-------------|
| `nv_inference_request_success` | Successful inference count |
| `nv_inference_request_failure` | Failed inference count |
| `nv_inference_count` | Total inferences (batch elements) |
| `nv_inference_exec_count` | Total inference executions (batches) |
| `nv_inference_request_duration_us` | End-to-end request latency |
| `nv_inference_queue_duration_us` | Time spent in queue (batching wait) |
| `nv_inference_compute_infer_duration_us` | Actual GPU compute time |
| `nv_gpu_utilization` | GPU utilization percentage |
| `nv_gpu_memory_used_bytes` | GPU memory usage |

---

## 7. When to Use Triton vs Alternatives

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Multiple models, different frameworks, single infrastructure | **Triton** | Multi-backend support, shared GPU |
| LLM serving (chat, generation) | **vLLM** or **TGI** | PagedAttention, continuous batching optimized for autoregressive |
| LLM on Triton specifically | **Triton + vLLM backend** | Gets Triton's infra (metrics, ensemble) + vLLM's efficiency |
| Simple single-model API, rapid prototyping | **Ray Serve** | Easier setup, Python-native, good autoscaling |
| PyTorch models only, simple deployment | **TorchServe** | Tight PyTorch integration, simpler config |
| Maximum GPU throughput, multi-model | **Triton** | Dynamic batching + concurrent execution |
| Kubernetes-native with autoscaling | **Triton + KServe** | Standardized inference protocol, canary deployments |
| Need custom pre/post-processing in pipeline | **Triton (BLS)** or **Ray Serve** | Both support custom Python logic in pipeline |
| Edge/embedded (Jetson) | **Triton** | Direct Jetson support with TensorRT |
| CPU-only inference | **Triton (ONNX/OpenVINO)** or **Ray Serve** | Both work, Triton adds batching benefit |

### Decision Matrix

```
Need multi-framework support?
  YES → Triton
  NO →
    Serving LLMs?
      YES → vLLM (standalone) or Triton + vLLM backend (if you need ensembles/metrics)
      NO →
        Need dynamic batching + high throughput?
          YES → Triton
          NO →
            Need complex Python orchestration?
              YES → Ray Serve or Triton BLS
              NO → TorchServe / FastAPI (simple cases)
```

---

## 8. Gotchas

1. **PyTorch input naming:** Inputs must match forward() argument names OR use `NAME__INDEX` convention (e.g., `INPUT__0`, `INPUT__1`)
2. **Batch dimension is implicit:** `dims: [512]` with `max_batch_size: 32` means actual tensor shape is `[batch, 512]`. Don't include batch dim in config.
3. **max_batch_size: 0** disables batching entirely — use for models that handle their own batching
4. **Python backend shared memory:** Default 1MB per instance, grows automatically. Set `--backend-config=python,shm-default-byte-size=4194304` for large tensors
5. **Model warmup blocks readiness:** Triton won't report model as ready until warmup completes — plan for startup time
6. **Response cache only works with identical inputs** — not useful for models with random sampling
7. **gRPC is ~2-3x faster than HTTP** for high-throughput scenarios — use port 8001
8. **Instance count × model size must fit in GPU memory** — 4 instances of a 4GB model needs 16GB GPU

---

## References

- Server repository: https://github.com/triton-inference-server/server
- Client libraries: https://github.com/triton-inference-server/client
- Python backend: https://github.com/triton-inference-server/python_backend
- Official documentation: https://docs.nvidia.com/triton-inference-server/
- Tutorials: https://github.com/triton-inference-server/tutorials
- Model configuration protobuf: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
- Performance Analyzer: https://github.com/triton-inference-server/perf_analyzer
- Model Analyzer: https://github.com/triton-inference-server/model_analyzer
- KServe protocol spec: https://github.com/kserve/kserve/tree/master/docs/predict-api/v2
- NGC container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
