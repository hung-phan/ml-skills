---
name: keras
description: Keras 3 multi-backend (TensorFlow/JAX/PyTorch) deep learning — Sequential and Functional APIs, transfer learning, callbacks, and rapid prototyping. Use when prototyping neural networks with high-level Keras APIs or building backend-portable models.
---

# Keras Skill Reference

## Why This Exists

1. **Problem solved**: Building neural networks without boilerplate training loops, device management, or gradient plumbing. Keras provides a high-level API where you define layers, call `model.fit()`, and get training with callbacks, metrics, and multi-GPU support — reducing hundreds of lines of PyTorch training code to a few.

2. **When to pick this over alternatives**: Choose Keras over PyTorch when you want fast prototyping of standard architectures, readable model definitions for education/collaboration, or backend flexibility (run same model on JAX, TF, or PyTorch). Choose PyTorch when you need custom training loops, low-level gradient manipulation, or the HuggingFace ecosystem. Choose JAX directly when you need functional transformations (vmap, pmap) or XLA compilation control.

3. **Mental model**: keras = layer graph + compile + fit. You build a graph of Layer objects (Sequential, Functional, or Subclass), compile it with an optimizer/loss/metrics specification, then call `fit()` which handles the training loop, callbacks, and distribution automatically. Layers own weights; Models own layers; `compile()` wires the optimizer; `fit()` runs the loop.

> Keras 3 — multi-backend deep learning (JAX, PyTorch, TensorFlow). Default backend: TensorFlow.

## Model APIs

### Sequential — linear stack of layers
```python
import keras
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
```

### Functional — arbitrary DAGs, multi-input/output
```python
inputs = keras.Input(shape=(224, 224, 3))
x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

### Subclassing — full control, imperative forward pass
```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation="relu")
        self.dense2 = keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.dense1(x)
        return self.dense2(x)
```

**When to use what:**
- Sequential: simple feed-forward, fast prototyping
- Functional: shared layers, skip connections, multiple I/O, most production models
- Subclassing: dynamic architectures, research, custom forward logic

---

## Common Layers

| Layer | Use Case | Key Args |
|-------|----------|----------|
| `Dense(units, activation)` | Fully connected | `units`, `activation` |
| `Conv2D(filters, kernel_size)` | Spatial features | `strides`, `padding="same"` |
| `LSTM(units, return_sequences)` | Sequences | `dropout`, `recurrent_dropout` |
| `GRU(units)` | Lighter RNN | same as LSTM |
| `Attention()` | Luong-style attention | `use_scale=True` |
| `MultiHeadAttention(num_heads, key_dim)` | Transformer attention | `value_dim`, `dropout` |
| `LayerNormalization()` | Transformer norm | `epsilon=1e-6` |
| `Embedding(input_dim, output_dim)` | Token embeddings | `mask_zero=True` |
| `GlobalAveragePooling2D()` | Spatial reduction | — |
| `BatchNormalization()` | Training stabilization | `momentum=0.99` |
| `Dropout(rate)` | Regularization | only active `training=True` |

### Transformer block pattern
```python
def transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.1):
    attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = keras.layers.Dropout(dropout)(attn)
    x = keras.layers.LayerNormalization()(x + attn)
    ff = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = keras.layers.Dense(x.shape[-1])(ff)
    ff = keras.layers.Dropout(dropout)(ff)
    return keras.layers.LayerNormalization()(x + ff)
```

---

## Compile Patterns

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",  # integer labels
    metrics=["accuracy"],
)
```

### Optimizer selection
| Optimizer | When |
|-----------|------|
| `Adam(lr=1e-3)` | Default, works well broadly |
| `AdamW(lr=1e-3, weight_decay=0.01)` | Transformers, fine-tuning |
| `SGD(lr=0.1, momentum=0.9)` | CNNs, large-batch training |
| `Lion(lr=1e-4)` | Memory-efficient Adam alternative |

### Loss selection
| Loss | Task |
|------|------|
| `sparse_categorical_crossentropy` | Multi-class, integer labels |
| `categorical_crossentropy` | Multi-class, one-hot labels |
| `binary_crossentropy` | Binary/multi-label |
| `mse` / `mae` | Regression |
| `CTC` | Sequence-to-sequence alignment |

### Metrics
```python
metrics=[
    "accuracy",
    keras.metrics.AUC(name="auc"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.F1Score(average="macro"),
]
```

---

## Callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training.csv"),
]

model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

### Custom callback
```python
class TimingCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: {time.time() - self.epoch_start:.1f}s")
```

---

## Data Pipelines

### tf.data (recommended for TF backend)
```python
import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds = (
    ds.shuffle(10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# From files
ds = tf.data.Dataset.list_files("data/*.jpg")
ds = ds.map(lambda path: load_and_preprocess(path), num_parallel_calls=tf.data.AUTOTUNE)
```

### keras.utils.PyDataset (Keras 3, all backends)
```python
class MyDataset(keras.utils.PyDataset):
    def __init__(self, x, y, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.x[start:end], self.y[start:end]
```

### Image augmentation
```python
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])
# Apply in model or as preprocessing
```

---

## Custom Training Loop

```python
model = MyModel()
optimizer = keras.optimizers.Adam(1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function  # or use keras.backend agnostic ops
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(epochs):
    for x_batch, y_batch in train_ds:
        loss = train_step(x_batch, y_batch)
```

### Keras 3 backend-agnostic training loop
```python
import keras.ops as ops

class CustomTrainer(keras.Model):
    def train_step(self, data):
        x, y = data
        with keras.backend.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

---

## Transfer Learning

### Feature extraction (freeze base)
```python
base = keras.applications.EfficientNetV2B0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False

model = keras.Sequential([
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
model.fit(train_ds, epochs=10)
```

### Fine-tuning (unfreeze top layers)
```python
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
)
model.fit(train_ds, epochs=5)
```

### Available pretrained models
- **Vision**: EfficientNetV2, ResNet, ConvNeXt, ViT (via keras_cv)
- **NLP**: BERT, GPT-2, T5 (via keras_nlp / keras_hub)
- **Audio**: via keras_hub

---

## Mixed Precision

```python
keras.mixed_precision.set_global_policy("mixed_float16")

# Model layers auto-cast to float16, loss computed in float32
model = build_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# Ensure output layer uses float32
outputs = keras.layers.Dense(10, dtype="float32")(x)
```

**Benefits**: ~2x throughput on NVIDIA GPUs (V100+), reduced memory.
**Caution**: Numerical instability possible — keep final layer and loss in float32.

---

## Multi-GPU Training

### TensorFlow backend — MirroredStrategy
```python
strategy = tf.distribute.MirroredStrategy()  # all visible GPUs

with strategy.scope():
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

model.fit(train_ds, epochs=10)  # auto-distributed
```

### Multi-worker
```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
# Set TF_CONFIG env var on each worker
```

### JAX backend — data parallelism
```python
# Keras 3 + JAX: use keras.distribution
distribution = keras.distribution.DataParallel(devices=keras.distribution.list_devices())
keras.distribution.set_distribution(distribution)
model = build_model()
model.compile(...)
model.fit(...)
```

---

## Model Saving & Loading

### Keras 3 native format (.keras) — recommended
```python
model.save("model.keras")                    # full model
loaded = keras.models.load_model("model.keras")
```

### Weights only
```python
model.save_weights("weights.weights.h5")
model.load_weights("weights.weights.h5")
```

### Export for serving
```python
# TF SavedModel (TF backend only)
model.export("saved_model/")

# ONNX (via tf2onnx or keras2onnx)
```

### Checkpoint during training
```python
keras.callbacks.ModelCheckpoint(
    "checkpoints/epoch_{epoch:02d}.keras",
    save_freq="epoch",
)
```

---

## Keras 3 Multi-Backend

Keras 3 supports JAX, PyTorch, and TensorFlow as backends.

### Select backend
```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "torch", "tensorflow"
import keras
```

### Backend-agnostic ops
```python
import keras.ops as ops

x = ops.convert_to_tensor([1.0, 2.0, 3.0])
y = ops.relu(x)
z = ops.matmul(a, b)
s = ops.softmax(logits, axis=-1)
```

### Key differences by backend

| Feature | TensorFlow | JAX | PyTorch |
|---------|-----------|-----|---------|
| Eager by default | Yes | Yes (via jit) | Yes |
| tf.data pipelines | Native | Via tf interop | torch DataLoader |
| Mixed precision | `set_global_policy` | `jax.default_backend` | `torch.amp` |
| Multi-GPU | MirroredStrategy | `keras.distribution` | `keras.distribution` |
| Custom ops | tf.custom_gradient | jax.custom_vjp | torch.autograd |
| Model export | SavedModel, TFLite | StableHLO | TorchScript |

### Conditional backend code
```python
if keras.backend.backend() == "tensorflow":
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(data)
elif keras.backend.backend() == "torch":
    from torch.utils.data import DataLoader
    ds = DataLoader(dataset, batch_size=32)
```

---

## Common Patterns & Tips

### Learning rate schedules
```python
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=1e-5
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Gradient clipping
```python
optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
```

### Label smoothing
```python
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

### Class weights for imbalanced data
```python
class_weight = {0: 1.0, 1: 5.0}  # upsample minority
model.fit(train_ds, class_weight=class_weight)
```

### Reproducibility
```python
keras.utils.set_random_seed(42)  # seeds all backends
```

### Memory-efficient large models
```python
# Gradient checkpointing (TF backend)
model.compile(..., steps_per_execution=8)  # fuse steps

# Keras 3
keras.config.set_dtype_policy("mixed_bfloat16")  # better than float16 for training
```

## When to Use

| ✅ Use Keras | ❌ Don't Use |
|---|---|
| Fast prototyping, standard architectures | Custom training loops with full control |
| Multi-backend (JAX, TF, PyTorch) | Research needing low-level gradient manipulation |
| Education, readable model definitions | Custom CUDA kernels or ops |
| Transfer learning with pretrained models | When team already uses pure PyTorch |
| Production with TF Serving/TFLite | When you need PyTorch ecosystem (timm, HF) |

**Decision rule**: Prototyping or standard architectures → Keras. Research or custom training → PyTorch.

---

## References

- [Keras API Reference](https://keras.io/api/)
- [Keras GitHub](https://github.com/keras-team/keras)