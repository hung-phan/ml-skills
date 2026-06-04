---
name: ml-architectures
description: Deep learning architecture skills with PyTorch patterns and decision guides. Covers attention, transformers, CNNs, RNNs, GANs, diffusion, MoE, and more.
---

# ML Architectures

## Skills

| Skill | Description |
|-------|-------------|
| [ANN](ann/) | Artificial Neural Networks — perceptron to deep MLP, activation functions, weight initialization, dropout |
| [Attention](attention/) | All attention mechanisms — MHA, MQA, GQA, MLA, FlashAttention, PagedAttention, KV-cache math, serving patterns |
| [Autoencoder](autoencoder/) | Autoencoders — vanilla, variational (VAE), denoising, sparse, and contrastive variants |
| [Boltzmann](boltzmann/) | Boltzmann Machines — energy-based models, RBMs, Deep Belief Networks, contrastive divergence |
| [CNN](cnn/) | Convolutional Neural Networks — conv layers, pooling, ResNet, EfficientNet, transfer learning |
| [Diffusion](diffusion/) | Diffusion Models — DDPM, DDIM, score-based generative modeling, classifier-free guidance |
| [Embeddings](embeddings/) | Dense vector representations — sentence-transformers, contrastive learning, FAISS, Milvus, CLIP |
| [GAN](gan/) | Generative Adversarial Networks — DCGAN, WGAN-GP, StyleGAN, training stability |
| [GNN](gnn/) | Graph Neural Networks — message passing, GCN, GAT, GraphSAGE, PyG patterns |
| [LLM](llm/) | Large Language Models — GPT/BERT/T5, RoPE, GQA, SwiGLU, Flash Attention, LoRA/QLoRA, DPO, vLLM |
| [Mamba](mamba/) | State Space Models — selective SSMs, hardware-aware parallel scan, Mamba-2 SSD, Jamba hybrid |
| [Mixture of Experts](mixture-of-experts/) | MoE — sparse activation, top-k routing, load balancing, Switch Transformer, Mixtral, DeepSeek-V2 |
| [Quantization](quantization/) | GPTQ, AWQ, FP8, INT8, bitsandbytes, GGUF — memory reduction for inference and QLoRA training |
| [Regression & Classification](regression-classification/) | Supervised learning foundations — loss functions, metrics, imbalance, calibration, pipelines |
| [Reinforcement Learning](reinforcement-learning/) | RL algorithms — DQN, PPO, SAC, gymnasium, stable-baselines3, CleanRL, multi-agent |
| [RNN](rnn/) | Recurrent Networks — LSTM, GRU, bidirectional, seq2seq + attention, packed sequences |
| [SOM](som/) | Self-Organizing Maps — competitive learning, U-matrix visualization, minisom, anomaly detection |
| [Transformer](transformer/) | Transformer architecture — self-attention, multi-head, positional encodings, KV-cache, GPT impl |
| [Vision](vision/) | Vision models — ViT, Swin, CLIP, SAM, DINOv2, DETR, YOLOv8, timm, transfer learning |

## When to Use What

| Task | Start With |
|------|-----------|
| Tabular data | Regression & Classification → XGBoost/MLP |
| Short sequences (<500 tokens) | RNN (LSTM/GRU) |
| Long sequences, parallel training | Transformer |
| Ultra-long sequences (>8K), streaming | Mamba |
| Images | CNN or Vision (ViT/Swin) |
| Graphs/networks | GNN |
| Generation (images) | Diffusion or GAN |
| Generation (text) | LLM (Transformer decoder-only) |
| Scale model without proportional compute | Mixture of Experts |
| Clustering/visualization | SOM |
| Feature learning (unsupervised) | Autoencoder or Boltzmann |
| Decision making | Reinforcement Learning |
| Optimizing inference memory/speed | Attention (GQA, MLA, FlashAttention) |
| Retrieval, similarity search, RAG | Embeddings (bi-encoder + vector DB) |
| Shrink model to fit on GPU | Quantization (AWQ/GPTQ/FP8) |
