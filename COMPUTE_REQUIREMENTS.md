# Compute Requirements

This document describes the expected runtime environment and compute resources for GeoOpKG.

## Reference Hardware (Developer Machine)

- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA GeForce RTX 3060 (6 GB VRAM)
- RAM: 16 GB or more recommended
- Storage: 10 GB free space or more recommended

## Software Requirements

- OS: Windows 10/11, Linux, or macOS
- Python: 3.9+ (3.10+ recommended)
- Neo4j: local instance (default `bolt://localhost:7687`)
- Weaviate: local instance (default HTTP 8080, gRPC 50051)
- CUDA: required for GPU acceleration (match PyTorch build)

## Runtime Notes

- The QA pipeline uses `sentence-transformers`, `torch`, and `torch-geometric`. GPU is optional but recommended for faster embedding and GNN inference.
- If running on CPU only, expect slower response times for large graphs or long queries.
- Ensure sufficient disk space for Neo4j and Weaviate data stores, especially when importing multiple software datasets.

## Recommended Environment (Example)

- PyTorch: CUDA-enabled build compatible with your GPU driver
- torch-geometric: version compatible with the installed PyTorch version
- Weaviate: local container or native install with persistence enabled

## Minimal Environment (CPU-only)

- No GPU required
- PyTorch CPU build
- Expect longer execution times for embedding and GNN steps
