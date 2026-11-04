"""Laptop inference service: scores station pictures with the approved part model (plan L1).

Run with ``python -m inference_service --models <dir> --host 127.0.0.1 --port 8765 --token-env INFERENCE_TOKEN``.
Standard library at import time; torch, numpy and cv2 load only when a model is loaded.
See docs/inference_service.md.
"""
