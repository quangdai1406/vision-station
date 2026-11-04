"""Entry point: ``python -m inference_service --models <dir> --host 127.0.0.1 --port 8765``."""
from inference_service.server import main

if __name__ == "__main__":
    raise SystemExit(main())
