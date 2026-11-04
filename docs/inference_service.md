# Inference service (laptop)

The inference service scores station pictures on the laptop GPU with each part's approved model. The Pi station calls it over the LAN (`station/remote_model.py`, plan step S2). It returns raw scores only; the station applies `t_low` and `t_img`.

Code: `inference_service/` (`server.py`, `registry.py`, `__main__.py`). Standard library at import time. torch, numpy and cv2 load only when a model is loaded.

## Run it

From the `vision-station` folder, on the laptop (PowerShell):

```powershell
$env:INFERENCE_TOKEN = "<the token, set by you, never written to a file>"
python -m inference_service --models "<ModelStore folder>" --host 127.0.0.1 --port 8765 --token-env INFERENCE_TOKEN --extractor-config "<extractor JSON>"
```

- `--models`: the `ModelStore` root (one folder per part, with `current.json`).
- `--host`: the address to bind. Default `127.0.0.1`. The LAN address is `[Engineer to define]`; set it only after the pilot decision on LAN security (plan decision 6).
- `--port`: default `8765` (an engineering default, change it if the port is taken).
- `--token-env`: the NAME of the environment variable that holds the token. Default `INFERENCE_TOKEN`.
- `--extractor-config`: a JSON file with the same keys as `config.model.extractor` in the station: `arch`, `device`, `input_hw`, `weights_path`. Without it, patchcore models return 503.
- `--max-body-bytes`: largest picture accepted (default 32 MiB).

The service refuses to start when the environment variable is empty or unset. It prints the variable's name, never its value.

## Environment variables it uses

The service reads one environment variable itself. Set its value yourself; never paste it into a file, a log or a chat.

| Name | Used for |
|---|---|
| `INFERENCE_TOKEN` | the bearer token every request must send. This is the default name; `--token-env` names another variable |

The service does not read any other variable. torch and cv2 may read their own standard settings when they load, but this service does not depend on them.
## Endpoints

`POST /v1/predict`
- Headers: `Authorization: Bearer <token>`, `Part-Number`, `Camera-Id` (1 to 16 letters, digits, `_` or `-`), `Side` (`TOP` or `BOTTOM`), `Content-Type: image/png` or `image/jpeg`.
- Body: the picture bytes (raw, not multipart). `Content-Length` is required.
- 200: `{"camera_id", "side", "score", "defects": [{"defect_class", "score", "box": [x, y, w, h]}], "model_version"}`.

`GET /v1/health` (also needs the token)
- 200: `{"status", "version", "models": [{"part_number", "version"}], "busy", "in_flight", "gpu"}`. `busy` is true while a predict request is in progress. The weekly training script reads it and waits.

Error answers are JSON `{"error": "..."}`:

| Status | Meaning |
|---|---|
| 400, 411, 413, 415 | bad headers, no length, picture too large, wrong type |
| 401 | missing or wrong token |
| 422 | the model rejected this input (for example an unknown camera or side) |
| 503 | no approved model, a tampered or unreadable version, torch missing, or GPU out of memory |
| 500 | unexpected failure (details go to stderr as the exception name only) |

A 503 or any non-200 makes the station go to FAULT. It never means GOOD.

## Model rules

- Only the version `ModelStore.current()` returns is loaded. Its sha256 is checked before loading.
- A version that fails verification is never replaced by an older one.
- The model is cached per part and reloaded when `current.json` changes (a promotion or a rollback).
- A GPU out-of-memory error unloads the model and frees the GPU cache. The next request reloads it.
- A file changed on disk after it was loaded is caught at the next reload, not on every request (hashing a model on every request would be too slow).

## Limits and open items

- Only `patchcore` models are served. Golden models are not.
- Plain HTTP, no TLS. The pilot runs on the plant LAN only after the owner confirms the LAN is isolated (plan decision 6).
- The token is compared in constant time. It is never returned, logged or echoed in an error.
- Not yet measured: the laptop round trip, the GPU memory headroom, and the `RemoteModel` timeout (`[Engineer to define]`).
- Not decided here: the LAN address, the backup folder and any threshold. None of those belong to this service.
