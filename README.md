# Kaggle Competition
online advisor for approving loans based on credit history using machine learning

## Hello World Python Service (FastAPI)

A minimal HTTPS-ready Python "hello world" web service built with FastAPI.

### Install

```bash
pip install -r requirements.txt
```

### Run (HTTP)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then visit http://localhost:8000/ to get `{"message": "Hello World"}`.

### Run with HTTPS / TLS

TLS is typically terminated by the host or a reverse proxy (e.g. nginx, a load
balancer, or your cloud platform). For local HTTPS you can pass certificates
directly to uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem
```

Generate a self-signed certificate for local testing:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365
```

Then visit https://localhost:8443/.

### Test

```bash
python -m pytest tests/test_main.py -x -q
```
