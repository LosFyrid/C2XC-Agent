FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal OS packages: keep lean; no build toolchain required for the pinned wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (include HKU LightRAG; import name remains `lightrag`).
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -U pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt "lightrag-hku==1.4.9.10"

# Copy runtime code + config + priors.
COPY src /app/src
COPY config /app/config
COPY docs /app/docs

# Non-root runtime user (data is mounted via k8s hostPath/PVC and chowned by initContainer).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

