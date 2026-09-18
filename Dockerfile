# The browser build and Python API share one runtime image.
FROM node:22-bookworm-slim AS web-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080
WORKDIR /app

COPY submission/nebulax/app/requirements.lock.txt /tmp/requirements.lock.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock.txt

COPY nebulax/ ./nebulax/
COPY models/ps3/ ./models/ps3/
COPY demo_data/ ./demo_data/
COPY results/ps3/ ./results/ps3/
COPY results/leaderboard.md ./results/leaderboard.md
COPY --from=web-build /build/web/dist/ ./web/dist/

RUN mkdir -p data/ps3_cache/uploads && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "exec python -m uvicorn nebulax.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
