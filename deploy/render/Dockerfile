FROM node:22-bookworm-slim AS web-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=10000 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY submission/nebulax/app/requirements.lock.txt requirements-r2.txt /tmp/
RUN pip install -r /tmp/requirements.lock.txt -r /tmp/requirements-r2.txt
COPY nebulax/ ./nebulax/
COPY models/ps3/ ./models/ps3/
COPY demo_data/ ./demo_data/
COPY results/ps3/ ./results/ps3/
COPY results/leaderboard.md ./results/leaderboard.md
COPY --from=web-build /build/web/dist/ ./web/dist/
RUN useradd --create-home --uid 1000 user && mkdir -p data/ps3_cache/uploads && chown -R user:user /app
USER user
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn nebulax.api.free_app:create_app --factory --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
