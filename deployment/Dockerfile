# CPU-only inference image built from the standalone package (no torch, no training code).
#   python scripts/build_package.py --ckpt checkpoints/best
#   docker build -f deployment/Dockerfile --build-arg PKG=dist/filmlook-infer-1.0.0 -t filmlook-infer:1.0.0 .
#   docker run --rm --memory=256m --cpus=0.25 -v $PWD/photo.jpg:/in.jpg:ro filmlook-infer:1.0.0 /in.jpg --lang zh
FROM python:3.12-slim
ARG PKG=dist/filmlook-infer-1.0.0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    FILMLOOK_MODEL_DIR=/app/model
WORKDIR /app
COPY ${PKG}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ${PKG}/filmlook_infer/ filmlook_infer/
COPY ${PKG}/model/ model/
RUN useradd -r -u 10001 app && chown -R app /app
USER app
ENTRYPOINT ["python", "-m", "filmlook_infer.cli"]
