



1）构建命令
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg TORCH_BASE=full \
  --build-arg WORKFLOW=false \
  -t gpt-sovits:custom .
COMPOSE_BAKE=true docker-compose build --progress=plain tts-server

conda activate GPTSoVits
COMPOSE_BAKE=true docker compose  --progress plain  build