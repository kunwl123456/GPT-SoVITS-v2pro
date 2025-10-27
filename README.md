



1）构建命令
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg TORCH_BASE=full \
  --build-arg WORKFLOW=false \
  -t gpt-sovits:custom .
COMPOSE_BAKE=true docker-compose build --progress=plain tts-server

conda activate GPTSoVits
COMPOSE_BAKE=true sudo docker compose  --progress plain  build


2)修改更新笔记
（1）10/28/2025
1. ✅ **BERT 维度** - 固定为 1024 维
2. ✅ **Pow 操作** - 全部替换为乘法/sqrt
3. ✅ **Squeeze** - 明确指定维度参数
4. ✅ **Softmax** - 确保至少 2 维输入 + 用索引提取
5. ✅ **ArgMax** - 确保至少 2 维输入 + 用索引提取