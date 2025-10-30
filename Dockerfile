ARG CUDA_VERSION=12.8
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}
#FROM registry.cn-hangzhou.aliyuncs.com/nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
#FROM ccr.ccs.tencentyun.com/nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
#FROM registry.cn-hangzhou.aliyuncs.com/nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
#FROM nvcr.io/nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

#FROM xxxxrt666/gpt-sovits:latest-cu128
#xxxxrt666/gpt-sovits:latest-cu128


LABEL maintainer="magi_tts"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS"

ENV CUDA_VERSION=${CUDA_VERSION}

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

# 安装基础工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    bash \
    ffmpeg \
    cmake \
    make \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN bash Docker/miniconda_install.sh

COPY extra-req.txt /workspace/GPT-SoVITS/

COPY requirements.txt /workspace/GPT-SoVITS/

COPY install.sh /workspace/GPT-SoVITS/


# 使用清华镜像源安装所有 Python 依赖（必须在 install.sh 之前）
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple line_profiler
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy<2.0"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple scipy
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "librosa==0.10.2"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple numba
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "pytorch-lightning>=2.4"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "gradio<5"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple ffmpeg-python
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "funasr==1.0.27"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple cn2an
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple pypinyin
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "pyopenjtalk>=0.4.1"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple g2p_en

# PyTorch 从官方 CUDA 源安装（CUDA 12.6 for RTX 50 series compatibility）
RUN pip install  torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126

RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "torch-complex==0.4.4"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "torchmetrics==1.4.3"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "modelscope==1.10.0"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple sentencepiece
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.43,<=4.50"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple peft
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple chardet
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple PyYAML
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple psutil
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple jieba_fast
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple jieba
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple split-lang
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "fast_langdetect>=0.3.1"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple wordsegment
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple rotary_embedding_torch
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple ToJyutping
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple g2pk2
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple ko_pron
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-binary opencc opencc
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple python_mecab_ko
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "fastapi[standard]>=0.115.2"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple x_transformers
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "torchmetrics<=1.5"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "pydantic<=2.10.6"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "ctranslate2>=4.0,<5"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface_hub>=0.13"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "tokenizers>=0.13,<1"
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple "av>=11"

# 应用程序额外依赖
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple pymongo
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple motor
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple aiocache
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple aiohttp
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple python-dotenv
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple pyloudnorm
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple py3langid

# 现在运行 install.sh（依赖项已经安装完成）
RUN bash Docker/install_wrapper.sh

# 使用本地 NLTK data（从宿主机复制）
COPY 3rd/nltk_data /root/miniconda3/nltk_data

# Download NLTK data（已改用本地数据，不再下载）
# RUN python3 -m nltk.downloader -d /root/miniconda3/nltk_data \
#     cmudict \
#     averaged_perceptron_tagger \
#     punkt

EXPOSE 8000 3000 4000 27017

ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

# 复制完整代码到容器
COPY . /workspace/GPT-SoVITS

# CMD ["/bin/bash", "-c", "\
#   rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
#   rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
#   rm -rf /workspace/GPT-SoVITS/tools/asr/models && \
#   rm -rf /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
#   ln -s /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
#   ln -s /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
#   ln -s /workspace/models/asr_models /workspace/GPT-SoVITS/tools/asr/models && \
#   ln -s /workspace/models/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
#   exec bash"]
# 复制模型文件到容器内（假设模型文件在宿主机的 models 目录下）
# 请确保以下目录存在于宿主机：
# - models/pretrained_models
# - models/G2PWModel
# - models/asr_models
# - models/uvr5_weights
COPY GPT_SoVITS/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models
COPY GPT_SoVITS/text/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel
COPY tools/asr/models /workspace/GPT-SoVITS/tools/asr/models
COPY tools/uvr5/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights

# 不再需要创建软链接，直接启动bash
CMD ["/bin/bash"]