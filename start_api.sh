#!/bin/sh
# 启动 api_v2（127.0.0.1:9880），读取 GPT_SoVITS/configs/tts_infer.yaml。
# 本机回环请求不应经过系统代理，因此清除代理相关环境变量。
cd "$(dirname "$0")"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy='*' NO_PROXY='*'
exec .venv/bin/python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml "$@"
