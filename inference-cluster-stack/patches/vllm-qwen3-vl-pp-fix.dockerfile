FROM vllm/vllm-openai:latest

COPY patches/vllm-qwen3-vl-pp-fix.py /tmp/patch.py
RUN python3 /tmp/patch.py
