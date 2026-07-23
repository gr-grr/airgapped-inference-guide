# Fix PR #43272 -- Qwen3-VL(-MoE): pass architectures to with_hf_config for pipeline parallelism
# Without this, PP>1 crashes with "No model architectures are specified"
# because config.text_config.architectures is None (architectures only exist on top-level config).
# https://github.com/vllm-project/vllm/pull/43272

import re

# Fix qwen3_vl_moe.py
path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl_moe.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    'vllm_config=vllm_config.with_hf_config(config.text_config),',
    'vllm_config=vllm_config.with_hf_config(\n        config.text_config, architectures=["Qwen3MoeForCausalLM"]),'
)
with open(path, 'w') as f:
    f.write(content)
print('Patched qwen3_vl_moe.py')

# Fix qwen3_vl.py
path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    'vllm_config=vllm_config.with_hf_config(config.text_config),',
    'vllm_config=vllm_config.with_hf_config(\n        config.text_config, architectures=["Qwen3ForCausalLM"]),'
)
with open(path, 'w') as f:
    f.write(content)
print('Patched qwen3_vl.py')
