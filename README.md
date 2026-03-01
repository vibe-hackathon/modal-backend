Modal + vLLM: OpenAI-Compatible Inference Server for MiniMax-M2.5

Deploys the MiniMaxAI/MiniMax-M2.5 model on Modal using vLLM,
exposing a standard OpenAI-compatible REST API (including /v1/chat/completions).

Deployment:
```bash
modal deploy modal_vllm_server.py
```

Dev/debug run (ephemeral):
```bash
modal serve modal_vllm_server.py
```

Once deployed, Modal will print a URL like:
`https://<your-workspace>--minimax-vllm-openai-server-serve.modal.run`

You can then call it exactly like the OpenAI API:
```bash
curl https://zhongyi070622--minimax-vllm-openai-server-serve.modal.run/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
    }'
```