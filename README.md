# llm-inference

This repository provides inference utilities including benchmarking tools for large language models served or accelerated through Hugging Face, Deep Speed and Faster Transformer, etc. over common Deep Learning frameworks and Data Center GPUs (AMD: MI300, MI250, MI200, MI100; Nvidia: H100, A100, V100).

### LLM models supported

- [OPT-66B](https://huggingface.co/facebook/opt-66b) https://arxiv.org/abs/2205.01068, verified
- [LLaMa-65B](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) https://arxiv.org/abs/2302.13971, verified
- [Falcon-40B-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct), Falcon-40B later
- Other OPT, LLaMa, Falcon models of smaller sizes can be extended
- Any Hugging Face models can be easily extended

### Inference benchmarking utilities

- `ibench_hf.py` reports **prefill** latency and **decode** (per token generation) latency to arbitary batch size, prompt (input) size, generation (output) size provided

```python
# prerequisites:
# To enable faster access and loading for models, we expect they stay local:
# - OPT-66B model tokenizer and parameters are prelocated at: /data/opt66b
# - LLaMa-65B model tokenizer and parameters are prelocated at: /data/llama65b
# - Falcon-40B-instruct tokenizer and parameters are prelocated at: /data/falcon40b-instruct
# when adding a new model of big size, you can do the same.

/dockerx/llm-inference# python ibench_hf.py --help
usage: ibench_hf.py [-h] [--model MODEL] [--n N] [--nocache] [--debug]

PyTorch minGPT Example

options:
  -h, --help     show this help message and exit
  --model MODEL  name of LLM (opt66b | llama65b | falcon40b-instruct) for inference (default: opt66b)
  --n N          number of iterations to inference; report an average of this number of runs (default: 8)
  --nocache      Disable KV caching (default: on) for transformer inference
  --debug        Print token generations for debugging (default: off)
/dockerx/llm-inference# 
```

Examples:

```python
python ibench_hf.py --n 1 --debug
python ibench_hf.py --model llama65b
python ibench_hf.py --model opt66b --n 5
```


### Status

We support multiple GPUs, multiple nodes, and multiple dimensional parallelism, some by implicit software setup, some by explicit argumentation.

Support and harness around following inference infrastructures are working in progress:
- Deep Speed
- Faster Transformer


### License

MIT. See the [LICENSE](LICENSE) file.
