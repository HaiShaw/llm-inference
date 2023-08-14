# llm-inference

This repository provides inference utilities including benchmarking tools for large language models served or accelerated through Hugging Face, Deep Speed and Faster Transformer, etc. over common Deep Learning frameworks and Data Center GPUs (AMD: MI300, MI250, MI200, MI100; Nvidia: H100, A100, V100).

### LLM models supported

- [OPT-66B](https://huggingface.co/facebook/opt-66b) https://arxiv.org/abs/2205.01068, verified
- [LLaMa-65B](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) https://arxiv.org/abs/2302.13971, verified
- [Falcon-40B-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct), Falcon-40B later
- [Llama-2](https://huggingface.co/models?search=llama2) https://arxiv.org/abs/2307.09288, 70B yet to be supported by DeepSpeed
- Other OPT, LLaMa, Falcon models of smaller sizes can be extended
- Any Hugging Face models can be easily extended

### Prerequisites:
- [transformers](https://github.com/huggingface/transformers.git)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed.git)
- AMD GPUs:  check github.com/ROCmSoftwarePlatform folks

### Inference benchmarking utilities

- `ibench_hf.py` reports **prefill** latency and **decode** (per token generation) latency to arbitary batch size, prompt (input) size, generation (output) size provided

```python
# prerequisites:
# To enable faster access and loading for models, we expect they stay local:
# - OPT-66B model tokenizer and parameters are prelocated at: /data/opt66b
# - LLaMa-65B model tokenizer and parameters are prelocated at: /data/llama65b
# - Falcon-40B-instruct tokenizer and parameters are prelocated at: /data/falcon40b-instruct
# - LLaMa-2-70B-chat model tokenizer and parameters are prelocated at: /data/llama2-70b-chat
# when adding a new model of big size, you can do the same.

/dockerx/llm-inference# python ibench_hf.py --help
usage: ibench_hf.py [-h] [--model MODEL] [--platform PLATFORM] [--precision PRECISION] [--n N] [--d] [--nocache] [--debug] [--profiling]

LLM Inference Benchmark Example

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         name of LLM (opt66b | llama65b | falcon40b-instruct | llama2-70b-chat | llama2-70b) for inference (default: opt66b)
  --platform PLATFORM   name of DL platform (MI300X | 2xH100 | 2xMI250) for inference (default: MI300X)
  --precision PRECISION
                        model precision and data type (float16 | bfloat16) for inference (default: float16)
  --n N                 number of iterations to inference; report an average of this number of runs (default: 10)
  --d                   use deterministic prompts like: An increasing sequence: -5 -4 -3 -2 -1 0
  --nocache             Disable KV caching (default: on) for transformer inference
  --debug               Print token generations for debugging (default: off)
  --profiling           Enable DeepSpeed Flops Profiler Profiling (default: off)
/dockerx/llm-inference#
```

Examples:

```python
python ibench_hf.py --n 1 --debug
python ibench_hf.py --model llama65b
python ibench_hf.py --model opt66b --n 5
```

### Inference with DeepSpeed Accelerations: Tensor Parallelism and Kernel Injections
- `deepspeed/ibench_ds.py` reports **prefill** latency and **decode** (per token generation) latency to arbitary batch size, prompt (input) size, generation (output) size provided, with DeepSpeed acceleration, with or without Tensor Parallelism, with or without Kernel injections.
- performance benefit from TP is best seen with very fast inter-GPU interconnect (faster than PCI-e): AMD Infinity Fabric Link or Nvidia NVLink
- note: with deepspeed 0.10.x, may need to update OpenAI Triton with `pip install --pre -U triton` or `pip install triton==2.0.0.dev20221120`
- note: with TP `--num_gpus <=` total available GPUs

```python
# prerequisites:
# To enable faster access and loading for models, download and store converted model weights and tokenizer local, provide the path to --name
# For AMD GPUs, install DeepSpeed from https://github.com/ROCmSoftwarePlatform/DeepSpeed -b kernel_injection_UT_enablement

/dockerx/llm-inference# python deepspeed/ibench_ds.py --help
usage: ibench_ds.py [-h] --name NAME [--checkpoint_path CHECKPOINT_PATH] [--save_mp_checkpoint_path SAVE_MP_CHECKPOINT_PATH] [--batch_size BATCH_SIZE] [--dtype {float32,float16,int8}] [--ds_inference] [--use_kernel] [--replace_method REPLACE_METHOD] [--max_tokens MAX_TOKENS]
                    [--prompting_length PROMPTING_LENGTH] [--max_new_tokens MAX_NEW_TOKENS] [--sampling] [--use_meta_tensor] [--performance] [--local_rank LOCAL_RANK] [--world_size WORLD_SIZE] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           model_name
  --checkpoint_path CHECKPOINT_PATH
                        model checkpoint path
  --save_mp_checkpoint_path SAVE_MP_CHECKPOINT_PATH
                        save-path to store the new model checkpoint
  --batch_size BATCH_SIZE
                        batch size
  --dtype {float32,float16,int8}
                        data-type
  --ds_inference        enable ds-inference
  --use_kernel          enable kernel-injection
  --replace_method REPLACE_METHOD
                        replace method['', 'auto']
  --max_tokens MAX_TOKENS
                        maximum tokens used for the text-generation KV-cache
  --prompting_length PROMPTING_LENGTH
                        length of prompts in tokens
  --max_new_tokens MAX_NEW_TOKENS
                        maximum new tokens to generate
  --sampling            sample generation mode
  --use_meta_tensor     use the meta tensors to initialize model
  --performance         enable latency, bandwidth and throughout run
  --local_rank LOCAL_RANK
                        local rank
  --world_size WORLD_SIZE
                        world_size
  --debug               Print token generations for debugging (default: off)
/dockerx/llm-inference#
```

Examples:

```python
deepspeed --num_gpus 1 deepspeed/ibench_ds.py --name /data/llama2-7b --batch_size  8 --prompting_length 512 --performance --ds_inference --max_new_tokens  32
deepspeed --num_gpus 1 deepspeed/ibench_ds.py --name /data/llama2-7b --batch_size 32 --prompting_length 512 --performance --ds_inference --max_new_tokens  64 --use_kernel
deepspeed --num_gpus 4 deepspeed/ibench_ds.py --name /data/llama65b  --batch_size 16 --prompting_length 512 --performance --ds_inference --max_new_tokens  64 --use_kernel
deepspeed --num_gpus 8 deepspeed/ibench_ds.py --name /data/opt66b    --batch_size 32 --prompting_length 512 --performance --ds_inference --max_new_tokens 256 --use_kernel

On AMD GPUs, to speedup DS JIT compilation, you may specify GCN architecture code:
- MI300X: PYTORCH_ROCM_ARCH='gfx940' deepspeed --num_gpus 1 deepspeed/ibench_ds.py --name /data/llama2-7b --batch_size 32 --prompting_length 512 --performance --ds_inference --max_new_tokens 32 --use_kernel
- MI2xx:  PYTORCH_ROCM_ARCH='gfx90a' deepspeed --num_gpus 4 deepspeed/ibench_ds.py --name /data/llama2-7b --batch_size 16 --prompting_length 512 --performance --ds_inference --max_new_tokens 32 --use_kernel
- MI100:  PYTORCH_ROCM_ARCH='gfx908' deepspeed --num_gpus 8 deepspeed/ibench_ds.py --name /data/llama2-7b --batch_size  8 --prompting_length 512 --performance --ds_inference --max_new_tokens 32 --use_kernel
```

### Status

We support multiple GPUs, multiple nodes, and multiple dimensional parallelism, some by implicit software setup, some by explicit argumentation.

Support and harness around following inference infrastructures are working in progress:
- Faster Transformer


### License

MIT. See the [LICENSE](LICENSE) file.
