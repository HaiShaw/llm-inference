from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from multiprocessing import shared_memory
import numpy as np
import deepspeed
import math
import re
import os
import torch
import time
from utils import DSPipeline
from deepspeed.runtime.utils import see_memory_usage

# final deterministic prompts like: "An increasing sequence: -1 0" (for prompt len = 8)
# deterministic prompts input base: len = 5 (with hidden </s>)
inputs = "An increasing sequence:"

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--platform", default="MI250", type=str, required=True, help="node arch")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
parser.add_argument("--replace_method", required=False, default='', type=str, help="replace method['', 'auto']")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--prompting_length", default=1024, type=int, help="length of prompts in tokens")
parser.add_argument("--max_new_tokens", default=512, type=int, help="maximum new tokens to generate")
parser.add_argument("--sampling", action='store_true', help="sample generation mode")
parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
parser.add_argument("--performance", action='store_true', help="enable latency, bandwidth and throughout run")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world_size")
parser.add_argument("--debug", action="store_true", default=False, help="Print token generations for debugging (default: off)")
args = parser.parse_args()

def print_perf_stats(batch_size, latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        if args.dtype == "float16":
            num_bytes = 2
        elif args.dtype == "float32":
            num_bytes = 4
        else:
            num_bytes = 1
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW:    {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * batch_size / 1e12))
        print("\n")

if not args.ds_inference and args.world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)

if args.local_rank == 0:
    see_memory_usage("before init", True)

if args.use_meta_tensor:
    print("use_meta_tensor = True...")

t0 = time.time()
pipe = DSPipeline(model_name=args.name,
                  dtype=data_type,
                  is_meta=args.use_meta_tensor,
                  device=args.local_rank,
                  checkpoint_path=args.checkpoint_path)
if args.local_rank == 0:
    print(f"initialization time: {(time.time()-t0) * 1000}ms")
    see_memory_usage("after init", True)
if args.use_meta_tensor:
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
else:
    ds_kwargs = dict()

if args.ds_inference:
    pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=args.world_size,
                                    replace_with_kernel_inject=args.use_kernel,
                                    replace_method=args.replace_method,
                                    max_tokens=args.max_tokens,
                                    save_mp_checkpoint_path=args.save_mp_checkpoint_path,
                                    **ds_kwargs
                                    )
if args.local_rank == 0:
    see_memory_usage("after init_inference", True)

prompts = []
for b in range(args.batch_size):
    sequence = inputs                                  # len of 5, plus 1 below count for </s>
    neg_nums = (args.prompting_length - (5+1))//2      # number of negative numbers to append
    for num in range(neg_nums, 0, -1):
        sequence = sequence + " -" + str(num)
    sequence = sequence + " 0"
    prompts.append(sequence)

if args.sampling:
    print("Inferencing with sampling...")

iters = 35 if args.performance else 2 #warmup

# Prefill time
prefills = []
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs = pipe(prompts,
            num_tokens=1,
            do_sample=(args.sampling))
    torch.cuda.synchronize()
    end = time.time()
    prefills.append(end - start)

prefill_avg = np.mean(prefills)

# Total generation time
times = []
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs = pipe(prompts,
            num_tokens=args.max_new_tokens,
            do_sample=(args.sampling))
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

time_avg = np.mean(times)
if args.local_rank == 0:
    print(f"\ntotal generation time is {time_avg} sec")

# outputs_shape = .shape[1]
decode_avg = (time_avg - prefill_avg) / (args.max_new_tokens - 1)
if args.local_rank == 0:
    print("Prefill phase latency on prompt of length   : " + str(args.prompting_length) + " = " + "{:.3f}".format(1000 * prefill_avg) + "ms")
    print("Decode latency per token on output of length: " + str(args.max_new_tokens) + " = " + "{:.3f}".format(1000 * decode_avg) + "ms")
    print("Batch size: " + str(args.batch_size))


if args.local_rank == 0:
    if args.debug:
        for i, o in zip(prompts, outputs):
            print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.performance:
        print_perf_stats(args.batch_size, map(lambda t: t / args.max_new_tokens, times), pipe.model.config)


######################################################################################
# Provide opportunity for extra inference runs, without needing to reload the model, #
#        but capable with different batch size, prompt length and generation length. #
######################################################################################

# Name for the shared object
shmem_name = 'shared_config'

if args.local_rank == 0:
    # initial vals in rank0 'no' 'bs' 'ps' 'gs'
    rank0_inputs = np.array([ 0,   0,   0,   0], dtype=np.uint16)
    # create named share memory object
    shmem_object = shared_memory.SharedMemory(name=shmem_name, create=True, size=rank0_inputs.nbytes)
    # create a NumPy array backed by shmem_object
    rank_configs = np.ndarray(rank0_inputs.shape, dtype=rank0_inputs.dtype, buffer=shmem_object.buf)
    # init vals of shmem_object
    rank_configs[:] = rank0_inputs[:]
else:
    # other proc
    time.sleep(1)      # wait for rank0's creation
    shmem_object = shared_memory.SharedMemory(name=shmem_name)
    # create NumPy reference array to shmem_object
    rank_configs = np.ndarray((4,), dtype=np.uint16, buffer=shmem_object.buf)

for i in range(10):
    torch.cuda.synchronize()
    if i==5:
        outputs = pipe(prompts,
                num_tokens=args.max_new_tokens,
                do_sample=(args.sampling),
                tracer=True, platform=args.platform, model=args.name)
    else:
        outputs = pipe(prompts,
            num_tokens=args.max_new_tokens,
            do_sample=(args.sampling))
    torch.cuda.synchronize()

while False:
    if args.local_rank == 0:
        cont = input("\nContinue another inference benchmark run? (yes | no) ")
        if cont.lower() == "no":
            rank_configs[0] = 1  # 1 - 'no'
            time.sleep(3)
            shmem_object.close()
            shmem_object.unlink()
            break
        else:
            rank_configs[0] = 2  # 2 - 'yes'
    else:
        while rank_configs[0] == 0:  # wait
            pass
        if rank_configs[0] == 1:
            shmem_object.close()
            break

    if args.local_rank == 0:
        iconfig_s = input("batch_size (1, 2, ..., 64, 128), prompt_len (8, 16, ..., 512, 1024, 1536, ...), new_tokens (16, 32, ..., 256, 512): ")

        # get inferencing config parameters
        try:
            bs_s, ps_s, gs_s = re.findall('\d+', iconfig_s)
        except ValueError:
            bs_s = '1'
            ps_s = '8'
            gs_s = '8'

        # batch size
        bs = int(bs_s)
        # prompt size
        ps = int(ps_s)
        # generation size
        gs = int(gs_s)

        if ps < 8:
            ps = 8
        if ps % 2:
            ps += 1

        ps_s = str(ps)

        # rank0 update shmem_object
        rank_configs[1] = bs
        rank_configs[2] = ps
        rank_configs[3] = gs

    else:
        # other proc validate last item: /0
        while rank_configs[3] == 0:  # wait
            pass
        if rank_configs[0] == 2:
            bs = rank_configs[1]
            ps = rank_configs[2]
            gs = rank_configs[3]

    if args.local_rank == 0:
        print("Batch size = " + bs_s + ", prompt length = " + ps_s + ", generation length = " + gs_s)

    prompts = []
    for b in range(bs):
        sequence = inputs               # len of 5, plus 1 below count for </s>
        neg_nums = (ps - (5+1))//2      # number of negative numbers to append
        for num in range(neg_nums, 0, -1):
            sequence = sequence + " -" + str(num)
        sequence = sequence + " 0"
        prompts.append(sequence)

    if args.sampling:
        print("Inferencing with sampling...")

    # Prefill time
    prefills = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = pipe(prompts,
                num_tokens=1,
                do_sample=(args.sampling))
        torch.cuda.synchronize()
        end = time.time()
        prefills.append(end - start)

    prefill_avg = np.mean(prefills)

    # Total generation time
    times = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = pipe(prompts,
                num_tokens=gs,
                do_sample=(args.sampling))
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    # reset shmem_object
    rank_configs[0] = 0
    rank_configs[1] = 0
    rank_configs[2] = 0
    rank_configs[3] = 0


    time_avg = np.mean(times)
    if args.local_rank == 0:
        print(f"\ntotal generation time is {time_avg} sec")

    # outputs_shape = .shape[1]
    decode_avg = (time_avg - prefill_avg) / (gs - 1)
    if args.local_rank == 0:
        print("Prefill phase latency on prompt of length   : " + str(ps) + " = " + "{:.3f}".format(1000 * prefill_avg) + "ms")
        print("Decode latency per token on output of length: " + str(gs) + " = " + "{:.3f}".format(1000 * decode_avg) + "ms")
        print("Batch size: " + str(bs))


    if args.local_rank == 0:
        if args.debug:
            for i, o in zip(prompts, outputs):
                print(f"\nin={i}\nout={o}\n{'-'*60}")
        if args.performance:
            print_perf_stats(bs, map(lambda t: t / gs, times), pipe.model.config)

