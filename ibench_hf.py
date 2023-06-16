import re
import sys
import torch
import argparse
import numpy as np
from time import perf_counter

# vocab used for input sequences
iVOCAB = ['Neural', 'networks', 'learn', 'or', 'are', 'trained', 'by', 'processing', 'examples,', 'each', 'of', 'which', 'contains', 'a', 'known', 'input', 'and', 'result', 'forming', 'probability', 'weighted', 'associations', 'between', 'the', 'two', 'which', 'are', 'stored', 'within', 'the', 'data', 'structure', 'of', 'the', 'net', 'itself.', 'The', 'training', 'of', 'a', 'neural', 'network', 'from', 'a', 'given', 'example', 'is', 'usually', 'conducted', 'by', 'determining', 'the', 'difference', 'between', 'processed', 'output', 'of', 'the', 'network', 'often', 'prediction', 'and', 'a', 'target', 'output', 'This', 'difference', 'is', 'the', 'error', 'The', 'network', 'then', 'adjusts', 'its', 'weighted', 'associations', 'according', 'to', 'a', 'learning', 'rule', 'and', 'using', 'this', 'error', 'value', 'Successive', 'adjustments', 'will', 'cause', 'the', 'neural', 'network', 'to', 'produce', 'output', 'that', 'is', 'increasingly', 'similar', 'to', 'the', 'target', 'output', 'After', 'a', 'sufficient', 'number', 'of', 'these', 'adjustments', 'the', 'training', 'can', 'be', 'terminated', 'based', 'on', 'certain', 'criteria', 'This', 'is', 'a', 'form', 'of', 'supervised', 'learning', 'Such', 'systems', 'learn', 'to', 'perform', 'tasks', 'by', 'considering', 'examples', 'generally', 'without', 'being', 'programmed', 'with', 'task', 'specific', 'rules', 'For', 'example', 'in', 'image', 'recognition', 'they', 'might', 'learn', 'to', 'identify', 'images', 'that', 'contain', 'cats', 'by', 'analyzing', 'example', 'images', 'that', 'have', 'been', 'manually', 'labeled', 'as', 'cat', 'or', 'no', 'cat', 'and', 'using', 'the', 'results', 'to', 'identify', 'cats', 'in', 'other', 'images', 'They', 'do', 'this', 'without', 'any', 'prior', 'knowledge', 'of', 'cats', 'for', 'example', 'that', 'they', 'have', 'fur', 'tails', 'whiskers', 'and', 'cat', 'like', 'faces', 'Instead', 'they', 'automatically', 'generate', 'identifying', 'characteristics', 'from', 'the', 'examples', 'that', 'they', 'process', ',', '.', '?']

# input tokens (to build randomized input sequences)
itokens = np.array(iVOCAB)

# final deterministic prompts like: "An increasing sequence: -1 0" (for prompt len = 8)
# deterministic prompts input base: len = 5 (with hidden </s>)
inputs = "An increasing sequence:"

# path to model files: tokenizer and weights
PATH1='/data/opt66b'
PATH2='/data/llama65b'
PATH3='/data/falcon40b-instruct'

def main():

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch minGPT Example")
    parser.add_argument(
        "--model",
        type=str,
        default="opt66b",
        help="name of LLM (opt66b | llama65b | falcon40b-instruct) for inference (default: opt66b)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        metavar="N",
        help="number of iterations to inference; report an average of this number of runs (default: 10)",
    )
    parser.add_argument(
            "--d", action="store_true", default=False, help="use deterministic prompts like: An increasing sequence: -5 -4 -3 -2 -1 0"
    )
    parser.add_argument(
        "--nocache", action="store_true", default=False, help="Disable KV caching (default: on) for transformer inference"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print token generations for debugging (default: off)"
    )
    args = parser.parse_args()

    if args.model == "opt66b":
        PATH = PATH1
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        tokenizer = AutoTokenizer.from_pretrained(PATH, use_fast=False, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
    elif args.model == "llama65b":
        PATH = PATH2
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
    elif args.model == "falcon40b-instruct":
        PATH = PATH3
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        sys.exit("Enter valid --model (opt66b | llama65b | falcon40b-instruct)")

    print("Model " + args.model + " loaded.")

    if args.n <= 0:
        args.n = 10
    print("Benchmark to report is an average of " + str(args.n) + " runs ....\n")

    while True:
        p_latencies = []
        d_latencies = []

        iconfig_s = input("batch_size (1, 2, ..., 64, 128), prompt_len (8, 16, ..., 512, 1024), new_tokens (16, 32, ..., 256, 512): ")

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

        print("Batch size = " + bs_s + ", prompt length = " + ps_s + ", generation length = " + gs_s)

        for i in range(1+args.n):
            prompts = []
            for b in range(bs):
                if args.d:
                    sequence = inputs               # len of 5
                    neg_nums = (ps - (5+1))//2      # number of negative numbers to append
                    for num in range(neg_nums, 0, -1):
                        sequence = sequence + " -" + str(num)
                    sequence = sequence + " 0"
                else:
                    prompt = []
                    for t in range(ps):
                        token = np.random.choice(itokens)
                        prompt.append(token)
                    # a sample sequence
                    sequence = " ".join(prompt)
                prompts.append(sequence)

            # input_ids = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).input_ids.cuda()
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()

            if args.nocache:
                start_time = perf_counter()
                generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1, use_cache=False)
                prefill_latency = perf_counter() - start_time
            else:
                if args.d:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, max_new_tokens=1)
                    prefill_latency = perf_counter() - start_time
                else:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1)
                    prefill_latency = perf_counter() - start_time

            # ignore the 1st - warmup
            if i != 0:
                p_latencies.append(prefill_latency)

            if args.nocache:
                start_time = perf_counter()
                generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1), use_cache=False)
                decode_latency = perf_counter() - start_time
            else:
                if args.d:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, max_new_tokens=(gs+1))
                    decode_latency = perf_counter() - start_time
                else:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1))
                    decode_latency = perf_counter() - start_time

            # ignore the 1st - warmup
            if i != 0:
                d_latencies.append(decode_latency)

        # show generations for the last iteration
        if args.debug:
            print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))

        # adjust
        Ptime = np.array(p_latencies).mean()
        Dtime = (np.array(d_latencies).mean() - Ptime)/gs
        Ptime -= Dtime

        # reports
        print("\n")
        print("Prefill phase latency on prompt of length   : " + ps_s + " = " + "{:.3f}".format(1000 * Ptime) + "ms")
        print("Decode latency per token on output of length: " + gs_s + " = " + "{:.3f}".format(1000 * Dtime) + "ms")


        cont = input("\nContinue another inference benchmark run? (yes | no) ")
        if cont.lower() == "no":
            break


if __name__=="__main__":
    main()
