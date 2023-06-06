import re
import sys
import torch
import argparse
import numpy as np
from time import perf_counter
# from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
# from transformers import LlamaForCausalLM, LlamaTokenizer

# vocab used for input sequences
iVOCAB = ['Neural', 'networks', 'learn', 'or', 'are', 'trained', 'by', 'processing', 'examples,', 'each', 'of', 'which', 'contains', 'a', 'known', 'input', 'and', 'result', 'forming', 'probability', 'weighted', 'associations', 'between', 'the', 'two', 'which', 'are', 'stored', 'within', 'the', 'data', 'structure', 'of', 'the', 'net', 'itself.', 'The', 'training', 'of', 'a', 'neural', 'network', 'from', 'a', 'given', 'example', 'is', 'usually', 'conducted', 'by', 'determining', 'the', 'difference', 'between', 'processed', 'output', 'of', 'the', 'network', 'often', 'prediction', 'and', 'a', 'target', 'output', 'This', 'difference', 'is', 'the', 'error', 'The', 'network', 'then', 'adjusts', 'its', 'weighted', 'associations', 'according', 'to', 'a', 'learning', 'rule', 'and', 'using', 'this', 'error', 'value', 'Successive', 'adjustments', 'will', 'cause', 'the', 'neural', 'network', 'to', 'produce', 'output', 'that', 'is', 'increasingly', 'similar', 'to', 'the', 'target', 'output', 'After', 'a', 'sufficient', 'number', 'of', 'these', 'adjustments', 'the', 'training', 'can', 'be', 'terminated', 'based', 'on', 'certain', 'criteria', 'This', 'is', 'a', 'form', 'of', 'supervised', 'learning', 'Such', 'systems', 'learn', 'to', 'perform', 'tasks', 'by', 'considering', 'examples', 'generally', 'without', 'being', 'programmed', 'with', 'task', 'specific', 'rules', 'For', 'example', 'in', 'image', 'recognition', 'they', 'might', 'learn', 'to', 'identify', 'images', 'that', 'contain', 'cats', 'by', 'analyzing', 'example', 'images', 'that', 'have', 'been', 'manually', 'labeled', 'as', 'cat', 'or', 'no', 'cat', 'and', 'using', 'the', 'results', 'to', 'identify', 'cats', 'in', 'other', 'images', 'They', 'do', 'this', 'without', 'any', 'prior', 'knowledge', 'of', 'cats', 'for', 'example', 'that', 'they', 'have', 'fur', 'tails', 'whiskers', 'and', 'cat', 'like', 'faces', 'Instead', 'they', 'automatically', 'generate', 'identifying', 'characteristics', 'from', 'the', 'examples', 'that', 'they', 'process', ',', '.', '?']

# input tokens (to build randomized input sequences)
itokens = np.array(iVOCAB)

PATH1='/data/opt66b'
PATH2='/data/llama65b'

#prompt = "Hello I am conscious and"
#prompts = ["what be why one this", "Hello I am conscious and"]
#prompts = []

def main():

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch minGPT Example")
    parser.add_argument(
        "--model",
        type=str,
        default="opt66b",
        help="name of LLM model for inference (default: opt66b)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        metavar="N",
        help="number of iterations to inference; report an average of this number of runs (default: 8)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Print token generations for debugging"
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
        tokenizer = LlamaTokenizer.from_pretrained(PATH)
        model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
    else:
        sys.exit("Enter valid --model (opt66b | llama65b)")

    print("Model loaded.")

    # prompt
    # sample = []

    while True:
        p_latencies = []
        d_latencies = []

        prompts = []

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

        print("Batch size = " + bs_s + ", prompt length = " + ps_s + ", generation length = " + gs_s)

        for i in range(1+args.n):
            for b in range(bs):
                prompt = []
                for t in range(ps):
                    token = np.random.choice(itokens)
                    prompt.append(token)
                # a sample sequence
                sequence = " ".join(prompt)
                prompts.append(sequence)

            # input_ids = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).input_ids.cuda()
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()

            start_time = perf_counter()
            generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1)
            prefill_latency = perf_counter() - start_time

            # ignore the 1st - warmup
            if i != 0:
                p_latencies.append(prefill_latency)

            start_time = perf_counter()
            generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1))
            decode_latency_per_token = (perf_counter() - start_time) / (gs+1)

            # ignore the 1st - warmup
            if i != 0:
                d_latencies.append(decode_latency_per_token)

            # fixup prefill (to cancel 1 token generation included), this fixup can cause Neg values!
            # prefill_latency = prefill_latency - decode_latency_per_token

        # show generations for the last iteration
        if args.debug:
            print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))

        # reports
        print("\n")
        print("Prefill phase latency on prompt of length   : " + ps_s + " = " + "{:.3f}".format(1000 * np.array(p_latencies).mean()) + "ms")
        print("Decode latency per token on output of length: " + gs_s + " = " + "{:.3f}".format(1000 * np.array(d_latencies).mean()) + "ms")


        cont = input("\nContinue another inference benchmark run? (yes | no) ")
        if cont.lower() == "no":
            break


if __name__=="__main__":
    main()
