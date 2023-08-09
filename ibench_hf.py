import re
import sys
import torch
import argparse
import numpy as np
from time import perf_counter
from deepspeed.profiling.flops_profiler import FlopsProfiler

# vocab used for input sequences
iVOCAB = ['Neural', 'networks', 'learn', 'or', 'are', 'trained', 'by', 'processing', 'examples,', 'each', 'of', 'which', 'contains', 'a', 'known', 'input', 'and', 'result', 'forming', 'probability', 'weighted', 'associations', 'between', 'the', 'two', 'which', 'are', 'stored', 'within', 'the', 'data', 'structure', 'of', 'the', 'net', 'itself.', 'The', 'training', 'of', 'a', 'neural', 'network', 'from', 'a', 'given', 'example', 'is', 'usually', 'conducted', 'by', 'determining', 'the', 'difference', 'between', 'processed', 'output', 'of', 'the', 'network', 'often', 'prediction', 'and', 'a', 'target', 'output', 'This', 'difference', 'is', 'the', 'error', 'The', 'network', 'then', 'adjusts', 'its', 'weighted', 'associations', 'according', 'to', 'a', 'learning', 'rule', 'and', 'using', 'this', 'error', 'value', 'Successive', 'adjustments', 'will', 'cause', 'the', 'neural', 'network', 'to', 'produce', 'output', 'that', 'is', 'increasingly', 'similar', 'to', 'the', 'target', 'output', 'After', 'a', 'sufficient', 'number', 'of', 'these', 'adjustments', 'the', 'training', 'can', 'be', 'terminated', 'based', 'on', 'certain', 'criteria', 'This', 'is', 'a', 'form', 'of', 'supervised', 'learning', 'Such', 'systems', 'learn', 'to', 'perform', 'tasks', 'by', 'considering', 'examples', 'generally', 'without', 'being', 'programmed', 'with', 'task', 'specific', 'rules', 'For', 'example', 'in', 'image', 'recognition', 'they', 'might', 'learn', 'to', 'identify', 'images', 'that', 'contain', 'cats', 'by', 'analyzing', 'example', 'images', 'that', 'have', 'been', 'manually', 'labeled', 'as', 'cat', 'or', 'no', 'cat', 'and', 'using', 'the', 'results', 'to', 'identify', 'cats', 'in', 'other', 'images', 'They', 'do', 'this', 'without', 'any', 'prior', 'knowledge', 'of', 'cats', 'for', 'example', 'that', 'they', 'have', 'fur', 'tails', 'whiskers', 'and', 'cat', 'like', 'faces', 'Instead', 'they', 'automatically', 'generate', 'identifying', 'characteristics', 'from', 'the', 'examples', 'that', 'they', 'process', ',', '.', '?']

# input tokens (to build randomized input sequences)
itokens = np.array(iVOCAB)

# final deterministic prompts like: "An increasing sequence: -1 0" (for prompt len = 8)
# deterministic prompts input base: len = 5 (with hidden </s>)
inputs = "An increasing sequence:"

# device map (model layers to device mapping) for MI300X, all fit in Single GPU: 0
DM_MI300X_opt66b={'': 0}
DM_MI300X_llama65b={'': 0}
DM_MI300X_falcon40b={'': 0}
DM_MI300X_llamaII70b={'': 0}

# device map (model layers to device mapping) for 2xMI250
DM_2xMI250_opt66b={'model.decoder.embed_tokens': 0, 'lm_head': 0, 'model.decoder.embed_positions': 0, 'model.decoder.final_layer_norm': 0, 'model.decoder.layers.0': 0, 'model.decoder.layers.1': 0, 'model.decoder.layers.2': 0, 'model.decoder.layers.3': 0, 'model.decoder.layers.4': 0, 'model.decoder.layers.5': 0, 'model.decoder.layers.6': 0, 'model.decoder.layers.7': 0, 'model.decoder.layers.8': 0, 'model.decoder.layers.9': 0, 'model.decoder.layers.10': 0, 'model.decoder.layers.11': 0, 'model.decoder.layers.12': 0, 'model.decoder.layers.13': 0, 'model.decoder.layers.14': 0, 'model.decoder.layers.15': 0, 'model.decoder.layers.16': 1, 'model.decoder.layers.17': 1, 'model.decoder.layers.18': 1, 'model.decoder.layers.19': 1, 'model.decoder.layers.20': 1, 'model.decoder.layers.21': 1, 'model.decoder.layers.22': 1, 'model.decoder.layers.23': 1, 'model.decoder.layers.24': 1, 'model.decoder.layers.25': 1, 'model.decoder.layers.26': 1, 'model.decoder.layers.27': 1, 'model.decoder.layers.28': 1, 'model.decoder.layers.29': 1, 'model.decoder.layers.30': 1, 'model.decoder.layers.31': 1, 'model.decoder.layers.32': 1, 'model.decoder.layers.33': 2, 'model.decoder.layers.34': 2, 'model.decoder.layers.35': 2, 'model.decoder.layers.36': 2, 'model.decoder.layers.37': 2, 'model.decoder.layers.38': 2, 'model.decoder.layers.39': 2, 'model.decoder.layers.40': 2, 'model.decoder.layers.41': 2, 'model.decoder.layers.42': 2, 'model.decoder.layers.43': 2, 'model.decoder.layers.44': 2, 'model.decoder.layers.45': 2, 'model.decoder.layers.46': 2, 'model.decoder.layers.47': 2, 'model.decoder.layers.48': 2, 'model.decoder.layers.49': 2, 'model.decoder.layers.50': 3, 'model.decoder.layers.51': 3, 'model.decoder.layers.52': 3, 'model.decoder.layers.53': 3, 'model.decoder.layers.54': 3, 'model.decoder.layers.55': 3, 'model.decoder.layers.56': 3, 'model.decoder.layers.57': 3, 'model.decoder.layers.58': 3, 'model.decoder.layers.59': 3, 'model.decoder.layers.60': 3, 'model.decoder.layers.61': 3, 'model.decoder.layers.62': 3, 'model.decoder.layers.63': 3}

DM_2xMI250_llama65b={'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 'model.layers.40': 1, 'model.layers.41': 2, 'model.layers.42': 2, 'model.layers.43': 2, 'model.layers.44': 2, 'model.layers.45': 2, 'model.layers.46': 2, 'model.layers.47': 2, 'model.layers.48': 2, 'model.layers.49': 2, 'model.layers.50': 2, 'model.layers.51': 2, 'model.layers.52': 2, 'model.layers.53': 2, 'model.layers.54': 2, 'model.layers.55': 2, 'model.layers.56': 2, 'model.layers.57': 2, 'model.layers.58': 2, 'model.layers.59': 2, 'model.layers.60': 2, 'model.layers.61': 2, 'model.layers.62': 3, 'model.layers.63': 3, 'model.layers.64': 3, 'model.layers.65': 3, 'model.layers.66': 3, 'model.layers.67': 3, 'model.layers.68': 3, 'model.layers.69': 3, 'model.layers.70': 3, 'model.layers.71': 3, 'model.layers.72': 3, 'model.layers.73': 3, 'model.layers.74': 3, 'model.layers.75': 3, 'model.layers.76': 3, 'model.layers.77': 3, 'model.layers.78': 3, 'model.layers.79': 3, 'model.norm': 3, 'lm_head': 3}

DM_2xMI250_falcon40b={'transformer.word_embeddings': 0, 'lm_head': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 1, 'transformer.h.15': 1, 'transformer.h.16': 1, 'transformer.h.17': 1, 'transformer.h.18': 1, 'transformer.h.19': 1, 'transformer.h.20': 1, 'transformer.h.21': 1, 'transformer.h.22': 1, 'transformer.h.23': 1, 'transformer.h.24': 1, 'transformer.h.25': 1, 'transformer.h.26': 1, 'transformer.h.27': 1, 'transformer.h.28': 1, 'transformer.h.29': 1, 'transformer.h.30': 2, 'transformer.h.31': 2, 'transformer.h.32': 2, 'transformer.h.33': 2, 'transformer.h.34': 2, 'transformer.h.35': 2, 'transformer.h.36': 2, 'transformer.h.37': 2, 'transformer.h.38': 2, 'transformer.h.39': 2, 'transformer.h.40': 2, 'transformer.h.41': 2, 'transformer.h.42': 2, 'transformer.h.43': 2, 'transformer.h.44': 2, 'transformer.h.45': 2, 'transformer.h.46': 3, 'transformer.h.47': 3, 'transformer.h.48': 3, 'transformer.h.49': 3, 'transformer.h.50': 3, 'transformer.h.51': 3, 'transformer.h.52': 3, 'transformer.h.53': 3, 'transformer.h.54': 3, 'transformer.h.55': 3, 'transformer.h.56': 3, 'transformer.h.57': 3, 'transformer.h.58': 3, 'transformer.h.59': 3, 'transformer.ln_f': 3}

DM_2xMI250_llamaII70b={'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 'model.layers.40': 1, 'model.layers.41': 2, 'model.layers.42': 2, 'model.layers.43': 2, 'model.layers.44': 2, 'model.layers.45': 2, 'model.layers.46': 2, 'model.layers.47': 2, 'model.layers.48': 2, 'model.layers.49': 2, 'model.layers.50': 2, 'model.layers.51': 2, 'model.layers.52': 2, 'model.layers.53': 2, 'model.layers.54': 2, 'model.layers.55': 2, 'model.layers.56': 2, 'model.layers.57': 2, 'model.layers.58': 2, 'model.layers.59': 2, 'model.layers.60': 2, 'model.layers.61': 2, 'model.layers.62': 3, 'model.layers.63': 3, 'model.layers.64': 3, 'model.layers.65': 3, 'model.layers.66': 3, 'model.layers.67': 3, 'model.layers.68': 3, 'model.layers.69': 3, 'model.layers.70': 3, 'model.layers.71': 3, 'model.layers.72': 3, 'model.layers.73': 3, 'model.layers.74': 3, 'model.layers.75': 3, 'model.layers.76': 3, 'model.layers.77': 3, 'model.layers.78': 3, 'model.layers.79': 3, 'model.norm': 3, 'lm_head': 3}

# device map (model layers to device mapping) for 2xH100
DM_2xH100_opt66b={'model.decoder.embed_tokens': 0, 'lm_head': 0, 'model.decoder.embed_positions': 0, 'model.decoder.final_layer_norm': 0, 'model.decoder.layers.0': 0, 'model.decoder.layers.1': 0, 'model.decoder.layers.2': 0, 'model.decoder.layers.3': 0, 'model.decoder.layers.4': 0, 'model.decoder.layers.5': 0, 'model.decoder.layers.6': 0, 'model.decoder.layers.7': 0, 'model.decoder.layers.8': 0, 'model.decoder.layers.9': 0, 'model.decoder.layers.10': 0, 'model.decoder.layers.11': 0, 'model.decoder.layers.12': 0, 'model.decoder.layers.13': 0, 'model.decoder.layers.14': 0, 'model.decoder.layers.15': 0, 'model.decoder.layers.16': 0, 'model.decoder.layers.17': 0, 'model.decoder.layers.18': 0, 'model.decoder.layers.19': 0, 'model.decoder.layers.20': 0, 'model.decoder.layers.21': 0, 'model.decoder.layers.22': 0, 'model.decoder.layers.23': 0, 'model.decoder.layers.24': 0, 'model.decoder.layers.25': 0, 'model.decoder.layers.26': 0, 'model.decoder.layers.27': 0, 'model.decoder.layers.28': 0, 'model.decoder.layers.29': 0, 'model.decoder.layers.30': 0, 'model.decoder.layers.31': 0, 'model.decoder.layers.32': 1, 'model.decoder.layers.33': 1, 'model.decoder.layers.34': 1, 'model.decoder.layers.35': 1, 'model.decoder.layers.36': 1, 'model.decoder.layers.37': 1, 'model.decoder.layers.38': 1, 'model.decoder.layers.39': 1, 'model.decoder.layers.40': 1, 'model.decoder.layers.41': 1, 'model.decoder.layers.42': 1, 'model.decoder.layers.43': 1, 'model.decoder.layers.44': 1, 'model.decoder.layers.45': 1, 'model.decoder.layers.46': 1, 'model.decoder.layers.47': 1, 'model.decoder.layers.48': 1, 'model.decoder.layers.49': 1, 'model.decoder.layers.50': 1, 'model.decoder.layers.51': 1, 'model.decoder.layers.52': 1, 'model.decoder.layers.53': 1, 'model.decoder.layers.54': 1, 'model.decoder.layers.55': 1, 'model.decoder.layers.56': 1, 'model.decoder.layers.57': 1, 'model.decoder.layers.58': 1, 'model.decoder.layers.59': 1, 'model.decoder.layers.60': 1, 'model.decoder.layers.61': 1, 'model.decoder.layers.62': 1, 'model.decoder.layers.63': 1}

DM_2xH100_llama65b={'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0, 'model.layers.32': 0, 'model.layers.33': 0, 'model.layers.34': 0, 'model.layers.35': 0, 'model.layers.36': 0, 'model.layers.37': 0, 'model.layers.38': 0, 'model.layers.39': 0, 'model.layers.40': 1, 'model.layers.41': 1, 'model.layers.42': 1, 'model.layers.43': 1, 'model.layers.44': 1, 'model.layers.45': 1, 'model.layers.46': 1, 'model.layers.47': 1, 'model.layers.48': 1, 'model.layers.49': 1, 'model.layers.50': 1, 'model.layers.51': 1, 'model.layers.52': 1, 'model.layers.53': 1, 'model.layers.54': 1, 'model.layers.55': 1, 'model.layers.56': 1, 'model.layers.57': 1, 'model.layers.58': 1, 'model.layers.59': 1, 'model.layers.60': 1, 'model.layers.61': 1, 'model.layers.62': 1, 'model.layers.63': 1, 'model.layers.64': 1, 'model.layers.65': 1, 'model.layers.66': 1, 'model.layers.67': 1, 'model.layers.68': 1, 'model.layers.69': 1, 'model.layers.70': 1, 'model.layers.71': 1, 'model.layers.72': 1, 'model.layers.73': 1, 'model.layers.74': 1, 'model.layers.75': 1, 'model.layers.76': 1, 'model.layers.77': 1, 'model.layers.78': 1, 'model.layers.79': 1, 'model.norm': 1, 'lm_head': 1}

DM_2xH100_falcon40b={'transformer.word_embeddings': 0, 'lm_head': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 0, 'transformer.h.15': 0, 'transformer.h.16': 0, 'transformer.h.17': 0, 'transformer.h.18': 0, 'transformer.h.19': 0, 'transformer.h.20': 0, 'transformer.h.21': 0, 'transformer.h.22': 0, 'transformer.h.23': 0, 'transformer.h.24': 0, 'transformer.h.25': 0, 'transformer.h.26': 0, 'transformer.h.27': 0, 'transformer.h.28': 0, 'transformer.h.29': 0, 'transformer.h.30': 1, 'transformer.h.31': 1, 'transformer.h.32': 1, 'transformer.h.33': 1, 'transformer.h.34': 1, 'transformer.h.35': 1, 'transformer.h.36': 1, 'transformer.h.37': 1, 'transformer.h.38': 1, 'transformer.h.39': 1, 'transformer.h.40': 1, 'transformer.h.41': 1, 'transformer.h.42': 1, 'transformer.h.43': 1, 'transformer.h.44': 1, 'transformer.h.45': 1, 'transformer.h.46': 1, 'transformer.h.47': 1, 'transformer.h.48': 1, 'transformer.h.49': 1, 'transformer.h.50': 1, 'transformer.h.51': 1, 'transformer.h.52': 1, 'transformer.h.53': 1, 'transformer.h.54': 1, 'transformer.h.55': 1, 'transformer.h.56': 1, 'transformer.h.57': 1, 'transformer.h.58': 1, 'transformer.h.59': 1, 'transformer.ln_f': 1}

DM_2xH100_llamaII70b={'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0, 'model.layers.32': 0, 'model.layers.33': 0, 'model.layers.34': 0, 'model.layers.35': 0, 'model.layers.36': 0, 'model.layers.37': 0, 'model.layers.38': 0, 'model.layers.39': 0, 'model.layers.40': 1, 'model.layers.41': 1, 'model.layers.42': 1, 'model.layers.43': 1, 'model.layers.44': 1, 'model.layers.45': 1, 'model.layers.46': 1, 'model.layers.47': 1, 'model.layers.48': 1, 'model.layers.49': 1, 'model.layers.50': 1, 'model.layers.51': 1, 'model.layers.52': 1, 'model.layers.53': 1, 'model.layers.54': 1, 'model.layers.55': 1, 'model.layers.56': 1, 'model.layers.57': 1, 'model.layers.58': 1, 'model.layers.59': 1, 'model.layers.60': 1, 'model.layers.61': 1, 'model.layers.62': 1, 'model.layers.63': 1, 'model.layers.64': 1, 'model.layers.65': 1, 'model.layers.66': 1, 'model.layers.67': 1, 'model.layers.68': 1, 'model.layers.69': 1, 'model.layers.70': 1, 'model.layers.71': 1, 'model.layers.72': 1, 'model.layers.73': 1, 'model.layers.74': 1, 'model.layers.75': 1, 'model.layers.76': 1, 'model.layers.77': 1, 'model.layers.78': 1, 'model.layers.79': 1, 'model.norm': 1, 'lm_head': 1}

# path to model files: tokenizer and weights
PATH1='/data/opt66b'
PATH2='/data/llama65b'
PATH3='/data/falcon40b-instruct'
PATH4='/data/llama2-70b-chat'
PATH5='/data/llama2-70b'

def main():

    # Training settings
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Example")
    parser.add_argument(
        "--model",
        type=str,
        default="opt66b",
        help="name of LLM (opt66b | llama65b | falcon40b-instruct | llama2-70b-chat | llama2-70b) for inference (default: opt66b)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="MI300X",
        help="name of DL platform (MI300X | 2xH100 | 2xMI250) for inference (default: MI300X)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        help="model precision and data type (float16 | bfloat16) for inference (default: float16)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        metavar="N",
        help="number of iterations to inference; report an average of this number of runs (default: 10)"
    )
    parser.add_argument(
        "--d",
        action="store_true",
        default=False,
        help="use deterministic prompts like: An increasing sequence: -5 -4 -3 -2 -1 0"
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        default=False,
        help="Disable KV caching (default: on) for transformer inference"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print token generations for debugging (default: off)"
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help="Enable DeepSpeed Flops Profiler Profiling (default: off)"
    )
    args = parser.parse_args()

    if args.model == "opt66b":
        PATH = PATH1
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        tokenizer = AutoTokenizer.from_pretrained(PATH, use_fast=False, padding_side='left')
        # tokenizer = AutoTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        if args.precision == "float16":     # pretrained precision : float16
            model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
        elif args.precision == "bfloat16":
            if args.platform == "MI300X":
                model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_MI300X_opt66b)
            elif args.platform == "2xH100":
                model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xH100_opt66b)
            elif args.platform == "2xMI250":
                 model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xMI250_opt66b)
            else:
                sys.exit("Enter valid --platform (MI300X | 2xH100)")
        else:
            sys.exit("Enter valid --precision (float16 | bfloat16)")

    elif args.model == "llama65b":
        PATH = PATH2
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        if args.precision == "float16": # pretrained precision : float16
            model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
        elif args.precision == "bfloat16":
            if args.platform == "MI300X":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_MI300X_llama65b)
            elif args.platform == "2xH100":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xH100_llama65b)
            elif args.platform == "2xMI250":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xMI250_llama65b)
            else:
                sys.exit("Enter valid --platform (MI300X | 2xH100)")
        else:
            sys.exit("Enter valid --precision (float16 | bfloat16)")

    elif args.model == "falcon40b-instruct":
        PATH = PATH3
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        if args.precision == "bfloat16":    # pretrained precision : bfloat16
            model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        elif args.precision == "float16":
            if args.platform == "MI300X":
                model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map=DM_MI300X_falcon40b)
            elif args.platform == "2xH100":
                model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map=DM_2xH100_falcon40b)
            elif args.platform == "2xMI250":
                model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map=DM_2xMI250_falcon40b)
            else:
                sys.exit("Enter valid --platform (MI300X | 2xH100)")
        else:
            sys.exit("Enter valid --precision (float16 | bfloat16)")

    elif args.model == "llama2-70b-chat":
        PATH = PATH4
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        if args.precision == "float16": # pretrained precision : float16
            model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
        elif args.precision == "bfloat16":
            if args.platform == "MI300X":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_MI300X_llamaII70b)
            elif args.platform == "2xH100":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xH100_llamaII70b)
            elif args.platform == "2xMI250":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xMI250_llamaII70b)
            else:
                sys.exit("Enter valid --platform (MI300X | 2xH100)")
        else:
            sys.exit("Enter valid --precision (float16 | bfloat16)")

    elif args.model == "llama2-70b":
        PATH = PATH5
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(PATH, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        if args.precision == "float16": # pretrained precision : float16
            model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="auto")
        elif args.precision == "bfloat16":
            if args.platform == "MI300X":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_MI300X_llamaII70b)
            elif args.platform == "2xH100":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xH100_llamaII70b)
            elif args.platform == "2xMI250":
                model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map=DM_2xMI250_llamaII70b)
            else:
                sys.exit("Enter valid --platform (MI300X | 2xH100)")
        else:
            sys.exit("Enter valid --precision (float16 | bfloat16)")

    else:
        sys.exit("Enter valid --model (opt66b | llama65b | falcon40b-instruct | llama2-70b-chat | llama2-70b)")

    print("Model " + args.model + " loaded.")

    if args.profiling:
        prof_pref = FlopsProfiler(model)
        prof_pref_dec = FlopsProfiler(model)    # for altogether, yet to have decode phase only API

    if args.n <= 0:
        args.n = 10
    print("Benchmark to report is an average of " + str(args.n) + " runs ....\n")

    while True:
        p_latencies = []
        d_latencies = []
        dlen_actual = []    # actual decoding length (for large number of new tokens to generate, e.g. 512, some models fall short)

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

        print("Batch size = " + bs_s + ", prompt length = " + ps_s + ", generation length = " + gs_s)

        # 0 - warmup, 1 - profiling if set
        for i in range(2+args.n):
            prompts = []
            for b in range(bs):
                if args.d:
                    sequence = inputs               # len of 5, plus 1 below count for </s>
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
                if args.profiling and i == 1:
                    start_time = perf_counter()

                    prof_pref.start_profile()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1, use_cache=False)
                    prof_pref.stop_profile()

                    prefill_latency = perf_counter() - start_time

                    flops  = prof_pref.get_total_flops()
                    macs   = prof_pref.get_total_macs()
                    params = prof_pref.get_total_params()
                    print("\n************************** Models Prefill Profiling **************************\n")
                    prof_pref.print_model_profile(profile_step=i)
                    print("\n\n")
                    prof_pref.end_profile()
                else:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1, use_cache=False)
                    prefill_latency = perf_counter() - start_time
            else:
                if args.d: # deterministic prompt
                    if args.profiling and i == 1:
                        start_time = perf_counter()

                        prof_pref.start_profile()
                        generate_ids = model.generate(input_ids, max_new_tokens=1)
                        prof_pref.stop_profile()

                        prefill_latency = perf_counter() - start_time

                        flops  = prof_pref.get_total_flops()
                        macs   = prof_pref.get_total_macs()
                        params = prof_pref.get_total_params()
                        print("\n************************** Models Prefill Profiling **************************\n")
                        prof_pref.print_model_profile(profile_step=i)
                        print("\n\n")
                        prof_pref.end_profile()
                    else:
                        start_time = perf_counter()
                        generate_ids = model.generate(input_ids, max_new_tokens=1)
                        prefill_latency = perf_counter() - start_time
                else:
                    if args.profiling and i == 1:
                        start_time = perf_counter()

                        prof_pref.start_profile()
                        generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1)
                        prof_pref.stop_profile()

                        prefill_latency = perf_counter() - start_time

                        flops  = prof_pref.get_total_flops()
                        macs   = prof_pref.get_total_macs()
                        params = prof_pref.get_total_params()
                        print("\n************************** Models Prefill Profiling **************************\n")
                        prof_pref.print_model_profile(profile_step=i)
                        print("\n\n")
                        prof_pref.end_profile()
                    else:
                        start_time = perf_counter()
                        generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=1)
                        prefill_latency = perf_counter() - start_time

            # ignore the 1st (warmup) and 2nd (warmup/profiling) round
            if i > 1:
                p_latencies.append(prefill_latency)

            if args.nocache:
                if args.profiling and i == 1:
                    start_time = perf_counter()

                    prof_pref_dec.start_profile()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1), use_cache=False)
                    prof_pref_dec.stop_profile()

                    decode_latency = perf_counter() - start_time

                    flops  = prof_pref_dec.get_total_flops()
                    macs   = prof_pref_dec.get_total_macs()
                    params = prof_pref_dec.get_total_params()
                    print("\n************************** Prefill+Decode Profiling **************************\n")
                    prof_pref_dec.print_model_profile(profile_step=i)
                    print("\n\n")
                    prof_pref_dec.end_profile()
                else:
                    start_time = perf_counter()
                    generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1), use_cache=False)
                    decode_latency = perf_counter() - start_time
            else:
                if args.d: # deterministic prompt
                    if args.profiling and i == 1:
                        start_time = perf_counter()

                        prof_pref_dec.start_profile()
                        generate_ids = model.generate(input_ids, max_new_tokens=(gs+1))
                        prof_pref_dec.stop_profile()

                        decode_latency = perf_counter() - start_time

                        flops  = prof_pref_dec.get_total_flops()
                        macs   = prof_pref_dec.get_total_macs()
                        params = prof_pref_dec.get_total_params()
                        print("\n************************** Prefill+Decode Profiling **************************\n")
                        prof_pref_dec.print_model_profile(profile_step=i)
                        print("\n\n")
                        prof_pref_dec.end_profile()
                    else:
                        start_time = perf_counter()
                        generate_ids = model.generate(input_ids, max_new_tokens=(gs+1))
                        decode_latency = perf_counter() - start_time
                else:
                    if args.profiling and i == 1:
                        start_time = perf_counter()

                        prof_pref_dec.start_profile()
                        generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1))
                        prof_pref_dec.stop_profile()

                        decode_latency = perf_counter() - start_time

                        flops  = prof_pref_dec.get_total_flops()
                        macs   = prof_pref_dec.get_total_macs()
                        params = prof_pref_dec.get_total_params()
                        print("\n************************** Prefill+Decode Profiling **************************\n")
                        prof_pref_dec.print_model_profile(profile_step=i)
                        print("\n\n")
                        prof_pref_dec.end_profile()
                    else:
                        start_time = perf_counter()
                        generate_ids = model.generate(input_ids, do_sample=True, max_new_tokens=(gs+1))
                        decode_latency = perf_counter() - start_time

            # ignore the 1st (warmup) and 2nd (warmup/profiling) round
            if i > 1:
                d_latencies.append(decode_latency)
                dlen_actual.append(generate_ids.size()[1] - ps)

        # show generations for the last iteration
        if args.debug:
            print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))

        # adjustment
        Ptime = np.array(p_latencies).mean()
        # Dtime = (np.array(d_latencies).mean() - Ptime)/gs

        # -1 to count 1 token included in Ptime
        Dtime = ((np.array(d_latencies) - Ptime)/(np.array(dlen_actual) - 1)).mean()
        Ptime -= Dtime
        # error bound 5ms @ lower end
        Ptime = max(0.005, Ptime)


        # reports
        print("\n")
        print("Prefill phase latency on prompt of length   : " + ps_s + " = " + "{:.3f}".format(1000 * Ptime) + "ms")
        print("Decode latency per token on output of length: " + gs_s + " = " + "{:.3f}".format(1000 * Dtime) + "ms")


        cont = input("\nContinue another inference benchmark run? (yes | no) ")
        if cont.lower() == "no":
            break


if __name__=="__main__":
    main()
