# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits



def predict(model, tokenizer, src, args): 

    model.eval()

    special_tokens_ids = tokenizer.convert_tokens_to_ids(list(SPECIAL_TOKENS))

    current_output = [] 

    for i in range(1024):

        instance = build_input_from_segments(src, current_output, tokenizer, with_eos = False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)

        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if prev in current_output[-args.no_repeat_length:] or (i < args.min_length and prev.item() in special_tokens_ids):
            while prev in current_output[-args.no_repeat_length:] or prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating repeated token or special token with probability 1.")
                    w_message = f"Current output: {tokenizer.decode(current_output)}, output to be added: {tokenizer.decode([prev.item()])}"
                    warnings.warn(w_message)
                    break  # avoid infinitely looping over repeated or special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        current_output.append(prev.item())
        # print(tokenizer.decode(current_output))


    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("-mc", "--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_repeat_length", type=int, default=5, help="Provide length from end of current output that should not be repeated for new token added to the current output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    while True:
        src = input(">>> ")
        while not src:
            print('Prompt should not be empty!')
            src = input(">>> ")
        src = tokenizer.encode(src)
        with torch.no_grad():
            out_ids = predict(model, tokenizer, src, args)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()
