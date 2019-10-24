from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import add_special_tokens_, predict, MAX_LEN, predict_next
from argparse import ArgumentParser
import torch
from pprint import pformat
import random

from flask import Flask, render_template, request, jsonify, redirect, url_for 
import os 
import logging

app = Flask(__name__)
app.secret_key = 'justin12'

model = None 
tokenizer = None 
args = None 
title = None
output = None 

@app.route('/api', methods=['GET'])
def api(): 
  """ Handle request and output model score in json format"""
  # Handle empty requests. 
  if not request.json: 
    return jsonify({'error': 'no request received'})

  else: 
    src = request.json.get('src', None)
    out_text = generate_article(model, tokenizer, src) 
    return jsonify({'article': out_text})
  # Parse request args into feature array for prediction

def get_model_and_tokenizer(): 
  global model 
  global tokenizer 
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__file__)
  logger.info(pformat(args))

  logger.info("Get pretrained model and tokenizer")
  tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
  tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
  model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
  model = model_class.from_pretrained(args.model_checkpoint)
  model.to(args.device)
  model.eval()
  add_special_tokens_(model, tokenizer)


def generate_article(model, tokenizer, source=None): 

  global args 
  src = tokenizer.encode(source)
  with torch.no_grad():
      out_ids = predict(model, tokenizer, src, args)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

  return out_text

@app.route('/main', methods=['GET', 'POST'])
def main(): 
  global title
  global output 
  global args

  parser = ArgumentParser()
  parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
  parser.add_argument("-mc", "--model_checkpoint", type=str, default="model/", help="Path, url or short name of the model")
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

  if not model: 
    get_model_and_tokenizer()

  if request.method =='GET': 
  	return render_template('index.html')
  if request.method =='POST': 
    print(request.form)
    title = request.form.get('title', None)
    if title:
      src = tokenizer.encode(title) 
      current_output = [] 
      with torch.no_grad():
        while len(current_output) < MAX_LEN: 
          next_token = predict_next(model, tokenizer, src, current_output, args)
          if next_token:
            current_output.append(next_token)
            output = tokenizer.decode(current_output, skip_special_tokens=True)
            render_template('index.html', title=title, output=output)
          else: 
            break

      return render_template('index.html', title=title, output=output)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=403, debug=True)