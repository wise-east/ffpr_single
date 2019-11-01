from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import add_special_tokens_, predict
from argparse import ArgumentParser
import torch
from pprint import pformat
import random

from flask import Flask, render_template, request, jsonify, redirect, url_for 
import os 
import logging
import yaml 
import mysql.connector 

app = Flask(__name__)
app.secret_key = 'justin12'

model = None 
tokenizer = None 
args = None 
db = None 

def save_to_db(title: str, output: str) -> None:
  
  cursor = db.cursor() 
  sql = "INSERT INTO main (title, article) VALUES (%s, %s)"
  val = (title, output)
  cursor.execute(sql, val) 
  db.commit() 
  
  cursor.close() 

@app.route('/api', methods=['GET'])
def api(): 
  """ Handle request and output model score in json format"""
  if args ==None: 
    initialize()

  title = None 

  # Handle GET requests: 
  if request.method == "GET": 
    if request.args: 
      title = request.args.get("src", None)

  if title is not None: 
    print(f"Received valid request through API - \"src\": {title}")
  else: 
    return jsonify({"error": "Invalid JSON request. Provide title as {\"src\": \"<your title>\"} for POST requests and ?src=\"<your title>\" for GET requests."})

  src = tokenizer.encode(title)
  with torch.no_grad():
      out_ids = predict(model, tokenizer, src, args)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

  # save_to_db(title, out_text)

  return jsonify({"title": title, "article": out_text})

def initialize(): 
  global args
  global model 
  global tokenizer 
  global db

  # initialize args 
  config = yaml.safe_load(open('config/config.yaml', 'r'))
  args = config['default']
  args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__file__)
  logger.info(pformat(args))

  # initialize model and tokenizer 
  logger.info("Get pretrained model and tokenizer")
  tokenizer_class = GPT2Tokenizer if "gpt2" == args['model'] else OpenAIGPTTokenizer
  tokenizer = tokenizer_class.from_pretrained(args['model_checkpoint'])
  model_class = GPT2LMHeadModel if "gpt2" == args['model'] else OpenAIGPTLMHeadModel
  model = model_class.from_pretrained(args['model_checkpoint'])
  model.to(args['device'])
  model.eval()
  add_special_tokens_(model, tokenizer)

  # connect to database 
  db_config = config['mysql']
  db = mysql.connector.connect(
    host=db_config['host'], 
    user=db_config['user'],
    passwd=db_config['passwd'], 
    database=db_config['database']
  )

  logger.info("Initialization of model, tokenizer, and DB connection complete.")

if args is None: 
  initialize()


if __name__ == '__main__':
  # app.run(host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False, passthrough_errors=True)
  app.run(host='0.0.0.0', port=5000, debug=True)
