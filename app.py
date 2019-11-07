from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import add_special_tokens_, predict_next
from argparse import ArgumentParser
import torch
import re 
from pprint import pformat

from flask import Flask, render_template, request, jsonify, redirect, url_for 
from flask_socketio import SocketIO, send 

import os 
import logging
import yaml 
import mysql.connector 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret' 
socketio = SocketIO(app, cors_allowed_origins="*") 


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

def check_repetition(text: str) -> list: 

  regexes = [
    "\n",
    "\. |\? |! |\.\"", 
    "\. ",
    "\."
  ]

  for regex in regexes: 
    sentences = re.split(regex, text)
    if len(sentences) > 1 and any([sentence ==sentences[-1] for sentence in reversed(sentences[:-1])]): 

      repeated_sentence = sentences[-1]
      previous_sentence = sentences[sentences[:-1].index(repeated_sentence)]
      print("Repeated text detected. Removing repetition and forcing new text... ")
      print(f"\tPrevious sentence: {previous_sentence}\n\t{repeated_sentence}")

      # tell model not to generate this token again that leads to repetition
      prevent_token = tokenizer.encode(sentences[-1])[0]
      print(f"prevent_token: {prevent_token} - {tokenizer.decode(prevent_token)}")

      # split as groups for restoring original text. 
      regex = f"({regex})"
      split_group = re.split(regex, text)
      out_text = ''.join(split_group[:-1])
      return out_text, prevent_token 

  return text, None  

@socketio.on('message')
def handle_input(message: str): 
  """ Handle request and output model score in json format"""
  if args is None: 
    print("No initialization done before. Initializing model.")
    initialize()

  title = message
  punc = ['.', '?', '!']
  prevent_token = None 

  src = tokenizer.encode(title)
  current_output = [] 
  while True: 
    with torch.no_grad():  
      out_id = predict_next(model, tokenizer, src, current_output, args, prevent_token)
      if out_id: 
        current_output.append(out_id)
      else: 
        break 
      out_text = tokenizer.decode(current_output, skip_special_tokens=True)
      # print(tokenizer.decode(out_id))
      new_token = tokenizer.decode(out_id)
      if any([(p in new_token) for p in punc]): 
        print("Check repetition")
        out_text, prevent_token = check_repetition(out_text)
        current_output = tokenizer.encode(out_text)
        send(out_text)
        socketio.sleep(0)

        # stop generation 
        if prevent_token is not None:
          break 
      send(out_text)
      socketio.sleep(0)
  
  send("complete")
  save_to_db(title, out_text)


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
  print("Initializing model.")
  initialize()

if __name__ == '__main__':
  # app.run(host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False, passthrough_errors=True)
  # app.run(host='0.0.0.0', port=5000, debug=True)
  socketio.run(app, debug=True)
