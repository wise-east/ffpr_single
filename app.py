from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import add_special_tokens_, predict, MAX_LEN, predict_next
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
title = None
output = None 
db = None 

@app.route('/api', methods=['GET'])
def api(): 
  """ Handle request and output model score in json format"""

  if args ==None: 
    initialize()

  # Handle empty requests. 
  if not request.json: 
    return jsonify({'error': 'no request received'})

  else: 
    src = request.json.get('src', None)
    print(f"Received GET request through API - \"src\": {src}")
    out_text = generate_article(model, tokenizer, src) 
    return jsonify({'article': out_text})
  # Parse request args into feature array for prediction


def generate_article(model, tokenizer, source=None): 

  global args 
  src = tokenizer.encode(source)
  with torch.no_grad():
      out_ids = predict(model, tokenizer, src, args)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

  return out_text

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

  # check connection 
  cursor = db.cursor()
  cursor.execute("SHOW TABLES")
  for x in cursor: 
    print(x) 

@app.route('/', methods=['GET', 'POST'])
def main(): 
  global title
  global output 

  # order is important 
  if args == None: 
    initialize()

  if request.method =='GET': 
  	return render_template('index.html')
  if request.method =='POST': 
    print(request.form)
    title = request.form.get('title', None)
    if title:

      # search if same query exists in the database 
      cursor = db.cursor() 
      sql=f"SELECT * FROM main WHERE title='{title}'"
      # print(sql)
      cursor.execute(sql)
      result = cursor.fetchone()
      if result: 
        # last column is the article 
        output = result[-1]

      # if it doesn't exist, run new prediction and store in database
      else: 
        src = tokenizer.encode(title) 
        current_output = [] 
        with torch.no_grad():
          while len(current_output) < MAX_LEN: 
            next_token = predict_next(model, tokenizer, src, current_output, args)
            if next_token:
              current_output.append(next_token)
              output = tokenizer.decode(current_output, skip_special_tokens=True)
              # doesn't do anything 
              # render_template('index.html', title=title, output=output)
            else: 
              break

        # insert result to the database 
        sql = "INSERT INTO main (title, article) VALUES (%s, %s)"
        val = (title, output)
        cursor.execute(sql, val)
        db.commit()


      cursor.close()
      return render_template('index.html', title=title, output=output)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=403, use_debugger=False, use_reloader=False, passthrough_errors=True)