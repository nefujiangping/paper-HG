from __future__ import division
import glob
import os
import io
import logging
import json
from stanfordcorenlp import StanfordCoreNLP
from threading import Thread
import threading
import time
import collections


global_idx = 1
total_count = 1	

def file_content(file):
	lines = []
	for line in file:
		lines.append(line)
	lines = [line.strip() for line in lines]
	return lines

def text_generator(data_path):
	global total_count
	file_list = glob.glob(data_path)
	total_count = len(file_list)
	logging.info('There are %d files in total.' % total_count)
	assert file_list, ('Error: Empty filelist at %s' % data_path)
	for f in file_list:
		_ , short_name = os.path.split(f);
		lines = [short_name]
		with open(f, 'r') as file:
			lines.append(file_content(file))
			yield lines

def start_tokenize(nlp,data_path):
	input_gen = text_generator(data_path)
	while True:
		try:
			lines = input_gen.next()
		except StopIteration:
			logging.info('There is no more example.')
			break;
		tokenize_pos_ner_one_example(nlp,lines)

def tokenize_pos_ner_one_example(nlp,lines):
	global global_idx
	(filename, abstract, title) = lines[0],lines[1][0],lines[1][1]
	if global_idx % 100 == 1:
		print threading.currentThread().getName() + " : Writing story %i of %i; %.2f percent done" % (global_idx, total_count, float(global_idx)*100.0/float(total_count))
	abs_tokens, abs_pos, abs_ner = nlp_operation(nlp,abstract)
	title_tokens, title_pos, title_ner = nlp_operation(nlp,title)
	pos_vocab_counter.update(abs_pos)
	pos_vocab_counter.update(title_pos)
	ner_vocab_counter.update(abs_ner)
	ner_vocab_counter.update(title_ner)
	try:
		with io.open(os.path.join(out_root, filename), 'w', encoding='utf-8') as out:
			out.write(' '.join(abs_tokens)+'\n')
			out.write(' '.join(abs_pos)+'\n')
			out.write(' '.join(abs_ner)+'\n')
			out.write(' '.join(title_tokens)+'\n')
			out.write(' '.join(title_pos)+'\n')
			out.write(' '.join(title_ner))
	except UnicodeEncodeError:
		print('%s has unicode character.' %filename)
	global_idx += 1


def nlp_operation(nlp,text):
	res = nlp.annotate(text, properties=props)
	json_obj = json.loads(res)
	tokens = []
	pos = []
	ner = []
	for sentence in json_obj['sentences']:
		for token in sentence['tokens']:
			tokens.append(token['word'])
			pos.append(token['pos'])
			ner.append(token['ner'])
	return (tokens,pos,ner)

def init_logging():
  logging.basicConfig(level=logging.INFO)

data_path = '/media/jayson/software/WS/PA-HG/raw/'
out_root = '/media/jayson/software/WS/preprocessed123'
vocab_root = '/media/jayson/software/WS/vocab'
# Because there's a mass of data to tokenize, we divivded it into 10 folder
# filename with *0,*1,*2...,*9 respectively.
tokenize_index = '0'

init_logging()		

props={'annotators': 'tokenize,ssplit,pos,ner','pipelineLanguage':'en','outputFormat':'json'}	

pos_vocab_counter = collections.Counter()
ner_vocab_counter = collections.Counter()

with  StanfordCoreNLP(r'/media/jayson/download/Java-Tools/stanford-corenlp-full-2018-02-27', memory='4g') as nlp:
	start_tokenize(nlp,data_path + '*' + tokenize_index)

logging.info('start to write pos_vocab...')
with open(os.path.join(vocab_root, 'pos_vocab_'+tokenize_index), 'w') as writer:
	for word, count in pos_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')

logging.info('start to write ner_vocab...')
with open(os.path.join(vocab_root, 'ner_vocab_'+tokenize_index), 'w') as writer:
 	for word, count in ner_vocab_counter.most_common():
 		writer.write(word + ' ' + str(count) + '\n')
	




