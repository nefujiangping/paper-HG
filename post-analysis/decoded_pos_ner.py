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
import re


UNKNOWN_TOKEN = '[UNK]'
UNKNOWN_REPLACEMENT = 'unkunkunk'

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
	global glb_time
	filename = lines[0]
	# fix [UNK] problem before pos/ner
	title = re.sub(r'\[UNK\]', UNKNOWN_REPLACEMENT, lines[1][0])
	if global_idx % 100 == 1:
		tm = time.time()
		print("Writing story %i of %i; %.2f percent done" % (global_idx, total_count, float(global_idx)*100.0/float(total_count)))
		print("This 100 examples use %.3f s" % (tm-glb_time))
		glb_time = tm
	title_tokens, title_pos, title_ner = nlp_operation(nlp,title)
	try:
		with io.open(os.path.join(out_root, filename), 'w', encoding='utf-8') as out:
			out.write(' '.join(title_tokens)+'\n')
			out.write(' '.join(title_pos)+'\n')
			out.write(' '.join(title_ner))
	except UnicodeEncodeError:
		print('%s has unicode character.' % filename)
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

model_name = 'May-pointer-pos-152199'
iteration_num = '152199'
data_path = '/media/jayson/study/graduation project/paper/experimens/moldes-May/' \
		+'%s/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-%s/decoded/*' % (model_name, iteration_num)
out_root =   '/media/jayson/study/graduation project/paper/experimens/moldes-May/' \
		+'%s/decoded_pos_ner' % model_name

if not os.path.exists(out_root): os.mkdir(out_root)

init_logging()		

props={'annotators': 'tokenize,ssplit,pos,ner','pipelineLanguage':'en','outputFormat':'json'}	

glb_time = time.time()																	
with  StanfordCoreNLP(r'/media/jayson/download/Java-Tools/stanford-corenlp-full-2018-02-27', memory='4g') as nlp:
	start_tokenize(nlp,data_path)
