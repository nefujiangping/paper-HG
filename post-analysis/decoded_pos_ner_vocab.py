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

pos_vocab_counter = collections.Counter()
ner_vocab_counter = collections.Counter()

def file_content(file):
	lines = []
	for line in file:
		lines.append(line)
	lines = [line.strip() for line in lines]
	return lines


def text_generator(data_path):
	file_list = glob.glob(data_path)
	assert file_list, ('Error: Empty filelist at %s' % data_path)
	for f in file_list:
		with open(f, 'r') as file:
			yield file_content(file)

def start(data_path):
	global pos_vocab_counter
	global ner_vocab_counter
	input_gen = text_generator(data_path)
	while True:
		try:
			lines = input_gen.next()
		except StopIteration:
			logging.info('There is no more example.')
			break;
		pos_vocab_counter.update(lines[1].split(' '))
		ner_vocab_counter.update(lines[2].split(' '))

model_name = 'May-pointer-pos-152199'
data_path = '/media/jayson/study/graduation project/paper/experimens/moldes-May/%s/decoded_pos_ner/*' % model_name
out_root =   '/media/jayson/study/graduation project/paper/experimens/moldes-May/%s' % model_name

start(data_path)

print "Writing pos_vocab file..."
with open(os.path.join(out_root, "pos_vocab"), 'w') as writer:
	for word, count in pos_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print "Finished writing pos_vocab file"

print "Writing ner_vocab file..."
with open(os.path.join(out_root, "ner_vocab"), 'w') as writer:
	for word, count in ner_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print "Finished writing ner_vocab file"

