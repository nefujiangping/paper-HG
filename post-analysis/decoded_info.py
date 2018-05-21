from __future__ import division
import glob
import numpy as np


def get_text(filename):
	lines = []
	with open(filename, 'r', encoding="utf-8") as f:
		for line in f:
			lines.append(line.strip())
	return lines


def get_tokens(text):
	return text.split()


def get_source_rate(abs_tokens, headline_tokens):
	count = 0
	for token in headline_tokens:
		if token in abs_tokens:
			count += 1
	return count / len(headline_tokens)


def get_sum(lis):
	ret = 0
	for e in lis:
		ret += e
	return ret

glod_headline_lens = []
decoded_headline_lens = []
glod_rates = []
decoded_rates = []

model_line_number = 16 # 8,10,14,16
data_path = 'D:/graduation project/paper/experimens/May-merge_result/*'
file_list = glob.glob(data_path)
file_count = len(file_list)
for filename in file_list:
	# get glod headline, decoded headline respectively
	all_text = get_text(filename)
	abs_text = all_text[1]
	glod_headline = all_text[3]
	decoded_headline = all_text[model_line_number]
	# get tokens of glod/decoded headline and abstract
	glod_headline_tokens = get_tokens(glod_headline)
	decoded_headline_tokens = get_tokens(decoded_headline)
	abs_tokens = get_tokens(abs_text)

	# print('Abstract tokens: ')
	# print(abs_tokens)

	# print('glod headline tokens: ')
	# print(glod_headline_tokens)

	# print('decoded headline tokens: ')
	# print(decoded_headline_tokens)

	# record each len to the list
	glod_headline_lens.append(len(glod_headline_tokens))
	decoded_headline_lens.append(len(decoded_headline_tokens))
	# calculate rate 
	glod_headline_rate = get_source_rate(abs_tokens, glod_headline_tokens)
	decoded_headline_rate = get_source_rate(abs_tokens, decoded_headline_tokens)
	# record rate 
	glod_rates.append(glod_headline_rate)
	decoded_rates.append(decoded_headline_rate)


np.save(str(model_line_number)+'-glod_headline_lens.npy',np.array(glod_headline_lens))
np.save(str(model_line_number)+'-decoded_headline_lens.npy',np.array(decoded_headline_lens))
np.save(str(model_line_number)+'-glod_rates.npy',np.array(glod_rates))
np.save(str(model_line_number)+'-decoded_rates.npy',np.array(decoded_rates))

total_glod_headline_len = get_sum(glod_headline_lens)
total_decoded_headline_len = get_sum(decoded_headline_lens)
total_glod_rate = get_sum(glod_rates)
total_decoded_rate = get_sum(decoded_rates)

print('Total count: %d ' % file_count)
print('model %d:' % model_line_number)
print('Average glod_headline_len : %.4f' % (total_glod_headline_len/file_count))
print('Average decoded_headline_len : %.4f' % (total_decoded_headline_len/file_count))
print('Average glod_rate : %.4f' % (total_glod_rate/file_count))
print('Average decoded_rate : %.4f' % (total_decoded_rate/file_count))





