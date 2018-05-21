import tensorflow as tf
import glob
import json
import os
import numpy as np
import time


UNKNOWN_TOKEN = '[UNK]'

# May-baseline-252716 ,Average pos-co score of 8000 examples is: 0.8423
#		T = -7.2894954506
# May-baseline-pos-238712 ,Average pos-co score of 8000 examples is: 0.8812

cluster_list = \
[
	['-LRB-', '-RRB-'],
	[','],
	[':'],
	['.'],
	['``', '\'\''],
	['#'],
	['$'],
	['CC'],
	['CD'],
	['PDT', 'WDT', 'DT'],
	['EX'],
	['FW'],
	['IN'],
	['JJ', 'JJR', 'JJS'],
	['LS'],
	['NN', 'NNS', 'NNP', 'NNPS'],
	['POS'],
	['PRP','PRP$'],
	['RB', 'RBR', 'RBS', 'RP'],
	['SYM'],
	['TO'],
	['UH'],
	['MD'],
	['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
	['WP', 'WP$', 'WRB']
]

use_cluster = True
skip_count = 0
skip_index = []

# May-pointer-pos-152199
model_name = 'May-pointer-pos-152199'
attn_root = \
	'/media/jayson/study/graduation project/paper/experimens/moldes-May/'+ \
		model_name + '/attn_vis'
decoded_pos_ner_root = \
	'/media/jayson/study/graduation project/paper/experimens/moldes-May/'+ \
		model_name + '/decoded_pos_ner'

def get_content(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f:
			lines.append(line.strip())
	return lines

def in_witch_cluster(tag):
	for idx, clstr in enumerate(cluster_list):
		if tag in clstr: return idx

def they_are_similar(decoded_tag, src_tag):
	# print('%s, %s' % (decoded_tag, src_tag) )
	cluster_idx = in_witch_cluster(decoded_tag)
	if src_tag in cluster_list[cluster_idx]: return True
	return False

def get_score(idx_lst, src_tags, tag, src_tokens=None):
	count = 0
	for i in idx_lst:
		# if use_cluster:
		if they_are_similar(tag, src_tags[i]): count += 1
		# else
		# 	if src_tags[i] == tag: count += 1
		# if src_tokens != None: print('%s(%s), ' % (src_tokens[i], src_tags[i]))
	return count

def cal_one_example(sess, index, attn_json, tokens_pos_ner):
	'''
	Args:
		attn_json: a json object, containing article_lst, article_pos_lst, article_ner_lst,
			decoded_lst, abstract_str, attn_dists, p_gens(if pointer_gen is ON).
		decoded_pos:
			string, pos_tag list split by space.  
	'''
	# print(attn_json['decoded_lst'])
	# print(decoded_pos)
	global skip_count
	global skip_index
	decoded_pos_lst = tokens_pos_ner[1].split(' ')
	# decoded token length should match the pos/ner tag list
	# assert len(attn_json['decoded_lst']) == len(decoded_pos_lst), '%d example: decoded token length should match the pos/ner tag list! ' % index
	if len(attn_json['decoded_lst']) != len(decoded_pos_lst):
		skip_count += 1
		skip_index.append(index)
		print('Example %d, length of decoded tokens and the pos/ner tag list do NOT match, skip it.' % index)
		return 0
	
	input_arr = tf.constant(attn_json['attn_dists'], tf.float32)
	_, top_k_indices = tf.nn.top_k(input_arr, 2)
	k_indices = sess.run(top_k_indices)
	decoded_token_lst = tokens_pos_ner[1].split(' ')
	# print(decoded_token_lst)
	decoded_len = len(decoded_pos_lst)
	t_score = 0
	for idx, tag in enumerate(decoded_pos_lst):
		# print('decoded:%s(%s) ' % (attn_json['decoded_lst'][idx], tag))
		# if current token is '[UNK]', then skip it.
		if attn_json['decoded_lst'][idx] == UNKNOWN_TOKEN: continue
		score = get_score(k_indices[idx], attn_json['article_pos_lst'], tag, attn_json['article_lst'])
		# print(score)
		t_score += score
	t_score /= float(decoded_len)
	# print(t_score)
	return t_score

def get_config():
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  return config

def start(EXAMPLE_NUM):
	global skip_count
	global skip_index
	score = float(0)
	scores = []
	pieces = [[6000,8000]]
	# pieces = [[3000, 3500]]
	aa = range(EXAMPLE_NUM)
	for j in pieces:
		with tf.Session(config=get_config()) as sess:
			for index in aa[j[0]: j[1]]:
				# print('index : %d' % index )
				if index%500 == 0: print('Example %d starts...' % index)
				attn_path = os.path.join(attn_root, '%06d_attn_vis_data.json' % index)
				decoded_pos_ner_path = os.path.join(decoded_pos_ner_root, '%06d_decoded.txt' % index)
				attn_str = get_content(attn_path)[0]
				tokens_pos_ner = get_content(decoded_pos_ner_path)
				scores.append(cal_one_example(sess, index, json.loads(attn_str), tokens_pos_ner))
				score += scores[-1]
		print('sleep 60 seconds......')
		time.sleep(60)
	# fill extra data
	scores.append(float(skip_count))
	for index in skip_index:
		scores.append(index)
	# save the score list for further use.
	np.save('6c-'+model_name+'decoded-pos-co-score.npy',np.array(scores))
	# print something.
	print('Total : %d,  Skip Count : %d .' % (EXAMPLE_NUM, skip_count))
	print('%s ,Average pos-co score of %d examples is: %.4f' % (model_name, EXAMPLE_NUM-skip_count, score/float(EXAMPLE_NUM-skip_count)))
	# scores: first EXAMPLE_NUM elements are scores to each decoded_headline,
	# 				EXAMPLE_NUM+1: the count of skipped examples,
	#					follows are skipped examples' index

def main():
	EXAMPLE_NUM = 8000
	print('%s, starts............' % model_name)
	start(EXAMPLE_NUM)

if __name__ == '__main__':
	tf.app.run()