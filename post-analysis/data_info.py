from __future__ import division
import os
import sys
import io
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline


train_list_dir='train_test_val_filenames/train_filenames'
test_list_dir='train_test_val_filenames/test_filenames'
val_list_dir='train_test_val_filenames/val_filenames'

tokenized_dir = 'tokenized'

def calculate_statistic_info(dir,option):
	art_token_num = 0
	abs_token_num = 0
	sample_num = 0
	art_data_list=[]
	abs_data_list=[]
	with io.open(dir, encoding='utf-8') as train_list:
		for sample_file_name in train_list:
			if(sample_num % 3000 == 0): print("sample[%d] starts..." % sample_num)
			art_len, abs_len = token_len_in_one_sample(os.path.join(tokenized_dir, sample_file_name.strip()))
			sample_num += 1
			art_token_num += art_len
			abs_token_num += abs_len
			art_data_list.append(art_len)
			abs_data_list.append(abs_len)
	np.save(option+'-article.npy',np.array(art_data_list))
	np.save(option+'-abstract.npy',np.array(abs_data_list))
	print("%d samples." % sample_num)
	print("article Total tokens: ", art_token_num)
	print("Average article length: %f" % (art_token_num/sample_num))
	print("Average abstract length: %.2f" % (abs_token_num/sample_num))


def token_len_in_one_sample(sample_dir):
	with io.open(sample_dir, encoding='utf-8') as f:
		lines = []
		for line in f.readlines():
			lines.append(line.strip())
		art_tokens = lines[0].split(' ')
		abs_tokens = lines[1].split(' ')
	return len(art_tokens), len(abs_tokens)

calculate_statistic_info(train_list_dir,'train')
calculate_statistic_info(test_list_dir,'test')
calculate_statistic_info(val_list_dir,'val')

def draw(option):
	art_data_list = np.load(option+'-article.npy')
	abs_data_list = np.load(option+'-abstract.npy')
	plt.figure()
	plt.hist(art_data_list,50)
	plt.figure()
	plt.hist(abs_data_list,50)
	
draw('train')
draw('test')
draw('val')