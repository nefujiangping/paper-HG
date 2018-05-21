import numpy as np
import scipy.stats as stats
import math


def clean(scores_1):
	scores_1 = scores_1.tolist()
	bad_count = int(scores_1[8000])
	del_idx_list = []
	for idx , e in enumerate(scores_1[8001:]):
		del_idx_list.append(int(e))
	del_idx_list = sorted(del_idx_list, reverse=True)
	# remove last elements
	for _ in range(bad_count+1):
		scores_1.pop()
	for idx in del_idx_list:
		scores_1.pop(idx)
	return scores_1

def cal(decoded_pos_co_scores_1, decoded_pos_co_scores_2):
	scores_1 = clean(decoded_pos_co_scores_1)
	scores_2 = clean(decoded_pos_co_scores_2)
	print(scores_1[100:120])
	print(scores_2[100:120])
	print(np.mean(scores_1))
	print(np.mean(scores_2))
	stat_val, p_val = stats.ttest_ind(scores_1, scores_2, equal_var=True)
	print('Two-sample t-statistic D = %6.3f, p-value = %6.4f' % (stat_val, p_val))
	# use formula to compute.
	n_1 = len(scores_1)
	n_2 = len(scores_2)
	mean_1 = np.mean(scores_1)
	mean_2 = np.mean(scores_2)
	print(mean_1)
	print(mean_2)
	variance_1 = (np.std(scores_1))**2
	variance_2 = (np.std(scores_2))**2
	T = (mean_1 - mean_2) / math.sqrt(  (1/float(n_1)+1/float(n_2)) \
		 * ( (n_1-1) * variance_1 + (n_2-1)*variance_2 ) / (n_1 + n_2 - 2)       )
	print(T)

def cal_with_clean():
	data_dir = 'data/co/'
	# model_name_1 = 'May-pointer-151850'
	# model_name_2 = 'May-pointer-pos-152199'

	model_name_1 = 'May-baseline-252716'
	model_name_2 = 'May-baseline-pos-238712'
	pos_co_scores_1 = np.load(data_dir + model_name_1+'decoded-pos-co-score.npy')
	pos_co_scores_2 = np.load(data_dir + model_name_2+'decoded-pos-co-score.npy')
	cal(pos_co_scores_1, pos_co_scores_2)

def simple_cal():
	data_dir = 'data/len/'
	data1 = np.load(data_dir + '8-decoded_rates.npy')
	data2 = np.load(data_dir + '10-decoded_rates.npy')
	stat_val, p_val = stats.ttest_ind(data1, data2, equal_var=True)
	print('Two-sample t-statistic D = %6.3f, p-value = %6.4f' % (stat_val, p_val))

simple_cal()