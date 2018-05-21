import tensorflow as tf
import glob
import json
import os
import numpy as np
import time


UNKNOWN_TOKEN = '[UNK]'

# May-baseline-252716, Average pos-co score of 1024 examples is: 0.8322
#   T = -1.305  p-value = 0.1921
# May-baseline-pos-238712 ,Average pos-co score of 1028 examples is: 0.8687

# May-pointer-151850 ,Average pos-co score of 1026 examples is: 1.1832
#   T = -0.379 p-value = 0.7044
# May-pointer-pos-152199 ,Average pos-co score of 1016 examples is: 1.1917

model_name = 'May-pointer-pos-152199'

attn_root = \
  'D:/graduation project/paper/experimens/moldes-May/'+ \
    model_name + '/attn_vis'
decoded_pos_ner_root = \
  'D:/graduation project/paper/experimens/moldes-May/'+ \
    model_name + '/decoded_pos_ner'

def get_content(filename):
  lines = []
  with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines

def in_witch_cluster(tag):
  for idx, clstr in enumerate(cluster_list):
    if tag in clstr: return idx

def they_are_similar(decoded_tag, src_tag):
  cluster_idx = in_witch_cluster(decoded_tag)
  if src_tag in cluster_list[cluster_idx]: return True
  return False

def get_score(idx_lst, src_tags, tag):
  count = 0
  for i in idx_lst:
    if src_tags[i] == tag: count += 1
  return count

def cal_one_example(sess, index, attn_json, decoded_ner_lst):
  '''
  Args:
    attn_json: a json object, containing article_lst, article_pos_lst, article_ner_lst,
      decoded_lst, abstract_str, attn_dists, p_gens(if pointer_gen is ON).
    decoded_pos:
      string, pos_tag list split by space.  
  '''
  input_arr = tf.constant(attn_json['attn_dists'], tf.float32)
  _, top_k_indices = tf.nn.top_k(input_arr, 2)
  k_indices = sess.run(top_k_indices)
  t_score = 0
  count = 0
  for idx, tag in enumerate(decoded_ner_lst):
    if attn_json['decoded_lst'][idx] == UNKNOWN_TOKEN: continue
    if tag == 'O': continue
    count += 1
    score = get_score(k_indices[idx], attn_json['article_ner_lst'], tag)
    t_score += score
  t_score /= float(count)
  return t_score

def get_config():
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  return config

def all_O(decoded_ner_lst):
  for tag in decoded_ner_lst:
    if tag != 'O': return False
  return True

def start(EXAMPLE_NUM):
  score = float(0)
  count = 0
  scores = []
  pieces = [[0,2000],[2000,4000],[4000,6000],[6000,8000]]
  aa = range(EXAMPLE_NUM)
  for j in pieces:
    with tf.Session(config=get_config()) as sess:
      t0=time.time()
      for index in aa[j[0]: j[1]]:
        # print('index : %d' % index )
        if index%500 == 0: print('Example %d starts...' % index)
        attn_path = os.path.join(attn_root, '%06d_attn_vis_data.json' % index)
        decoded_pos_ner_path = os.path.join(decoded_pos_ner_root, '%06d_decoded.txt' % index)
        attn_str = get_content(attn_path)[0]
        tokens_pos_ner = get_content(decoded_pos_ner_path)
        decoded_ner_lst = tokens_pos_ner[2].split(' ')
        # everything is 'O', skip
        if all_O(decoded_ner_lst): continue
        attn_json = json.loads(attn_str)
        # something wrong, skip
        if len(attn_json['decoded_lst']) != len(decoded_ner_lst): continue
        # everything is good, then calculate
        one_score = cal_one_example(sess, index, attn_json, decoded_ner_lst)
        if one_score >= float(1.9): print()
        scores.append(one_score)
        score += scores[-1]
        count += 1
      print('Use %.2f s.' % (time.time()-t0))
    print('sleep 60 seconds......')
    time.sleep(10)
  np.save(''+model_name+'decoded-pos-co-score.npy',np.array(scores))
  # print something.
  print('%s ,Average pos-co score of %d examples is: %.4f' % (model_name, len(scores), np.mean(np.array(scores))))

def main(args):
  EXAMPLE_NUM = 8000
  print('%s, starts............' % model_name)
  start(EXAMPLE_NUM)

if __name__ == '__main__':
  tf.app.run()