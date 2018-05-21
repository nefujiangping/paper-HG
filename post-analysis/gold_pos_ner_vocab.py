import glob
import os
import collections

def file_content(file_name):
	lines = []
	with open(file_name, 'r') as file:
		for line in file:
			lines.append(line)
	lines = [line.strip() for line in lines]
	return lines

data_path = '/media/jayson/software/WS/preprocessed/preprocessed'
name_list_file_path = '/media/jayson/software/WS/PA-HG/train_test_val_filenames/test_filenames'
vocab_root = '/media/jayson/study/graduation project/paper/experimens/moldes-May'

pos_vocab_counter = collections.Counter()
ner_vocab_counter = collections.Counter()

short_name_list = file_content(name_list_file_path)
for short_name in short_name_list:
	lines = file_content(os.path.join(data_path, short_name))
	pos_lst = lines[4].split(' ')
	ner_lst = lines[5].split(' ')
	pos_vocab_counter.update(pos_lst)
	ner_vocab_counter.update(ner_lst)

print('start to write pos_vocab...')
with open(os.path.join(vocab_root, 'pos_vocab_gold'), 'w') as writer:
	for word, count in pos_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print('pos_vocab is written to %s.' % os.path.join(vocab_root, 'pos_vocab_full'))

print('start to write ner_vocab...')
with open(os.path.join(vocab_root, 'ner_vocab_gold'), 'w') as writer:
	for word, count in ner_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print('pos_vocab is written to %s.' % os.path.join(vocab_root, 'ner_vocab_glod'))