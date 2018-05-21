import glob
import collections
import os


def file_content(file_name):
	lines = []
	with open(file_name, 'r') as file:
		for line in file:
			lines.append(line)
	lines = [line.strip() for line in lines]
	return lines

in_path = '/media/jayson/software/WS/preprocessed/preprocessed/*'
vocab_root = '/media/jayson/study/graduation project/paper/experimens/moldes-May'

pos_vocab_counter = collections.Counter()
ner_vocab_counter = collections.Counter()
file_list = glob.glob(in_path)

for idx, file_name in enumerate(file_list):
	if idx % 10000 == 0: print('start to update vocab of example %d...' % idx)
	lines = file_content(file_name)
	pos_vocab_counter.update(lines[1].split(' '))
	pos_vocab_counter.update(lines[4].split(' '))
	ner_vocab_counter.update(lines[2].split(' '))
	ner_vocab_counter.update(lines[5].split(' '))

print('start to write pos_vocab...')
with open(os.path.join(vocab_root, 'pos_vocab_full'), 'w') as writer:
	for word, count in pos_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print('pos_vocab is written to %s.' % os.path.join(vocab_root, 'pos_vocab_full'))

print('start to write ner_vocab...')
with open(os.path.join(vocab_root, 'ner_vocab_full'), 'w') as writer:
	for word, count in ner_vocab_counter.most_common():
		writer.write(word + ' ' + str(count) + '\n')
print('pos_vocab is written to %s.' % os.path.join(vocab_root, 'ner_vocab_full'))
