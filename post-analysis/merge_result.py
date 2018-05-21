import glob
import struct
from tensorflow.core.example import example_pb2
import re

def example_generator(data_path):
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    filelist = sorted(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

def return_full_index_str(index):
	return '0'*(6-len(str(index)))+str(index)

def return_decoded_title(path):
	with open(path,'r', encoding='utf-8') as f:
		for line in f:
			return line

def main():
	out_root = 'D:/graduation project/paper/experimens/May-merge_result/'
	root_path = 'D:/graduation project/paper/experimens/moldes-May/'
	decoded_dirs = \
	[root_path+'May-baseline-252716/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-252716/decoded/',
	root_path+'May-baseline-pos-238712/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-238712/decoded/',
	root_path+'May-pointer-151850/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-151850/decoded/',
	root_path+'May-pointer-pos-152199/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-152199/decoded/' ]
	
	model_names = ['=======baseline=======\n252k:\n',
								'238k-with pos:\n',
								'\n=======pointer=======\n151k:\n',
								'152k-with pos:\n']

	input_gen = example_generator('J:/WS/PA-HG/finished_files_pos_ner/chunked/test_*')
	index = 0
	while True:
		if index % 1000 == 0:
			print('start to write %d' % index)
		e = next(input_gen)
		abstract_text = e.features.feature['article'].bytes_list.value[0].decode()
		title_text = e.features.feature['abstract'].bytes_list.value[0].decode()
		# discard <s>  </s>
		title_text = title_text[4:-5]
		decoded_name = return_full_index_str(index) + '_decoded.txt'
		with open(out_root+return_full_index_str(index)+'.txt','w',encoding='utf-8') as out:
			out.write('=======abstract=======\n')
			out.write(abstract_text+'\n')
			out.write('=======title=======\n')
			out.write(title_text+'\n\n\n')
			for i in range(4):
				out.write(model_names[i])
				decoded_title = return_decoded_title(decoded_dirs[i]+decoded_name)
				out.write('\t'+decoded_title+'\n')

		index += 1

if __name__ == '__main__':
	main()
