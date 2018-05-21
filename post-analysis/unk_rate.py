import glob
import os

UNKNOWN_TOKEN = '[UNK]'
model_name = 'May-baseline-252716'
iterations = model_name[-6:]

def get_content(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f:
			lines.append(line.strip())
	return lines

decoded_path = '/media/jayson/study/graduation project/paper/experimens/moldes-May/'\
+'%s/decode_test_400maxenc_4beam_3mindec_20maxdec_ckpt-%s/decoded/*' % (model_name, iterations)
file_list = glob.glob(decoded_path)

count = 0
for file in file_list:
	decoded_headline = get_content(file)[0]
	if UNKNOWN_TOKEN in decoded_headline: count += 1
print(' %s , unk rate : %.4f , count : %d.' % (model_name ,count/float(len(file_list)), count))
# May-baseline-252716 ,          unk rate : 0.2075 , count : 1660.
# May-baseline-pos-238712 , unk rate : 0.2259 , count : 1807.
# May-pointer-151850 ,            unk rate : 0.0103 , count : 82.
# May-pointer-pos-152199 ,   unk rate : 0.0186 , count : 149.
#  0.2075,0.2259,0.0103, 0.0186

