import matplotlib.pyplot as plt
import numpy as np

model_line_number = 8 # 8,10,14,16

glod_headline_lens = np.load(str(model_line_number)+'glod_headline_lens.npy')
decoded_headline_lens = np.load(str(model_line_number)+'-decoded_headline_lens.npy')

plt.figure()
plt.hist(glod_headline_lens,50)
plt.figure()
plt.hist(decoded_headline_lens,50)