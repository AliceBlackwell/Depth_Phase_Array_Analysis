import numpy as np

txt_file = 'total_pP_arrays.txt'
#peru = 'total_pP_arrays_peru.txt'

name, no = np.loadtxt(txt_file, unpack=True, usecols=(0,1), dtype='int')
#peru_name, peru_no = np.loadtxt(peru, unpack=True, usecols=(0,1), dtype='int')

total_no = 0
print(no)

for i in range (len(no)):
	total_no += no[i]
	
print('total tested arrays', total_no)
print('Success rate', (3172/total_no)*100)
