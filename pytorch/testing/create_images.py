
from PIL import Image
import numpy as np

#import matplotlib.pyplot as plt

import csv

images = []

with open( '../../augmented_images.txt' ) as imfile: 
	csvreader = csv.reader( imfile )
	for row in csvreader: 
		im = row
		images.append( im )

index = 0
for row in images: 

	im = np.asarray( row[:-1], dtype='float64' )
	print( im.shape )

	im = np.reshape( im, (32, 32, 3), order='F' )
	#im = im/255.
	#im = im.transpose( 1, 2, 0 )

	img = Image.fromarray( im, 'RGB' )
	img.save( 'cifar-augmented-' + str(index) + '.png' )
	index += 1
