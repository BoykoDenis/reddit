import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import pandas as pd
import glob
from memory_profiler import profile
from sys import getsizeof
import gc
# %% [code]
class Conv_land(nn.Module):
	#@profile
	def __init__(self, input_chanels, n_featurs, output_featurs):
		super().__init__()

		self.conv1_1 = nn.Conv2d(input_chanels, n_featurs*4, kernel_size = 3, stride = 1, padding = 0)
		self.conv1_2 = nn.Conv2d(n_featurs*4, n_featurs*4, kernel_size = 3, stride = 1, padding = 0)

		#self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

		self.conv2_1 = nn.Conv2d(n_featurs*4, n_featurs*6, kernel_size = 3, stride = 1, padding = 0)
		self.conv2_2 = nn.Conv2d(n_featurs*6, n_featurs*6, kernel_size = 3, stride = 1, padding = 0)

		self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

		self.conv3_1 = nn.Conv2d(n_featurs*6, n_featurs*8, kernel_size = 3, stride = 1, padding = 1)
		self.conv3_2 = nn.Conv2d(n_featurs*8, n_featurs*8, kernel_size = 3, stride = 1, padding = 1)

		self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

		self.conv4_1 = nn.Conv2d(n_featurs*8, n_featurs*10, kernel_size = 3, stride = 1, padding = 1)
		self.conv4_2 = nn.Conv2d(n_featurs*10, n_featurs*10, kernel_size = 3, stride = 1, padding = 1)

		self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

		self.conv5_1 = nn.Conv2d(n_featurs*10, n_featurs*12, kernel_size = 2, stride = 1, padding = 0)

		self.linear1 = nn.Linear(n_featurs*12, output_featurs)
	#@profile
	def forward(self, x):

		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))

		#x = self.pool1(x)

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))

		x = self.pool2(x)

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))

		x = self.pool3(x)

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))

		x = self.pool4(x)

		x = F.relu(self.conv5_1(x))
		
		x = x.view(x.shape[0], -1) 

		x = F.softmax(self.linear1(x))

		return x

	def save_mod(self, filename):

		checkpoint = {"state_dictionary" : self.state_dict()}
		torch.save(checkpoint, filename)

	def load_mod(self, filename):

		checkpoint = torch.load(filename)
		self.load_state_dict(checkpoint['state_dictionary'])
	#@profile
	def make_df(self, data, ids):

		df = pd.DataFrame(columns=['id', 'landmarks'])
		torch.cuda.empty_cache()
		for i, [img, ids] in enumerate(zip(data, ids)):
			
			output = self.forward(img.unsqueeze(0)).cpu().detach()
			output = torch.max(output, 1)
			#ids.append([output[1] + 1, output[0]])
			out1 = output[1] + 1
			out2 = output[0]
			out = str(out1.numpy()) + ' ' + str(out2.numpy())
			df.loc[i] = ids, out


		#df = pd.DataFame({'id': data,
		#		  		  'landmarks': ids})

		return df
		
	#@profile
	def load_data(self, data_path, main_file, full_set = False, select_batch = None, return_labels = True, LABELS = None):
        
		ids = []
		batch = []
		transform_train = transforms.Compose([transforms.Resize((24, 24)),
											transforms.ToTensor()])
		if return_labels:
			labels = []
			LABELS = list(LABELS[::-1])
			LABELS = [int(i) for i in LABELS]
            
		if full_set:
			for idx0, first in enumerate(os.listdir(data_path + main_file)):
				for idx, second in enumerate(os.listdir(data_path + main_file + '/' + first)):
					for idx2, third in enumerate(os.listdir(data_path + main_file + '/' + first + '/' + second)):
						print('loading: ', str(idx0) + '/16', '||', str(idx) + '/16 ', '||', str(idx2) + '/16 ',  end = '\r')
						for image in os.listdir(data_path + main_file + '/' + first + '/' + second + '/' + third):

							img = Image.open(data_path + main_file + '/' + first + '/' + second + '/' + third + '/' + image).convert('1')
							im = transform_train(img)
							batch.append(im)

							img.close()
							ids.append(image[:-4])
							img = None
							im = None
							image = None
							del img
							del im
							del image
							if return_labels:
								labels.append(LABELS.pop(-1)-1)
					gc.collect(generation=2)
		else:
			for idx, second in enumerate(os.listdir(data_path + main_file + '/' + select_batch)):
				for idx2, third in enumerate(os.listdir(data_path + main_file + '/' + select_batch + '/' + second)):
					print('loading: ', idx, '/16 ', '||', idx2, '/16 ',  end = '\r')
					for image in os.listdir(data_path + main_file + '/' + select_batch + '/' + second + '/' + third):

						img = Image.open(data_path + main_file + '/' + select_batch + '/' + second + '/' + third + '/' + image).convert('1')
						#print("img before: ", getsizeof(image)/(1024)**2, "GB")
						im = transform_train(img)
						batch.append(im)
						img.close()
						#print("img after: ", getsizeof(img)/(1024)**2, "GB")
						ids.append(image[:-4])
						img = None
						im = None
						image = None
						#print("img after after: ", getsizeof(image)/(1024)**2, "GB")
						del im
						del img
						del image
						if return_labels:
							labels.append(LABELS.pop(-1)-1)
				print("labe: ", getsizeof(labels)/(1024)**2, "GB")
				print("img: ", getsizeof(batch)/(1024)**2, "GB")
				gc.collect(generation=2)
		transform_train = None
		del transform_train
		if return_labels:
			return batch, labels

		batch = torch.stack(batch)
		return batch, ids
