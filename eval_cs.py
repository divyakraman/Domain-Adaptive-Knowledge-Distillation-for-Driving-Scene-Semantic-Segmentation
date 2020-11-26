import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import os.path as osp

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dtype = torch.cuda.FloatTensor #GPU


val_folder_imgs = 'path'

val_folder_labels = 'path'

val_labels_list = glob.glob(val_folder_labels+"/**/*_labelIds.png")
val_images_list = glob.glob(val_folder_imgs+"/**/*.png")

val_labels_list.sort()
val_images_list.sort()

batch_size = 1 
iterations = len(val_images_list)
num_classes = 19 

colors = [ [128,64,128],
[244,35,232],
[70,70,70],
[102,102,156],
[190,153,153],
[153,153,153],
[250,170,30],
[220,220,0],
[107,142,35],
[152,251,152],
[70,130,180],
[220,20,60],
[255,0,0],
[0,0,142],
[0,0,70],
[0,60,100],
[0,80,100],
[0,0,230],
[119,11,32] ]
#ignoring void class


def get_data(iter_num):
	images = torch.zeros([batch_size,3,512,1024])
	#images = torch.zeros([batch_size,3,1024,2048])
	images = images.cuda()
	RGB_image = Image.open(val_images_list[int(iter_num)]) #1024*2048*3
	RGB_image = RGB_image.resize((1024,512), Image.BILINEAR)
	RGB_image = np.array(RGB_image) #downsample image by 2
	RGB_image = RGB_image/255.0
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*1024*2048
	images[0,:,:,:] = RGB_image
	label_image = Image.open(val_labels_list[int(iter_num)]) #1024*2048
	label_image = label_image.resize((1024,512), Image.NEAREST)
	label_image = np.array(label_image)

	label_image[np.where(label_image == 0)] = 255
	label_image[np.where(label_image == 1)] = 255
	label_image[np.where(label_image == 2)] = 255
	label_image[np.where(label_image == 3)] = 255
	label_image[np.where(label_image == 4)] = 255
	label_image[np.where(label_image == 5)] = 255
	label_image[np.where(label_image == 6)] = 255
	label_image[np.where(label_image == 7)] = 0 #road
	label_image[np.where(label_image == 8)] = 1 #sidewalk
	label_image[np.where(label_image == 9)] = 255
	label_image[np.where(label_image == 10)] = 255
	label_image[np.where(label_image == 11)] = 2 #building
	label_image[np.where(label_image == 12)] = 3 #wall
	label_image[np.where(label_image == 13)] =  4 #fence
	label_image[np.where(label_image == 14)] = 255
	label_image[np.where(label_image == 15)] = 255
	label_image[np.where(label_image == 16)] = 255
	label_image[np.where(label_image == 17)] = 5 #pole
	label_image[np.where(label_image == 18)] = 255
	label_image[np.where(label_image == 19)] = 6 #traffic light
	label_image[np.where(label_image == 20)] = 7 #traffic sign
	label_image[np.where(label_image == 21)] = 8 #vegetation
	label_image[np.where(label_image == 22)] = 9 #nature
	label_image[np.where(label_image == 23)] = 10 #sky
	label_image[np.where(label_image == 24)] = 11 #person
	label_image[np.where(label_image == 25)] = 12 #rider
	label_image[np.where(label_image == 26)] = 13 #car
	label_image[np.where(label_image == 27)] = 14 #truck
	label_image[np.where(label_image == 28)] = 15 #bus
	label_image[np.where(label_image == 29)] = 255
	label_image[np.where(label_image == 30)] = 255
	label_image[np.where(label_image == 31)] = 16 #train
	label_image[np.where(label_image == 32)] = 17 #motorcycle
	label_image[np.where(label_image ==33)] = 18 #bicycle
	label_image[np.where(label_image == -1)] = 255
	
	label_image = torch.from_numpy(label_image)
	label_image = label_image.cuda()
	labels = torch.zeros([batch_size,512,1024])
	#labels = torch.zeros([batch_size,1024,2048])
	labels[0,:,:] = label_image
	
	del RGB_image,label_image
	return images,labels

def fast_hist(a,b,n):
	k = (a>=0) & (a<n)
	return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
	return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

net = torch.load('model.pth')


hist = np.zeros((num_classes,num_classes))
for iteration in range(len(val_labels_list)):
	images, label_ss = get_data(iteration)
	images = images.type(dtype)
	label_ss = label_ss.type(dtype)
	pred_intermediate, pred = net(images)
	pred = pred.detach()
	pred = pred.cpu()
	pred = pred.numpy()
	pred = pred[0,:,:,:]
	pred = np.argmax(pred,0)
	label_ss = label_ss.cpu()
	label_ss = label_ss.numpy()
	label_ss = label_ss[0,:,:]
	accuracy = sum(sum(label_ss==pred))/(512*1024)
	print(accuracy)
	'''
	pred_color_labels = np.zeros((720,1280,3))
	for i in range(len(colors)):
		pred_color_labels[np.where(pred_labels_upsampled==i)]=colors[i]
	image_name = str(iteration)+'.jpg'
	pred_color_labels = pred_color_labels/255.0
	#plt.imsave(image_name,pred_color_labels)
	'''
	hist += fast_hist(label_ss.flatten(), pred.flatten(), num_classes)
	torch.cuda.empty_cache() #clear cached memory
	print(iteration)

mIoUs = per_class_iu(hist)

for ind_class in range(num_classes):
	print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))

print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

print('===> Accuracy Overall: ' + str(np.diag(hist).sum() / hist.sum() * 100))
acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) 
'''
for ind_class in range(num_classes):
	print('===> Class '+str(ind_class)+':\t'+str(round(acc_percls[ind_class] * 100, 2))) 
'''





