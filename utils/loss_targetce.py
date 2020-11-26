import torch.nn as nn
import torch
from utils.loss import CrossEntropy2d
import numpy as np 

def loss_target_crossEntropy(student_pred, teacher_pred, gpu_id):
	teacher_pred = teacher_pred.detach()
	teacher_pred = teacher_pred.cpu()
	teacher_pred = teacher_pred.numpy()
	teacher_pred = teacher_pred[:,:,:,:]
	teacher_pred = np.argmax(teacher_pred,1)
	teacher_pred = torch.from_numpy(teacher_pred)
	teacher_pred = teacher_pred.cuda(gpu_id)
	criterion = CrossEntropy2d().cuda(gpu_id)
	return criterion(student_pred, teacher_pred)