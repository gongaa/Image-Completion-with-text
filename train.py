import os
import sys
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_loader import Samplerian, DataLoaderian
from data_utils import get_data
from models import *
from net_utils import *

from subprocess import call

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a segmentation completion network')
	parser.add_argument('--dataset', dest='dataset',
											help='training dataset',
											choices=['cityscape'],
											default='cityscape')
	parser.add_argument('--val', dest='val',
											help='whether eval after each training ',
											action='store_true')
	parser.add_argument('--val_interval', dest='val_interval',
											help='number of epochs to evaluate',
											type=int,
											default=1)
	parser.add_argument('--model', dest='model',
											help='model to use',
											choices=['u_net', 'wgan', 'encoder_decoder'],
											default='u_net')
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--epochs', dest='max_epochs',
											help='number of epochs to train',
											default=20, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=10, type=int)
	parser.add_argument('--save_dir', dest='save_dir',
											help='directory to save models', default="models",
											type=str)
	parser.add_argument('--nw', dest='num_workers',
											help='number of worker to load data',
											default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',
											help='whether use CUDA',
											action='store_true')                    
	parser.add_argument('--mGPUs', dest='mGPUs',
											help='whether use multiple GPUs',
											action='store_true')
	parser.add_argument('--bs', dest='batch_size',
											help='batch_size',
											default=1, type=int)

	# config optimization
	parser.add_argument('--o', dest='optimizer',
											help='training optimizer',
											choices =['adam', 'sgd'],
											default="adam")
	parser.add_argument('--lr', dest='lr',
											help='starting learning rate',
											default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
											help='step to do learning rate decay, unit is epoch',
											default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
											help='learning rate decay ratio',
											default=0.1, type=float)

	# set training session
	parser.add_argument('--s', dest='session',
											help='training session',
											default=1, type=int)

	# resume trained model
	parser.add_argument('--r', dest='resume',
											help='resume checkpoint or not',
											default=False, type=bool)
	parser.add_argument('--checksession', dest='checksession',
											help='checksession to load model',
											default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
											help='checkepoch to load model',
											default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
											help='checkpoint to load model',
											default=0, type=int)
	# log and diaplay
	# not implemented yet
	parser.add_argument('--use_tfb', dest='use_tfboard',
											help='whether use tensorboard',
											action='store_true')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)


	####################### set up data loader for train and val ######################
	n_classes, train_imgs, train_segs, train_masks, \
	val_imgs, val_segs, val_masks = get_data(args.dataset, args.val)

	train_size = train_imgs.shape[0]
	sampler_batch_train = Samplerian(train_size, args.batch_size)
	dataset_train 	= DataLoaderian(train_imgs, train_segs, train_masks, n_classes, training=True)
	dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
													sampler=sampler_batch_train, num_workers=args.num_workers)
	if args.val:
		val_size = val_imgs.shape[0]
		sampler_batch_val = Samplerian(val_size, val_size)
		dataset_val 	= DataLoaderian(val_imgs, val_segs, val_masks, n_classes, training=False)
		dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=val_size,
														sampler=sampler_batch_val, num_workers=args.num_workers)

	###################### set up model #####################
	if args.model == "u_net":
		model = Model(n_classes, "u_net")
	elif args.model == "encoder_decoder":
		model = Model(n_classes, "encoder_decoder")

	###################### init variable #####################
	im_data = torch.FloatTensor(1)
	seg_data = torch.FloatTensor(1)
	mask_data = torch.FloatTensor(1)
	gt_data = torch.LongTensor(1)

	if args.cuda:
		im_data = im_data.cuda()
		seg_data = seg_data.cuda()
		mask_data = mask_data.cuda()
		gt_data =  gt_data.cuda()
		model.cuda()

	if args.mGPUs:
		model = nn.DataParallel(model)

	##################### set up optimizer ####################
	lr = args.lr
	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif args.optimizer == "sgd":
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

	###################### minor stuff #########################
	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	##################### training begins #######################
	iters_per_epoch = math.ceil(train_size / args.batch_size) 

	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	for epoch in range(args.start_epoch, args.max_epochs + 1):
		# setting to train mode
		model.train()
		loss_temp = 0
		tic = time.time()
		if epoch % (args.lr_decay_step + 1) == 0:
				adjust_learning_rate(optimizer, args.lr_decay_gamma) # zl decay ALL learning rate, lr times decay_gamma
				lr *= args.lr_decay_gamma

		data_iter = iter(dataloader_train)
		for step in range(iters_per_epoch):
			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			seg_data.data.resize_(data[1].size()).copy_(data[1])
			mask_data.data.resize_(data[2].size()).copy_(data[2])
			# gt_data_i = squeeze_seg(data[1], n_classes)
			gt_data.data.resize_(data[3].size()).copy_(data[3])
			model.zero_grad()

			output_segs, rec_loss = model(im_data, seg_data, mask_data, gt_data) 
			loss = rec_loss.mean()
			loss_temp += float(loss.item())
			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % args.disp_interval == 0:
				toc = time.time()
				if step > 0:
					loss_temp /= (args.disp_interval +1 )
				if args.mGPUs:
					loss_rec = rec_loss.mean().item()	
				else:
					loss_rec = rec_loss.item()	

				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																% (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
				print("\t\t\ttime cost: %f\trecons: %.4f" % (toc-tic, loss_rec))
				sys.stdout.flush()
				if args.use_tfboard:
					info = {
						'loss': loss_temp,
						'loss_rec': loss_rec,
					}
					logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
				loss_temp = 0
				tic = time.time()
		if args.val:
			model.eval()
			data_iter = iter(dataloader_val)
			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			seg_data.data.resize_(data[1].size()).copy_(data[1])
			mask_data.data.resize_(data[2].size()).copy_(data[2])

			output_segs, rec_loss = model(im_data, seg_data, mask_data) 
			loss = rec_loss.mean()

			print("eval loss: %.4f " % (float(loss.item)))

		save_name = os.path.join(args.save_dir, '{}_{}_{}_{}.pth'.format(args.model, args.session, epoch, step)) 

		torch.save({
			'session': args.session,
			'epoch': epoch + 1,
			'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
			'optimizer': optimizer.state_dict()
		}, save_name)
		print('save model: {}'.format(save_name))
		sys.stdout.flush()

	if args.use_tfboard:
		logger.close()



