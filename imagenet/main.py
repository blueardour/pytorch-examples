import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import model_zoo
import utils
import logging

from tensorboardX import SummaryWriter

pytorch_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names = sorted(model_zoo.model_names + pytorch_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', type=str, default='/data/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--iter-size', default=1, type=int)
parser.add_argument('--val-batch-size', '-v', default=50, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_policy', type=str, default='fix_step', help='learning rate update policy')
parser.add_argument('--lr_fix_step', type=int, default=30, help='learning rate step for fix_step')
parser.add_argument('--lr_custom_step', type=list, default=[20,30,40], help='learning rate steps for custom_step')
parser.add_argument('--lr_decay', type=float, default=0.98, help='decay for every epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--no-decay-small', action='store_false', default=True)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=-1, type=int, help='epoch to save model (default: -1)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--resume', '-r', action='store_true', default=False)
parser.add_argument('--log_dir', type=str, default='logs', help='log dir')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--resume_file', type=str, default='checkpoint.pth.tar')
parser.add_argument('--case', type=str, default='normal', help='identify the configuration of the training')
parser.add_argument('--tensorboard', action='store_true', default=False)

best_acc1 = 0

def main():
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_filename = args.arch + '-' + args.case
    if args.evaluate:
        log_filename += '-eval.txt'
    else:
        log_filename += '.txt'
    log_filename = os.path.join(args.log_dir, log_filename)
    utils.setup_logging(log_filename, resume=args.resume)

    if args.tensorboard and args.evaluate == False:
        args.tensorboard = SummaryWriter(args.log_dir, filename_suffix='.' + args.arch + '.' + args.case)
    else:
        args.tensorboard = None

    utils.check_folder(args.weights_dir)
    args.weights_dir = os.path.join(args.weights_dir, args.arch)
    utils.check_folder(args.weights_dir)
    args.resume_file = os.path.join(args.weights_dir, args.case + '-' + args.resume_file)

    if args.iter_size < 1:
        logging.info('iter_size must be equal or greater than one')
        return

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    logging.info('gpu number on the node {}'.format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        logging.info('multiprocessing distributed training')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        logging.info('Simply single node training with single process')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    logging.info('traing case: %s' % args.case)
    # create model
    logging.info("=> creating model '{}'".format(args.arch))
    if args.arch in model_zoo.model_names and args.arch in pytorch_names:
        logging.info("=> netork both defined in Pytorch official zoo and custom local zoo folder")
        logging.info("=> custom local zoo folder get more priority")
        model = model_zoo.models(args.arch)
    elif args.arch in pytorch_names:
        if args.pretrained:
            logging.info("=> load pre-trained parapmter")
            model = models.__dict__[args.arch](pretrained=True)
        else:
            model = models.__dict__[args.arch]()
    else:
        model = model_zoo.models(args.arch)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume_file):
            logging.info("=> loading checkpoint '{}'".format(args.resume_file))
            checkpoint = torch.load(args.resume_file)
            if 'best_acc1' in checkpoint:
                best_acc1 = checkpoint['best_acc1']
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            logging.info("=> loaded parameter from epoch {} with best acc {}"
                .format(args.start_epoch, best_acc1))

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            utils.load_state_dict(model, state_dict)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume_file))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        ## DataParallel will divide and allocate batch_size to all available GPUs
        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features)
        #    model.cuda()
        #else:
        #    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    logging.info("loading dataset with batch_size {} and test-batch-size {}".
        format(args.batch_size, args.val_batch_size))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valdir = os.path.join(args.data, 'val')
    if args.val_batch_size < 1:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate and val_loader is not None:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        logging.info('evaluate accuracy top1(%f), top5(%f)' % (acc1, acc5))
        return

    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        shape = value.shape
        if args.no_decay_small and ((len(shape) == 4 and shape[1] == 1) or (len(shape) == 1)):
            params += [{'params':value, 'weight_decay':0}]
        else:
            params += [{'params':value}]
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = utils.adjust_learning_rate(optimizer, epoch, args)
        logging.info('learning rate %f at epoch %d' %(lr, epoch))

        # train for one epoch
        tc1, tc5 = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        logging.info('evaluate accuracy top5(%f), top1(%f), best top1: %f' % (acc5, acc1, best_acc1))

        if args.tensorboard is not None:
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/lr', lr, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/eval-top5', acc5, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/eval-top1', acc1, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/train-top5', tc5, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/train-top1', tc1, epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        if i % args.iter_size == 0:
            optimizer.zero_grad()

        loss.backward()

        if i % args.iter_size == (args.iter_size - 1):
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, epoch=0):
    if val_loader is None:
        logging.info("val_loader is None, skip evaluation")
        return 0, 0

    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

    return top1.avg, top5.avg


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
