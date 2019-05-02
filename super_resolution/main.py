import argparse, os
import random, time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import utils
import logging
from tensorboardX import SummaryWriter

import model_zoo
import dataset

pytorch_names = []
model_names = sorted(model_zoo.model_names + pytorch_names)

scales = [2,3,4]

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SR")
parser.add_argument('--upscale_factor', '-u', type=int, default=2, choices=scales, help="super resolution upscale factor")
parser.add_argument('--dataset', metavar='DIR', type=str, default='/data/super-resolution/vdsr/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='VDSR', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: VDSR)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--iter-size', default=1, type=int)
parser.add_argument('--test-batch-size', '-t', default=1, choices=[1], type=int)
parser.add_argument('--batch-size', '-b', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_policy', type=str, default='fix_step', help='learning rate update policy')
parser.add_argument('--lr_fix_step', type=int, default=30, help='learning rate step for fix_step')
parser.add_argument('--lr_custom_step', type=list, default=[20,30,40], help='learning rate steps for custom_step')
parser.add_argument('--lr_decay', type=float, default=0.98, help='decay for every epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--no-decay-small', action='store_false', default=True)
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-s', default=-1, type=int, help='epoch to save model (default: -1)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=3, type=int, help='seed for initializing training. ')
parser.add_argument('--resume', '-r', action='store_true', default=False)
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument('--log_dir', type=str, default='logs', help='log dir')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--resume_file', type=str, default='checkpoint.pth.tar')
parser.add_argument('--tensorboard', action='store_true', default=False)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--case', type=str, default='normal', help='identify the configuration of the training')

def main():
    opt = parser.parse_args()
    args = opt
    is_best = True
    best_acc = 0.0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_filename = os.path.join(args.log_dir, args.arch + '-' + args.case + '.txt')
    utils.setup_logging(log_filename, resume=args.resume)

    if args.tensorboard:
        args.tensorboard = SummaryWriter(args.log_dir, filename_suffix='.' + args.arch + '.' + args.case)
    else:
        args.tensorboard = None

    utils.check_folder(args.weights_dir)
    args.weights_dir = os.path.join(args.weights_dir, args.arch)
    utils.check_folder(args.weights_dir)
    args.resume_file = os.path.join(args.weights_dir, args.case + '-' + args.resume_file)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    else:
        logging.info('You have chosen to training without setting a seed')

    ngpus_per_node = torch.cuda.device_count()
    logging.info('gpu number on the node {}'.format(ngpus_per_node))

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True
    logging.info('traing case: %s' % args.case)
    # create model
    logging.info("=> creating model '{}'".format(args.arch))
    if args.arch in pytorch_names:
        if args.pretrained:
            logging.info("=> load pre-trained parapmter")
            model = models.__dict__[args.arch](pretrained=True)
        else:
            model = models.__dict__[args.arch]()
    elif args.arch in model_zoo.model_names:
        model = model_zoo.models(args.arch)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume_file):
            logging.info("=> loading checkpoint '{}'".format(opt.resume_file))
            checkpoint = torch.load(opt.resume_file)
            opt.start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint['best_acc']
            utils.load_state_dict(model, checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_file, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(opt.resume_file))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    logging.info("===> Loading datasets with batch_size %d and test-batch-size %d" % (args.batch_size, args.test_batch_size))
    val_dataset = DatasetFromMat(os.path.join(args.dataset, 'val'), filters='x' + str(args.upscale_factor),
            input_label='im_b_y', target_label='im_gt_y')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    if args.evaluate:
        acc = validate(val_loader, model, criterion, args)
        logging.info('evaluate accuracy %f' % acc)
        return

    train_dataset = DatasetFromHdf5(os.path.join(args.dataset, 'train'))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    logging.info("===> Setting Optimizer")
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
        lr = utils.adjust_learning_rate(optimizer, epoch, args)
        logging.info('learning rate %f at epoch %d' %(lr, epoch))

        loss = train(training_loader, optimizer, model, criterion, epoch, args)

        # evaluate on validation set
        predict, bicubic = validate(val_loader, model, criterion, args, epoch)
        is_best = predict > best_acc
        if is_best:
            best_acc = predict
        logging.info('evaluate accuracy: {}, best: {}'.format(predict, best_acc))

        if args.tensorboard is not None:
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/lr', lr, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/eval-predict', predict, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/eval-bicubic', bicubic, epoch)
            args.tensorboard.add_scalar(args.arch + '-' + args.case + '/train-loss', loss, epoch)

        utils.save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args)


def train(training_loader, optimizer, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(training_loader):
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        if i % args.iter_size == 0:
            optimizer.zero_grad()

        loss.backward() 

        nn.utils.clip_grad_norm_(model.parameters(), args.clip) 

        if i % args.iter_size == (args.iter_size - 1):
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info("===> Epoch[{}]({}/{}): Loss: {:.10f}, Batch time: {} / {}, Data time: {} / {}"
                    .format(epoch, i, len(training_loader), losses.avg, batch_time.avg, batch_time.val, data_time.avg, data_time.val))

        return losses.avg
 
def validate(val_loader, model, criterion, args, epoch):
    predict = AverageMeter()
    bicubic = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        #end = time.time()
        for i, (input, target) in enumerate(val_loader):

            psnr = utils.PSNR(target, input, shave_border=args.upscale_factor)
            bicubic.update(psnr, args.val_batch_size)

            input = input / 255.
            tensor = torch.from_numpy(input).float().view(1, -1, input.shape[0], input.shape[1])
            tensor = tensor.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(tensor)

            HR = output.cpu()
            im_h_y = HR.data[0].numpy().astype(np.float32)
            im_h_y = im_h_y * 255.
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.] = 255.
            im_h_y = im_h_y[0,:,:]

            psnr = utils.PSNR(target, im_h_y, shave_border=args.upscale_factor)
            predict.update(psnr, args.val_batch_size)

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            #if i % args.print_freq == 0:

    return predict.avg, bicubic.avg

if __name__ == "__main__":
    main()

