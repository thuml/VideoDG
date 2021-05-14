import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from module import models_adv
from opts.opts import args
from utils.dataset_ucf101 import TSNDataSet
from utils.transforms import *
from utils.utils import *


def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_loss_0 = AverageMeter()
    adv_loss_1 = AverageMeter()

    mse_criterion = torch.nn.MSELoss().cuda()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    model.train_inception = True
    model.judge_test = False
    end = time.time()
    n = 0

    for i, (input,  target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)

        if args.add_dg:
            adv_target = target.cuda(async=True)
            adv_target = torch.autograd.Variable(adv_target)
            source_input = input
            source_input = torch.autograd.Variable(source_input, volatile=True)
            adv_input = input.clone()
            adv_input = torch.autograd.Variable(adv_input, requires_grad=True)
            # print(adv_input.requires_grad)
            adv_input_block = input.clone()
            adv_input_block_1 = input.clone()
            adv_input_block = torch.autograd.Variable(adv_input_block, requires_grad=True)
            max_optimizer = torch.optim.SGD([{'params': adv_input}], 1.0)
            max_optimizer_1 = torch.optim.SGD([{'params': adv_input_block}], 1.0)
            # src = torch.autograd.Variable(src)
            # tar = torch.autograd.Variable(tar)

            for k in range(5):
                with torch.no_grad():
                    _, source_feature, adv_, adv_temp_feature = model(source_input)

                adv_y, adv_feature, adv_feature_other, adv_feature_other_temp = model(adv_input)
                max_loss = criterion(adv_y, adv_target) - args.alpha * mse_criterion(adv_feature, source_feature)
                max_loss = -max_loss
                max_optimizer.zero_grad()
                max_loss.backward()
                max_optimizer.step()
                adv_loss_0.update(max_loss.item(), input.size(0))

                # with torch.no_grad():
                #    _,  source_feature_1, adv_1, adv_temp_feature_1 = model(source_input)

                adv_y_1, adv_feature_1, adv_feature_other_1, adv_feature_other_1_temp = model(adv_input_block)
                max_loss_1 = criterion(adv_feature_other_1, adv_target) - args.alpha * mse_criterion(
                    adv_feature_other_1_temp,
                    adv_temp_feature)
                max_loss_1 = -max_loss_1
                max_optimizer_1.zero_grad()
                max_loss_1.backward()
                max_optimizer_1.step()

                adv_loss_1.update(max_loss_1.item(), input.size(0))

            all_input = torch.cat((input_var, adv_input), dim=0)
            all_input = torch.cat((all_input, adv_input_block), dim=0)
            all_label = torch.cat((adv_target, target), dim=0)
            all_label = torch.cat((all_label, target), dim=0)

        else:
            all_input = input_var
            all_label = target

        output, temp_feature, _, _ = model(all_input)
        loss = criterion(output, all_label)

        prec1, prec5 = accuracy(output.data, all_label, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        n = n + 1

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.train_inception = False
    model.judge_test = True
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output, _, _, _ = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test : [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            log.flush()

    return top1.avg, losses.avg

def test(val_loader, model):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.train_inception = False
    model.judge_test = True
    end = time.time()

    outputs = []
    labels = []
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            # compute output
            output, _, _, _ = model(input)
            loss = criterion(output, target)
            outputs.append(output.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)

    outputs = np.concatenate(outputs)
    labels = np.concatenate(labels)
    return outputs, labels





def make_train_loader(train_list, model):
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation()

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if "pku" in train_list:
        prefix = "image_{:05d}.jpg"
    elif "something" in train_list:
        prefix = "{:06d}.jpg"
    elif ("NTU" in train_list) or ("ucf" in train_list):
        prefix = "image_{:04d}.jpg"
    elif ("hmdb" in train_list):
        prefix = "image_{:06d}.jpg"
    else:
        prefix = "{:06d}.jpg"

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix if args.modality in ["RGB",
                                                          "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return train_loader

def make_test_loader(test_list, model):
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if "pku" in test_list:
        prefix = "image_{:05d}.jpg"
    elif "something" in test_list:
        prefix = "{:06d}.jpg"
    elif ("NTU" in test_list) or ("ucf" in test_list):
        prefix = "image_{:04d}.jpg"
    elif ("hmdb" in test_list):
        prefix = "image_{:06d}.jpg"
    else:
        prefix = "{:06d}.jpg"

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", test_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix if args.modality in ["RGB",
                                                          "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader

def main():
    global best_prec1
    check_rootfolders()

    args.store_name = '_'.join(
        ['APN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)
    args.store_name = args.snapshot_pref

    model = models_adv.APN(args.num_class, args.num_segments, args.modality,
                               base_model=args.arch,
                               consensus_type=args.consensus_type,
                               dropout=args.dropout,
                               img_feature_dim=args.img_feature_dim,
                               partial_bn=not args.no_partialbn)
        

    policies = model.get_optim_policies()
    model_backup = model
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[30,60,90,120],
                                                     gamma=0.1)

    test_list = args.test_list.split(" ")
    train_loader = make_train_loader(args.train_list, model_backup)
    val_loader = make_test_loader(args.val_list, model_backup)
    test_loader01 = make_test_loader(test_list[0], model_backup)
    test_loader02 = make_test_loader(test_list[1], model_backup)


    if args.mode == "test":
        test(val_loader, model)
        test(test_loader01, model)
        test(test_loader02, model)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.snapshot_pref), 'w')

    summary_dir = "summary/{}".format(args.snapshot_pref)
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    checkpoint_dir = "model/{}".format(args.snapshot_pref)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, log_training, writer)
        writer.add_scalar("Train/acc", train_acc, epoch)
        writer.add_scalar("Train/loss", train_loss, epoch)
        output = "Train Epoch {}: accuracy={:.4f}, loss={:.4f}".format(epoch, train_acc, train_loss)
        print(output)
        log_training.write(output+'\n')

        scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_acc, val_loss = validate(val_loader, model, criterion, log_training)
            test01_acc, test01_loss = validate(test_loader01, model, criterion, log_training)
            test02_acc, test02_loss = validate(test_loader02, model, criterion, log_training)
            test_acc = (test01_acc*len(test_loader01) + test02_acc*len(test_loader02)) / (len(test_loader01) + len(test_loader02))

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
            # remember best prec@1 and save checkpoint

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': test_acc,
            }
            save_file = "{}/model.{}.pth".format(checkpoint_dir, epoch)
            torch.save(state, save_file)

            writer.add_scalar("Valid/acc", val_acc, epoch)
            writer.add_scalar("Test01/acc", test01_acc, epoch)
            writer.add_scalar("Test02/acc", test02_acc, epoch)
            writer.add_scalar("Test/acc", test_acc, epoch)

            output = ("Test Epoch {}: val_acc={:.4f}, "
                  "test01_acc={:.4f}, test02_acc={:.4f}, test_acc={:.4f} \n best epoch:{}, best_acc={:.4f}".format(
                epoch, val_acc, test01_acc, test02_acc, test_acc, best_epoch, best_acc))
            print(output)
            log_training.write(output+'\n')

if __name__ == '__main__':
    main()
