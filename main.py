import argparse
import builtins
import random
import time
import warnings

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import apex

import torch_pruning as tp
from dali import get_dali_dataloader, get_torch_dataloader
from utils import *

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

best_acc1 = 0


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--schedule",
        default=[15, 30, 45],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by a ratio)",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 0.)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained", default="", type=str, help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--data-path", default="", type=str, metavar="DIR", help="path to dataset"
    )
    parser.add_argument(
        "--save-path", default="", type=str, help="path to save checkpoint"
    )
    parser.add_argument(
        "--dali", action="store_true", help="use dali dataloader",
    )
    parser.add_argument(
        "--apex", action="store_true", help="use apex",
    )

    # distribute
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )

    # prune
    parser.add_argument(
        "--sparse-train", action="store_true", help="sparse training",
    )
    parser.add_argument(
        "--reg", default=1e-5, type=float, help="reg for l1"
    )
    parser.add_argument(
        "--prune", action="store_true", help="prune for model",
    )
    parser.add_argument(
        "--prune-rate", default=0.5, type=float, help="rm channel rate for prune"
    )
    parser.add_argument(
        "--pruned-model", action="store_true", help="load original pruned model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_path and (not os.path.exists(args.save_path)):
        os.mkdir(args.save_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # create model
    if args.pruned_model:
        print("=> load pruned model")
        model = torch.load(args.pretrained, map_location="cpu")
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained and not args.pruned_model:
        if os.path.isfile(args.pretrained):
            args.start_epoch = 0
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.prune:
        print("=> do prune")
        print("=> prune rate:", args.prune_rate)
        model = prune(model, args.prune_rate)
        print("=> do finetune")

    if args.sparse_train:
        print("=> do sparse train")
        print("=> reg:", args.reg)

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.apex:
        print("=> use apex")
        apex.amp.register_float_function(torch, 'sigmoid')
        apex.amp.register_float_function(torch, 'softmax')
        model.cuda()
        model, optimizer = apex.amp.initialize(model, optimizer)

    model_without_dist = None
    if args.distributed:
        print("=> use DDP")
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
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.apex:
                model = apex.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if args.apex:
                model = apex.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_dist = model.module
    elif args.gpu is not None:
        print("=> use gpu id:", args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("=> use DP")
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        model_without_dist = model.module

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dali:
        train_loader = get_dali_dataloader('train', args.data_path, args.batch_size, args.workers, 224)
        # val_loader = get_dali_dataloader('val', args.data_path, args.batch_size, args.workers, 224)
    else:
        train_loader, train_sampler = get_torch_dataloader('train', args.data_path, args.batch_size, args.workers, 224)
        # val_loader = get_torch_dataloader('val', args.data_path, args.batch_size, args.workers, 224)
    val_loader = get_torch_dataloader('val', args.data_path, args.batch_size, args.workers, 224)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and (not args.dali):
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and dist.get_rank() == 0
        ):
            if args.prune:
                state = model_without_dist if (model_without_dist is not None) else model
            else:
                state = {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                }

            save_checkpoint(
                state,
                is_best,
                args
            )

        if args.dali:
            train_loader.reset()
            # val_loader.reset()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1),
    )

    model.train()
    end = time.time()

    if args.dali:
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = data[0]["data"]
            target = data[0]["label"].squeeze().long()

            if args.gpu is not None:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()

            if args.apex:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.sparse_train:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.grad.data.add_(args.reg * torch.sign(m.weight.data))

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    else:
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()

            if args.apex:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.sparse_train:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.grad.data.add_(args.reg * torch.sign(m.weight.data))

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    if args.apex:
        model = model.module

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def prune(model, prune_rate):
    model.cpu()
    inputs = torch.randn(1, 3, 224, 224)

    # 重要性准则
    # imp = tp.importance.MagnitudeImportance(p=1, group_reduction='mean')
    # imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
    # imp = tp.importance.BNScaleImportance()
    imp = tp.importance.GroupNormImportance(p=2)

    # 不剪枝的层
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    print("ignored_layers:", ignored_layers)

    # 剪枝迭代次数
    iterative_steps = 10

    # 剪枝器
    pruner = tp.pruner.MagnitudePruner(
        model,
        inputs,
        global_pruning=False,  # If False, a uniform sparsity will be assigned to different layers.
        importance=imp,  # importance criterion for parameter selection
        iterative_steps=iterative_steps,  # the number of iterations to achieve target sparsity
        ch_sparsity=prune_rate,  # 剪掉的通道百分比
        ignored_layers=ignored_layers,
    )

    base_flops, base_nparams = tp.utils.count_ops_and_params(model, inputs)

    # 每次迭代剪掉一些不重要的通道
    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(model, inputs)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, FLOPs: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_flops / 1e6, macs / 1e6)
        )
        print('*' * 50)

    return model


if __name__ == "__main__":
    main()
