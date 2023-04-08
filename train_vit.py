import os
import math
import argparse
import time
import datetime
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from timm.utils import accuracy, AverageMeter

# from vit.model import *
import vit.model as model_ViT
from vit.datasets.cifar100_dataset import load_data
from vit.utils.utils import *

# from my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224_in21k as create_model
# from utils import read_split_data

def parse_option():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--name', required=True, type=str, help='create model name')
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--device_name', type=str, default='torch.cuda.get_device_name(0)')
    parser.add_argument('--pretrain_dir', type=str, default='./vit/pretrain')
    # data
    parser.add_argument('--data', default='cifar100', type=str, help='inat21_mini|inat21_full')
    parser.add_argument('--data_dir', type=str, default="./data/cifar100")
    parser.add_argument('--num_classes', type=int, default=100)
    # train
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_workers', default=8, type=int)
    # model
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--pretrain', type=str, default='vit_base_patch16_224_in21k.pth')
    parser.add_argument('--model_file', type=str, default='model_ViT')
    parser.add_argument('--model_name', type=str, default='vit_base_patch32_224_ImageNet21k', help='model type in detail')
    parser.add_argument('--resume', type=str, default='Latest', help='path to Latest checkpoint (default: none)')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=True)
    args = parser.parse_args()
    args.device_name = str(torch.cuda.get_device_name(0))
    return args


def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
    
    # get datasets
    train_loader, val_loader, num_iter = load_data(args)

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get net
    logger.info(f"Creating model:{args.model_file} -> {args.model_name}")
    model = model_ViT.__dict__[args.model_file].__dict__[args.model_name](logger, args)  # 从mode_file中找到对应model_name的模型
    model.cuda()
    model = torch.nn.DataParallel(model)
    # logger.info(model)  # 打印网络结构

    # get criterion 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).cuda()
    # get optimizer 优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # tb_writer = SummaryWriter()
    if args.pretrain != "":
        assert os.path.exists(args.pretrain), "pretrain file: '{}' not exist.".format(args.pretrain)
        weights_dict = torch.load(args.pretrain)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    # 只训练最后的MLP权重，前面部分全部冻结
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    start_epoch = 1
    max_accuracy = 0.0
    '''
    # 如果之前有与训练权重，直接作为基础恢复训练
    if args.resume:
        if args.resume in ['Best', 'Latest']:
            args.resume = os.path.join(args.path_log, '%s-%s.pth' % (args.name, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            state_dict = torch.load(args.resume)
            if 'model' in state_dict:
                start_epoch = state_dict['epoch'] + 1
                model.load_state_dict(state_dict['model'],strict=False)
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
            else:
                model.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
            if 'max_accuracy' in state_dict:
                max_accuracy = state_dict['max_accuracy']
            acc1_, acc5_, loss_ = validate(val_loader, model, loss_function, state_dict['epoch'], logger, args)
            max_accuracy = max(max_accuracy, acc1_)
            logger.info(f'Max accuracy: {max_accuracy:.4f}%')
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        epoch = start_epoch - 1
        acc1, acc5, outputs = validate(val_loader, model, loss_function, epoch, logger, args)
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)
        return
    '''

    best_acc1 = 0.0
    args.time_sec_tot = 0.0
    args.start_epoch = start_epoch
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.stop_epoch + 1):
        # train
        train_loss, train_acc = train_one_epoch_local_data(train_loader, model, loss_function, optimizer, epoch, logger, args)
        save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='Latest')
        scheduler.step()  # 更新lr
        # validate
        logger.info(f"**********Latest val***********")
        val_loss, val_acc = validate(val_loader, model, loss_function, epoch, logger, args)

        # 保存最好效果
        max_accuracy = max(max_accuracy, val_acc)
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')
        if val_acc > best_acc1:
            best_acc1 = val_acc
            save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='Best')
        logger.info('Exp path: %s' % args.path_log)

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    # 总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


# def train_one_epoch_local_data(model, optimizer, data_loader, device, epoch):
def train_one_epoch_local_data(train_loader, model, loss_function, optimizer, epoch, logger, args):
    model.train()
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()  # 自动混合精度训练
    
    accu_loss = torch.zeros(1)  # 累计损失
    accu_num = torch.zeros(1)   # 累计预测正确的样本数
    sample_num = 0

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for iter, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sample_num += images.shape[0]

        output = model(images)
        loss = loss_function(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, target).sum()

        loss.backward()
        accu_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()

        # 储存batch_time和loss
        batch_time.update(time.time() - end)  # 记录每次迭代batch所需时间
        end = time.time()
        loss_meter.update(loss.item(), output.size(0))  # output.size(0)
        # log输出训练参数
        if iter % 300 == 0:
            etas = batch_time.avg * (num_steps - iter)
            # lr = optimizer.param_groups[0]['lr']
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{args.stop_epoch}][{iter}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc@1: {acc1.item():.4f}\t'
                f'acc@5: {acc5.item():.4f}\t')
            # logger.info('\t'.join(outputs))
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return accu_loss.item() / (iter + 1), accu_num.item() / sample_num


@torch.no_grad()
def validate(val_loader, model, loss_function, epoch, logger, args):
# def validate(model, data_loader, device, epoch):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    model.eval()

    accu_num = torch.zeros(1)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1)  # 累计损失
    sample_num = 0

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end = time.time()
    for iter, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        sample_num += images.shape[0]

        output = model(images)
        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, target).sum()

        loss = loss_function(output, target)
        accu_loss += loss

        # 更新记录
        acc1, acc5 = accuracy(output, output, topk=(1, 5))
        loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        acc5_meter.update(acc5.item(), output.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # log输出测试参数
        if iter % 200 == 0:
            logger.info(
                f'Test: [{iter}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}') 
    return accu_loss.item() / (iter + 1), accu_num.item() / sample_num


if __name__ == '__main__':
    args = parse_option()
    main(args)