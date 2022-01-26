import argparse
import glob
import os
import socket
from datetime import datetime

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders import custom_transforms as tr
from dataloaders import pascal
from dataloaders import utils
from model_utils import load_model, train_epoch, test_epoch


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int, help='设置训练或测试时使用的显卡')
    parser.add_argument('--change_lr', default=30, type=int, help='多少个epoch改变学习率')
    parser.add_argument('--lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--aveGrad', default=1, type=int, help='梯度积累次数')
    parser.add_argument('--epoch', default=400, type=int, help='梯度积累次数')
    parser.add_argument('--resume_epoch', default=0, type=int, help='如果需要继续训练则修改其到相应的轮数')
    parser.add_argument('--train_batch', default=16, type=int, help='训练的batch size')
    parser.add_argument('--test_batch', default=12, type=int, help='测试的batch size')
    parser.add_argument('--test_epoch', default=1, type=int, help='多少epoch测试一次')
    parser.add_argument('--save_model', default=4, type=int, help='多少epoch保存模型')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.95, type=int, help='momentum')
    parser.add_argument('--backbone', default='resnet', type=str, help='预训练模型, resnet or xception')
    parser.add_argument('--model_name', default='deeplab', type=str, help='需要训练的模型架构')
    parser.add_argument('--gamma', default=0.1, type=float, help='学习率衰减比率')
    return parser.parse_args()



if __name__ == '__main__':
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    args = set_args()
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', '{}run_*'.format(args.model_name))))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', '{}run_*'.format(args.model_name))))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', '{}run_'.format(args.model_name) + str(run_id))

    # Network definition

    if args.resume_epoch != args.epoch:
        # Logging into Tensorboard
        log_dir = os.path.join(save_dir, 'models',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        net, criterion, model_name = load_model(args, save_dir)

        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.change_lr, gamma=args.gamma)
        # Use the following optimizer

        composed_transforms_tr = transforms.Compose([
            tr.RandomSized(512),
            tr.RandomRotate(15),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        composed_transforms_ts = transforms.Compose([
            tr.FixedResize(size=(512, 512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
        voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
        db_train = voc_train

        trainloader = DataLoader(db_train, batch_size=args.train_batch, shuffle=True, num_workers=0)
        testloader = DataLoader(voc_val, batch_size=args.test_batch, shuffle=False, num_workers=0)

        utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), args)

        num_img_ts = len(testloader)
        running_loss_tr = 0.0
        running_loss_ts = 0.0
        aveGrad = 0
        global_step = 0
        print("Training Network")

        # Main Training and Testing Loop
        for epoch in range(args.resume_epoch, args.epoch):
            global_step = train_epoch(net, global_step, args, optimizer, writer, trainloader, model_name, criterion,
                                      epoch, save_dir, scheduler)

            if epoch % args.test_epoch == (args.test_epoch - 1):
                test_epoch(net, testloader, writer, args, criterion, epoch)
            # One testing epoch

        writer.close()
