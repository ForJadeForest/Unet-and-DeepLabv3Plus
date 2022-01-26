import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders import custom_transforms as tr
from dataloaders import pascal
from dataloaders import utils
from model_utils import load_model


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int, help='设置训练或测试时使用的显卡')
    parser.add_argument('--model_path',
                        default='model/deeplabv3plus-resnet.pth',
                        type=str, help='模型路径')
    parser.add_argument('--test_batch', default=12, type=int, help='Batch size大小')
    parser.add_argument('--backbone', default='resnet', type=str, help='预训练模型, resnet or xception')
    parser.add_argument('--model_name', default='deeplab', type=str, help='需要训练的模型架构')
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    model_path = args.model_path
    net = load_model(args, state=True)
    net.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU
    net.to('cuda')
    net.eval()
    testBatch = 12
    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    criterion = utils.cross_entropy2d
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)
    total_iou = 0.0
    net.eval()
    running_loss_ts = 0.0
    metrics = utils.IOUMetric(21)
    num_img_ts = len(testloader)
    for ii, sample_batched in enumerate(testloader):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']

        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = net.forward(inputs)
        predictions = torch.max(outputs, 1)[1]
        metrics.add(predictions, labels)
    acc, acc_cls, iou, miou, fwavacc = metrics.evaluate_dir()
    print(' acc:{}, acc_cls:{}, miou:{}, fwavacc:{}'.format(acc, acc_cls, miou, fwavacc))
    metrics.reset()
