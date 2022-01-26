import argparse
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dataloaders import custom_transforms as tr
from dataloaders import pascal
from dataloaders import utils
from model_utils import load_model

composed_transforms_ts = transforms.Compose([
    tr.FixedResize(size=(512, 512)),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])

photo_path = '/'


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int, help='设置训练或测试时使用的显卡')
    parser.add_argument('--backbone', default='resnet', type=str, help='预训练模型, resnet or xception')
    parser.add_argument('--batch', default=12, type=int, help='处理多少张图片')
    parser.add_argument('--model_name', default='unet', type=str, help='需要训练的模型架构')
    return parser.parse_args()


def show(imgs, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig(path)


if __name__ == '__main__':
    args = set_args()
    if args.model_name.lower() == 'deeplab':
        model_path = 'model/deeplabv3plus-resnet.pth'
    elif args.model_name.lower() == 'unet':
        model_path = 'model/Unet.mdl'
    else:
        print('model name {} not available.'.format(args.model_name))
        raise NotImplementedError

    assert args.batch % 4 == 0, f'batchsize必须是4的倍数'

    net = load_model(args, state=True)
    net.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    testloader = DataLoader(voc_val, batch_size=args.batch, shuffle=False, num_workers=0)
    device = 'cuda:{}'.format(args.device) if args.device > 0 else 'cpu'
    net.to(device)
    net.eval()
    epoch = random.randint(0, len(testloader) - 1)
    # 随机选择一个batch进行测试
    for num, (sample_batched, _) in enumerate(testloader):

        if num != epoch:
            continue
        inputs, labels = sample_batched['image'], sample_batched['label']
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net.forward(inputs)

        grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels, 1).detach().cpu().numpy()), 4,
                               normalize=False, range=(0, 255))
        show(grid_image, './result/{}_label_{}.png'.format(args.model_name, num))
        grid_image = make_grid(inputs.clone().cpu().data, 4, normalize=True)

        show(grid_image, './result/{}_ori_{}.png'.format(args.model_name, num))

        grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs, 1)[1].detach().cpu().numpy()), 4,
                               normalize=False,
                               range=(0, 255))
        show(grid_image, './result/{}_pre_{}.png'.format(args.model_name, num))
        torch.cuda.empty_cache()
        break
    print('结果已经保存在 ./result 文件夹中')
