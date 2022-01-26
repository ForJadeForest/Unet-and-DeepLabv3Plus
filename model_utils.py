import os
import timeit

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from tqdm import tqdm

# Custom includes
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet, Unet


# Tensorboard include


def write_photo(writer, inputs, outputs, labels, global_step):
    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image('Image', grid_image, global_step)
    grid_image = make_grid(
        utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
        range=(0, 255))
    writer.add_image('Predicted label', grid_image, global_step)
    grid_image = make_grid(
        utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False,
        range=(0, 255))
    writer.add_image('Groundtruth label', grid_image, global_step)


def train_epoch(model, global_step, args, optimizer, writer, trainloader, model_name, criterion, epoch, save_dir, scheduler):
    start_time = timeit.default_timer()
    running_loss_tr = 0
    num_img_tr = len(trainloader)
    gs = global_step
    aveGrad = 0
    # 修改学习率

    model.train()
    bar = tqdm(trainloader, desc="Loss: X.XXX", disable=False, ncols=100)
    for ii, (sample_batched, _) in enumerate(bar):

        inputs, labels = sample_batched['image'], sample_batched['label']
        # Forward-Backward of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)
        gs += inputs.data.shape[0]

        if args.device >= 0:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        if args.model_name == 'unet':
            target = labels.squeeze(1).long()
            loss = criterion(outputs, target)
        else:
            loss = criterion(outputs, labels, size_average=True, batch_average=True)
        running_loss_tr += loss.item()

        # Print stuff
        if ii % num_img_tr == (num_img_tr - 1):
            running_loss_tr = running_loss_tr / num_img_tr
            writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * args.train_batch + inputs.data.shape[0]))
            print('Loss: %f' % running_loss_tr)
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # Backward the averaged gradient
        loss /= args.aveGrad
        bar.set_description(desc='Loss={}'.format(round(loss.item(), 3)))
        loss.backward()
        aveGrad += 1

        # Update the weights once in p['nAveGrad'] forward passes
        if aveGrad % args.aveGrad == 0:
            writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('lr', scheduler.get_last_lr()[0], ii + num_img_tr * epoch)
            aveGrad = 0

        # Show 10 * 3 images results each epoch
        if ii % (num_img_tr // 10) == 0:
            write_photo(writer, inputs, outputs, labels, gs)
    scheduler.step()
    # Save the model
    if (epoch % args.save_model) == args.save_model - 1:
        torch.save(model.state_dict(), os.path.join(save_dir, 'models', model_name + '_epoch-' + str(epoch) + '.pth'))
        print(
            "Save model at {}\n".format(os.path.join(save_dir, 'models', model_name + '_epoch-' + str(epoch) + '.pth')))
    return gs


def test_epoch(model, testloader, writer, args, criterion, epoch):
    metrics = utils.IOUMetric(21)
    running_loss_ts = 0
    num_img_ts = len(testloader)
    model.eval()
    for ii, (sample_batched, _) in enumerate(testloader):
        inputs, labels = sample_batched['image'], sample_batched['label']

        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        if args.device >= 0:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model.forward(inputs)

        predictions = torch.max(outputs, 1)[1]
        if args.model_name == 'unet':
            target = labels.squeeze(1).long()
            loss = criterion(outputs, target)
        else:
            loss = criterion(outputs, labels, size_average=True, batch_average=True)
        running_loss_ts += loss.item()

        metrics.add(predictions, labels)
        # Print stuff
    acc, acc_cls, iou, miou, fwavacc = metrics.evaluate_dir()
    running_loss_ts = running_loss_ts / num_img_ts

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(testloader) * args.test_batch))
    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
    writer.add_scalar('data/test_miour', miou, epoch)
    writer.add_scalar('data/test_acc', acc, epoch)
    writer.add_scalar('data/test_acc_cls', acc_cls, epoch)
    writer.add_scalar('data/test_fwavacc', fwavacc, epoch)
    print('Loss: %f' % running_loss_ts)
    print('MIoU: %f\n' % miou)


def load_model(args, save_dir=None, state=False):
    if args.model_name.lower() == 'unet':
        net = Unet.UNet(3, 21)
        criterion = torch.nn.CrossEntropyLoss()
        model_name = 'unet' + '-voc'
        if state:
            return net
        if args.resume_epoch == 0:
            print("Training unet from scratch...")
        else:
            print("Initializing weights from: {}...".format(
                os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')))
            net.load_state_dict(
                torch.load(
                    os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth'),
                    map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU
        if args.device >= 0:
            torch.cuda.set_device(device=args.device)
            net.cuda()
    else:
        if args.backbone == 'xception':
            net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)
        elif args.backbone == 'resnet':
            net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)
        else:
            raise NotImplementedError
        if state:
            return net
        model_name = 'deeplabv3plus-' + args.backbone + '-voc'
        criterion = utils.cross_entropy2d

        if args.resume_epoch == 0:
            print("Training deeplabv3+ from scratch...")
        else:
            print("Initializing weights from: {}...".format(
                os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')))
            net.load_state_dict(
                torch.load(
                    os.path.join(save_dir, 'models', model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth'),
                    map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU

        if args.device >= 0:
            torch.cuda.set_device(device=args.device)
            net.cuda()
    return net, criterion, model_name
