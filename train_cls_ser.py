
import torch
import numpy as np
import random
torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=False # cudnn
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
import torch.nn.functional as F
from tensorboardX import SummaryWriter

def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)
            label = label.unsqueeze(2).unsqueeze(3)

            x = model(img)
            x = F.adaptive_avg_pool2d(x, (1,1))	
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss:', val_loss_meter.pop('loss'))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_ser", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_cls_ser", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./log', type=str)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    tblogger = SummaryWriter(args.tblog_dir)	
    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = voc12.data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
                                             transform=transforms.Compose([
                        np.asarray,
                        model.normalize,
                        imutils.CenterCrop(500),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls_ser"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "network.vgg16_cls_ser"
        import network.vgg16d
        weights_dict = network.vgg16d.convert_caffe_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss','loss_cls','loss_cls_s','loss_r')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1]
            label = pack[2].cuda(non_blocking=True)
            label = label.unsqueeze(2).unsqueeze(3)

            cam = model(img)
            N,C,H,W = cam.size()
            predicts = F.adaptive_avg_pool2d(cam, (1,1))	
            loss_cls = F.multilabel_soft_margin_loss(predicts, label)
            branch_rate = 0.3
            img_s = F.interpolate(img, scale_factor=branch_rate,mode='bilinear')
            cam_s = model(img_s)
            Ns,Cs,Hs,Ws = cam_s.size()
            predicts_s = F.adaptive_avg_pool2d(cam_s, (1,1))
            loss_cls_s = F.multilabel_soft_margin_loss(predicts_s, label)
            
            
            cam_sn = F.relu(cam_s)
            cam_sn_max = torch.max(cam_sn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_sn = F.relu(cam_sn-1e-5, inplace=True)/cam_sn_max
            cam_sn = cam_sn * label

            cam_r = F.interpolate(cam, scale_factor=branch_rate, mode='bilinear')
            cam_rn = F.relu(cam_r)
            cam_rn_max = torch.max(cam_rn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_rn = F.relu(cam_rn-1e-5, inplace=True)/cam_rn_max
            cam_rn = cam_rn * label


            loss_r = torch.mean(torch.pow(cam_sn - cam_rn, 2))
            loss = (loss_cls/2 + loss_cls_s/2) + loss_r
 
            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_cls_s': loss_cls_s.item(), 'loss_r':loss_r.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%10 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f %.4f %.4f %.4f' % (avg_meter.get('loss','loss_cls','loss_cls_s','loss_r')),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()


                img_8 = img[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2,0,1))
                h = H//4; w = W//4
                p = F.interpolate(cam,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p_s = F.interpolate(cam_s,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                bg_score = np.zeros((1,h,w),np.float32)
                p = np.concatenate((bg_score,p), axis=0)
                p_s = np.concatenate((bg_score,p_s), axis=0)
                bg_label = np.ones((1,1,1),np.float32)
                l = label[0].detach().cpu().numpy()
                l = np.concatenate((bg_label,l),axis=0)
                image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                CLS, CAM, CLS_crf, CAM_crf = visualization.generate_vis(p, l, image, func_label2color=visualization.VOClabel2colormap)
                CLS_s, CAM_s, CLS_crf_s, CAM_crf_s = visualization.generate_vis(p_s, l, image, func_label2color=visualization.VOClabel2colormap)
                loss_dict = {'loss':loss.item(), 
                             'loss_cls':loss_cls.item(),
                             'loss_cls_s':loss_cls_s.item(),
                             'loss_r':loss_r.item()}	
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_image('Image', input_img, itr)
                tblogger.add_image('CLS', CLS, itr)
                tblogger.add_image('CLS_s', CLS_s, itr)
                tblogger.add_images('CAM', CAM, itr)
                tblogger.add_images('CAM_s', CAM_s, itr)
        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
