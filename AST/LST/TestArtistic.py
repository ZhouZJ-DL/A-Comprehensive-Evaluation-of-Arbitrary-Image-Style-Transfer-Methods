import os
import torch
import argparse
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.models import encoder3,encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5

import datetime

os.environ['CUDA_VISIBLE_DEVICES']='2' # 改GPU编号

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="/home/hfle/new_website_source/Line_Diffusion/style_new",
                    help='path to style image')
parser.add_argument("--contentPath", default="/home/hfle/new_website_source/Line_Diffusion/content_new",
                    help='path to frames')
parser.add_argument("--outf", default="/home/hfle/new_website_source/Line_Diffusion/Line_stylized2",
                    help='path to transferred images')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument('--loadSize', type=int, default=512,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='crop image size')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize,test=True)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                             batch_size = opt.batchSize,
                                             shuffle = False,
                                             num_workers = 1)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize,test=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = False,
                                           num_workers = 1)

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

index_c = 0
starttime = datetime.datetime.now()
print('start: {}'.format(starttime))

for ci,(content,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    contentV.resize_(content.size()).copy_(content)

    index_s = 0
    for sj,(style,styleName) in enumerate(style_loader):
        if(index_c == index_s) :
            styleName = styleName[0]
            styleV.resize_(style.size()).copy_(style)

            # forward
            with torch.no_grad():
                sF = vgg(styleV)
                cF = vgg(contentV)

                if(opt.layer == 'r41'):
                    feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
                else:
                    feature,transmatrix = matrix(cF,sF)
                transfer = dec(feature)

            transfer = transfer.clamp(0,1)
            vutils.save_image(transfer,'%s/%s_%s.png'%(opt.outf,contentName,styleName),normalize=True,scale_each=True,nrow=opt.batchSize)
            print('Transferred image saved at %s%s_%s.png'%(opt.outf,contentName,styleName))
        index_s += 1
    index_c += 1
endtime = datetime.datetime.now()
print('end: {}'.format(endtime))
print('running time: ', (starttime-endtime).seconds)

