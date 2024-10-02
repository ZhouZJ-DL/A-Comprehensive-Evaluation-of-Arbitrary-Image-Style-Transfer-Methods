# coding=UTF-8
import os
import time
import argparse
import torch
from torchvision.utils import save_image
from lib.models.base_models import Encoder, Decoder
from lib.core.mast import MAST
from lib.core.config import get_cfg
from lib.dataset.Loader import single_load
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def multi_level_test(encoder, decoders, layers, mast, style_weight, args, device, cpath, spath):
    print(f'processing [{cpath}] and [{spath}]...')
    c_tensor = single_load(cpath, resize=args.resize)
    s_tensor = single_load(spath, resize=args.resize)
    c_tensor = c_tensor.to(device)
    s_tensor = s_tensor.to(device)
    for layer_name in layers:
        with torch.no_grad():
            cf = encoder(c_tensor)[layer_name]
            sf = encoder(s_tensor)[layer_name]
            csf = mast.transform(cf, sf, args.content_seg_path, args.style_seg_path, args.seg_type)
            csf = style_weight * csf + (1 - style_weight) * cf
            out_tensor = decoders[layer_name](csf, layer_name)
        c_tensor = out_tensor
    os.makedirs(args.output_dir, exist_ok=True)
    c_basename = os.path.splitext(os.path.basename(cpath))[0]
    s_basename = os.path.splitext(os.path.basename(spath))[0]
    output_path = os.path.join(args.output_dir, f'{c_basename}_{s_basename}.png')
    save_image(out_tensor, output_path, nrow=1, padding=0)
    print(f'[{output_path}] saved...')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description='Artistic Test')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml',
                        help='config path')
    parser.add_argument('--content_path', '-c', type=str, default='/home/hfle/MachineLearning/Manifold/data/custom/content',
                        help='path of content image')
    parser.add_argument('--style_path', '-s', type=str, default='/home/hfle/MachineLearning/Manifold/data/custom/style',
                        help='path of style image')
    parser.add_argument('--output_dir', type=str, default='/home/hfle/MachineLearning/Manifold/data/custom/stylized',
                        help='the output dir to save the output image')
    parser.add_argument('--content_seg_path', type=str, default=None,
                        help='content_seg_path')
    parser.add_argument('--style_seg_path', type=str, default=None,
                        help='style_seg_path')
    parser.add_argument('--seg_type', type=str, default='dpst',
                        help='the type of segmentation type, [dpst, labelme]')
    parser.add_argument('--resize', type=int, default=-1,
                        help='resize the image, -1: no resize, x: resize to [x, x]')

    args = parser.parse_args()
    cfg = get_cfg(cfg_path=args.cfg_path)
    assert cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT is False, 'cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT must be False'

    decoders_path = {
        'r11': cfg.TEST.MODEL.DECODER_R11_PATH,
        'r21': cfg.TEST.MODEL.DECODER_R21_PATH,
        'r31': cfg.TEST.MODEL.DECODER_R31_PATH,
        'r41': cfg.TEST.MODEL.DECODER_R41_PATH,
        'r51': cfg.TEST.MODEL.DECODER_R51_PATH
    }

    device = torch.device('cuda') if torch.cuda.is_available() and cfg.DEVICE == 'gpu' else torch.device('cpu')
    # set the model
    print(f'Building models...')
    encoder = Encoder()
    encoder.load_state_dict(torch.load(cfg.TEST.MODEL.ENCODER_PATH, map_location=device))
    encoder = encoder.to(device)
    layers = cfg.TEST.ARTISTIC.LAYERS.split(',')
    decoders = {}
    for layer_name in layers:
        decoder = Decoder(layer=layer_name)
        decoder.load_state_dict(torch.load(decoders_path[layer_name], map_location=device))
        decoder = decoder.to(device)
        decoders[layer_name] = decoder
    print(f'Finish!')

    mast = MAST(cfg)

    style_weight = cfg.TEST.ARTISTIC.STYLE_WEIGHT

    cpaths = os.listdir(args.content_path)
    spaths = os.listdir(args.style_path)
    cpaths = sorted(cpaths)
    spaths = sorted(spaths)
    start = time.time()
    for index in range(len(cpaths)):
        cpath = os.path.join(args.content_path, cpaths[index])
        spath = os.path.join(args.style_path, spaths[index])
        multi_level_test(encoder, decoders, layers, mast, style_weight, args, device, cpath, spath)
    end = time.time()
    print('start: {}'.format(start))
    print('end: {}'.format(end))
    print('Running time is: %f' % (end - start))
    '''
    for croot, _, cfnames in sorted(os.walk(args.content_path, followlinks=True)):
        for cfname in cfnames:
            if is_image_file(cfname):
                cpath = os.path.join(croot, cfname)

            for sroot, _, snames in sorted(os.walk(args.style_path, followlinks=True)):
                for sname in snames:
                    if is_image_file(sname):
                        spath = os.path.join(sroot, sname)
                        multi_level_test(encoder, decoders, layers, mast, style_weight, args, device, cpath, spath)
    '''

if __name__ == '__main__':
    main()
