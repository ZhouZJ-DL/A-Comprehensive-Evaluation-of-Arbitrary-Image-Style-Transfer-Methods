import numpy
import math
import cv2
import os
import argparse
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-o","--original", required=True, type=str, help="require original file path")
ap.add_argument("-s","--contrast", required=True, type=str, help="require contrast file path")
ap.add_argument("-i","--interpolation", type=str, default=cv2.INTER_AREA)
ap.add_argument("-v","--video", action='store_true')
ap.add_argument('--outf', default='logs.txt', type=str, required=True)
args = ap.parse_args()

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

def compute_psnr(img1, img2):
    if args.video:
        dir_len = len(os.walk(args.original).__next__()[2])
        print("total file number:{}".format(dir_len))
        total = 0

        for i in tqdm(range(1, dir_len)):
            try:
                o_image = "%05d" % i +".png"
                c_image = "%05d" % i +".png"

                original = cv2.imread(args.original + o_image)
                contrast = cv2.imread(args.contrast + c_image)

                o_height, o_width, o_channel = original.shape
                contrast = cv2.resize(contrast, dsize=(o_width,o_height), interpolation=cv2.INTER_AREA)

                total += psnr(original, contrast)

            except Exception as e:
                    print(str(e) + ": Total count mismatch!!!!")

            #if(i%100 == 0):
            #     print("PSNR: {}".format(psnr(original, contrast)))

        video_psnr_mean = total / dir_len
        print("Video PSNR Mean : {}".format(video_psnr_mean))

    else:
        original = cv2.imread(img1)
        contrast = cv2.imread(img2)

        o_height, o_width, o_channel = original.shape
        contrast = cv2.resize(contrast, dsize=(o_width, o_height), interpolation=args.interpolation)

        print("{}_{} PSNR Mean: {}".format(os.path.basename(img1), os.path.basename(img2), psnr(original, contrast)))
        f.writelines("{}_{} PSNR Mean: {}\n".format(os.path.basename(img1), os.path.basename(img2), psnr(original, contrast)))

        return psnr(original, contrast)


content_imgs = os.listdir(args.original)
stylized_imgs = os.listdir(args.contrast)
content_imgs.sort()
stylized_imgs.sort()
sum = 0

f = open(args.outf, 'a')
for index in range(len(content_imgs)):
    psnr_score = compute_psnr(args.original + '/' + content_imgs[index], args.contrast + '/' + stylized_imgs[index])
    sum += psnr_score
print("{} images avg PSNR: {}".format(len(content_imgs), sum / len(content_imgs)))
f.writelines("{} images avg SSIM: {}\n".format(len(content_imgs), sum / len(content_imgs)))
