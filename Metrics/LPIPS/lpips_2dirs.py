import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/home/hfle/dataset/website_for_replace/extractade_s10/')
parser.add_argument('-d1','--dir1', type=str, default='/home/hfle/Website/s10_stylized/CAST_stylized2')
parser.add_argument('-o','--out', type=str, default='./logs/CAST2_LPIPS_ade.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if opt.use_gpu:
	loss_fn.cuda()

# crawl directories
f = open(opt.out, 'a')
files1 = os.listdir(opt.dir0)
files2 = os.listdir(opt.dir1)

files1.sort()
files2.sort()

disttotal = 0
for index in range(len(files1)):
	if os.path.exists(os.path.join(opt.dir0,files1[index])):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,files1[index])))  # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,files2[index])))

		if opt.use_gpu:
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0, img1)
		disttotal += dist01
		print('%s_%s: %.3f'%(files1[index], files2[index], dist01))
		f.writelines('%s_%s: %.6f\n'%(files1[index], files2[index], dist01))

print('%dimages average: %.6f\n'%(len(files1), disttotal/52))
f.writelines('%dimages average: %.6f\n'%(len(files1), disttotal/52))
f.close()
