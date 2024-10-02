#!/bin/bash
#CUDA_VISIBLE_DEVICES=1,2,3 nohup python neural_style.py -num_iterations 1 -init image -style_dir /home/hfle/dataset/website_for_replace/source_stylized/AdaIN_stylized
# -content_dir /home/hfle/dataset/website_for_replace/source_content -out_dir /home/hfle/dataset/website_for_replace/source_stylized/test_stylized > loss_source_AdaIN.log 2>&1 &
gpu="0,1,2,3"
root_dir="/home/hfle/dataset/website_for_replace/"
content_dir=$root_dir"extractade_3/"
save_dir="/home/hfle/dataset/website_for_replace/source_stylized/test_stylized"

for method in 'AdaIN' 'Artflow' 'CAST' 'Line' 'Manifold' 'NST' 'SANet' 'StyTR2' 'WCT' 'Diffusion'
do
	stylized_dir=$root_dir"ade3_stylized/"$method"_stylized/"
	echo $content_dir
	echo $stylized_dir
	CUDA_VISIBLE_DEVICES=$gpu nohup python neural_style.py -num_iterations 1 -init image -content_dir $content_dir     \
                        -style_dir $stylized_dir                       \
	                      -out_dir $save_dir  > ade3_contentloss_$method.log 2>&1 & \
                        # --use_mask
done