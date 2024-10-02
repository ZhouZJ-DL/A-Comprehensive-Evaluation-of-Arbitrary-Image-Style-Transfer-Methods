#!/bin/bash
gpu=0
root_dir="/home/hfle/new_website_source/"

for match in 'AdaIN_Diffusion' 'Artflow_Diffusion' 'CAST_Diffusion' 'Line_Diffusion' 'Manifold_Diffusion' 'NST_Diffusion' 'SANet_Diffusion' 'StyTR2_Diffusion' 'WCT_Diffusion'
do
	save_dir=$root_dir$match"/AdaIN_stylized2"
	content_dir=$root_dir$match"/content_new"
	style_dir=$root_dir$match"/style"
	echo $save_dir
	echo $content_dir
	echo $style_dir
	CUDA_VISIBLE_DEVICES=$gpu python test.py --content_dir $content_dir                          \
                        --style_dir $style_dir                              \
	                      --output $save_dir \
                        # --use_mask
done