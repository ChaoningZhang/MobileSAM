CUDA_VISIBLE_DEVICES=0 python Inference.py  \
    --img_path './test_images/' \
    --yolo_model_path './weight/PromtModel.pt' \
    --output_dir './' \
    --model_type 'efficientvit_l2' \