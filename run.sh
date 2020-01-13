############# office-31 partial ###############

#CUDA_VISIBLE_DEVICES=5 python train.py --dataset="office_31" --source="webcam" --target="amazon" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset="office_31" --source="dslr" --target="amazon" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=6 python train.py --dataset="office_31" --source="amazon" --target="webcam" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset="office_31" --source="amazon" --target="dslr" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 36 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=4 train.py --dataset="office_31" --source="amazon" --target="dslr" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=4 train.py --dataset="office_31" --source="amazon" --target="dslr" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31_partial"  --batch_size 32 --gpu_ids 0


############# office-31 ###############
#CUDA_VISIBLE_DEVICES=0 python train.py  --dataset="office_31" --source="dslr" --target="amazon" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 28 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=0 python train.py  --dataset="office_31" --source="webcam" --target="amazon" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 28 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=5  python train.py  --dataset="office_31" --ssource="amazon" --target="dslr" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=0  python train.py  --dataset="office_31" --source="amazon" --target="webcam" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 36 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=5 python train.py  --dataset="office_31" --source="dslr" --target="webcam" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 28 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=5 python train.py  --dataset="office_31" --source="webcam" --target="dslr" --train_dir="/users/leo/Datasets/DA/office31" --test_dir="/users/leo/Datasets/DA/office31"  --batch_size 32 --gpu_ids 0

############# office-home partial ###############

#CUDA_VISIBLE_DEVICES=5 python train.py --dataset "office_home" --source="Art.txt" --target="Clipart_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
CUDA_VISIBLE_DEVICES=4 python train.py --dataset "office_home" --source="Art.txt" --target="Product_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=6 python train.py --dataset "office_home" --source="Art.txt" --target="Real_World_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=4 python train.py --dataset "office_home" --source="Clipart.txt" --target="Art_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=3 python train.py --dataset "office_home" --source="Clipart.txt" --target="Product_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=2 python train.py --dataset "office_home" --source="Clipart.txt" --target="Real_World_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset "office_home" --source="Product.txt" --target="Art_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=6 python train.py --dataset "office_home" --source="Product.txt" --target="Clipart_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=3 python train.py --dataset "office_home" --source="Product.txt" --target="Real_World_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset "office_home" --source="Real_World.txt" --target="Art_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=5 python train.py --dataset "office_home" --source="Real_World.txt" --target="Clipart_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=7 python train.py --dataset "office_home" --source="Real_World.txt" --target="Product_partial.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0


############# office-home  ###############
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset "office_home" --source="Product.txt" --target="Real_World.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset "office_home" --source="Product.txt" --target="Clipart.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=2 python train.py --dataset "office_home" --source="Real_World.txt" --target="Art.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=3 python train.py --dataset "office_home" --source="Real_World.txt" --target="Clipart.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset "office_home" --source="Art.txt" --target="Clipart.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=7 python train.py --dataset "office_home" --source="Art.txt" --target="Real_World.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0
#CUDA_VISIBLE_DEVICES=7 python train.py --dataset "office_home" --source="Art.txt" --target="Product.txt" --train_dir="/users/leo/Codes/MOCA/data/office_home" --test_dir="/users/leo/Codes/MOCA/data/office_home"  --batch_size 32 --gpu_ids 0


