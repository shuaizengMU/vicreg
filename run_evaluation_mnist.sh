nohup python evaluate_mnist.py --pretrained exp_mnist/resnet50.pth --exp_dir ./exp_mnist --lr_head 0.02 --data_dir ./data --epochs 100  > training_mnist.out 2>&1 &