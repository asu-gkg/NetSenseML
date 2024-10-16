run-bert:
	python3 app/train_bert_large_bugs_prediction.py --log_file ./results/benchmark_bert.log

run-dist-vgg16:
	python3 app/dist_vgg16.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4003 --log_file ./results/benchmark_vgg16_${rank}.log

run-dist-vgg16-ours:
	python3 app/dist_vgg16_ours.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4003 --log_file ./results/benchmark_vgg16_ours1_${rank}.log

run-dist-vgg16-1g:
	python3 app/dist_vgg16_1g_allreduce.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4004 --log_file ./results/benchmark_vgg16_${rank}.log

run-dist-vgg16-topk:
	python3 app/dist_vgg16_topk.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4005 --log_file ./results/benchmark_vgg_topk_${rank}.log

run-dist-gpt2-ours:
	python3 app/train_gpt2_dist_ours.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4005 --log_file ./results/benchmark_gpt2_${rank}.log

nas:
	sudo apt install nfs-common
	sudo mount 10.4.5.140:/home/asu/PycharmProjects/NetSenseML /mnt/nfs

run-dist-gpt2-allreduce:
	python3 app/train_gpt2_dist_allreduce.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4005 --log_file ./results/benchmark_gpt2_${rank}.log

run-dist-gpt2-topk:
	torchrun --nnodes=$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154 --master_port=8003 app/train_gpt2_dist_topk.py --world_size $(world_size) --rank $(rank) --log_file ./results/benchmark_gpt2_topk_${rank}.log


# run-dist-resnet18:
# 	python3 app/dist_resnet18.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:8003  --log_file ./results/benchmark_resnet18_${rank}.log


run-dist-resnet18_baseline:
	torchrun --nnodes=$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154 --master_port=8003 app/dist_resnet18_baseline.py --world_size $(world_size) --rank $(rank) --log_file ./results/benchmark_resnet18_baseline_${rank}.log

run-dist-resnet18-topk:
	torchrun --nnodes=2:$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154  --master_port=8003 app/dist_resnet18_topk.py --log_file ./results/benchmark_resnet18_topk_${rank}.log

run-dist-resnet18:
	torchrun --nnodes=2:$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154  --master_port=8003 app/dist_resnet18.py --log_file ./results/benchmark_dist_resnet18_${rank}.log