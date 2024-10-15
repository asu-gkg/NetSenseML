run-gpt2:
	HF_DATASETS_OFFLINE=1 python3 

run-dist-vgg16:
	python3 app/dist_vgg16.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4003 --log_file ./results/benchmark_vgg16_ours_${rank}.log

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
	python3 app/train_gpt2_dist_topk.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4005 --log_file ./results/benchmark_gpt2_topk_${rank}.log