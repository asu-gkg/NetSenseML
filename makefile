run-gpt2:
	HF_DATASETS_OFFLINE=1 python3 

run-dist-vgg16:
	python3 app/dist_vgg16.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4001 --log_file ./results/benchmark_vgg16_${rank}.log

run-dist-vgg16-topk:
	python3 app/dist_vgg16_topk.py --rank $(rank) --world_size $(world_size) --dist_url tcp://192.168.1.170:4003 --log_file ./results/benchmark_vgg16_topk_${rank}.log