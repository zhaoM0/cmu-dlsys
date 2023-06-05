.PHONY: forward_test backward_test topo_sort_test reverse_autodiff_test softmax_ad_test mlp_ad_test

test:
	python -m pytest -v

forward_test:
	python -m pytest -v -k "forward"

backward_test:
	python -m pytest -l -v -k "backward"

topo_sort_test:
	python -m pytest -k "topo_sort"

reverse_autodiff_test:
	python -m pytest -k "compute_gradient"

softmax_ad_test:
	python -m pytest -k "softmax_loss_ndl"

mlp_ad_test:
	python -m pytest -l -k "nn_epoch_ndl"