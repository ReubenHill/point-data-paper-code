all: experiment1.json experiment2.json experiment3.json experiment4.json

exact-solution.h5: forward.py
	python forward.py --output $@

experiment1.json: inverse.py exact-solution.h5
	python inverse.py \
		--num-observation-wells 6 \
		--num-observation-times 3 \
		--num-trials 30 \
		--std-dev 0.05 \
		--input exact-solution.h5 \
		--output $@

experiment2.json: inverse.py exact-solution.h5
	python inverse.py \
		--num-observation-wells 6 \
		--num-observation-times 3 \
		--num-trials 30 \
		--std-dev 0.01 \
		--input exact-solution.h5 \
		--output $@

experiment3.json: inverse.py exact-solution.h5
	python inverse.py \
		--num-observation-wells 3 \
		--num-observation-times 6 \
		--num-trials 30 \
		--std-dev 0.05 \
		--input exact-solution.h5 \
		--output $@

experiment4.json: inverse.py exact-solution.h5
	python inverse.py \
		--num-observation-wells 3 \
		--num-observation-times 6 \
		--num-trials 30 \
		--std-dev 0.01 \
		--input exact-solution.h5 \
		--output $@

