#!/bin/bash
# counter=1

for value in {1..10}
do
python -m experiments.maml --dataset miniImageNet --order 1 --n 1 --k 5 --q 5 --meta-batch-size 4 \
	    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400 --exp_name 'run'$value
done
echo All done

