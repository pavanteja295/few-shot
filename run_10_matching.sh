#!/bin/bash
# counter=1

for value in {1..10}
do
CUDA_VISIBLE_DEVICES=1 python -m experiments.matching_nets --dataset miniImageNet --fce True  --k-test 5 --n-test 1 --distance l2 --exp_name 'run'$value
done
echo All done

