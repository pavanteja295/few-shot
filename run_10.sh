#!/bin/bash
# counter=1

for value in {1..10}
do
CUDA_VISIBLE_DEVICES=1 python -m experiments.proto_nets --dataset fashion-dataset  --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 10 --exp_name 'run'$value
done
echo All done

