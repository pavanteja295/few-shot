#!/bin/bash
# counter=1

for value in {1..10}
do
python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15 --exp_name 'run'$value
done
echo All done

