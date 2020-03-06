#!/bin/bash
# counter=1
#k-way = (2, 5, 15)
#n-shot = (1, 5)

for k in 2 5 15; 
do 
	for n in 1 5; 
	do
	       
	       n_trains_4=$(( $k*4))
		for n_train in $n_trains_4;
	   do
CUDA_VISIBLE_DEVICES=0 python -m experiments.maml --dataset fashion-dataset --order 1 --n $n --k $k --q 5 --meta-batch-size 4 --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400

#	   echo $k $n $n_train; 
	   done; 
       done
done
#for value in {1..10}
#do
done
#echo All done

