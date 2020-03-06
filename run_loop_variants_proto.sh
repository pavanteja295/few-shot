#!/bin/bash
# counter=1
#k-way = (2, 5, 15)
#n-shot = (1, 5)

for k in 15;
do 
	for n in 1 5; 
	do
	       
	       n_trains_4=$(( $k*2))
		for n_train in $n_trains_4;
	   do
		   CUDA_VISIBLE_DEVICES=0 python -m experiments.proto_nets --dataset fashion-dataset  --k-test $k  --n-test $n  --k-train $n_train --n-train $n  --q-train $((10 - $n)) --exp_name 'fashion_loop_new_proto' --q-test $((10 - $n)) 
#	   echo $k $n $n_train; 
	   done; 
       done
done
#for value in {1..10}
#do
done
#echo All done

