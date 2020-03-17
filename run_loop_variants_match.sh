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
		   CUDA_VISIBLE_DEVICES=1 python -m experiments.matching_nets --dataset fashion-dataset --fce True --k-test $k  --n-test $n --distance l2 --exp_name fashion_loop_match_l2_distance_new --q-train $((11 - $n )) --q-test $(( 11 - $n )) --k-train $k --n-train $n
#	   echo $k $n $n_train; 
	   done; 
       done
done
#for value in {1..10}
#do
done
#echo All done

