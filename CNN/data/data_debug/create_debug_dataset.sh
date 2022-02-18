#!/bin/bash

cd ./valid
labels=$(ls)

for label in $(ls)
do
	cd ${label}
	n_examples=$(ls | wc -l)
	echo -n "in ${label} there were ${n_examples} examples"
	n_portion=$(expr ${n_examples} / 10)
	i=1
	for file in $(ls)
	do
		if [ "$i" -gt "$n_portion" ]
		then
			rm $file
		fi
		i=$(expr $i + 1)
	done
	echo " and now there are $(ls | wc -l) examples"
	cd ..
done
