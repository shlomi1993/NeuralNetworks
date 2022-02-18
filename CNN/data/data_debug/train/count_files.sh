#!/bin/bash

for label in $(ls)
do
	if [ -d "${label}" ]
	then
		cd ${label}
		n_examples=$(ls -a | wc -l)
		if [ "${n_examples}" -ne "103" ]
		then
			echo "in ${label} there were ${n_examples} examples"
		fi
		cd ..
	fi
done
