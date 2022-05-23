#!/usr/bin/env bash

for i in configs/artif-final-*
	do 	
		exp=`basename $i`
		new_exp=${exp}-batches
		cp -r artif-final-template ${new_exp}
		for file in ${new_exp}/*
			do
				echo "exp ${exp}"
				echo "new_exp ${new_exp}"
				echo "file ${file}"
				sed -i "s/artif-final-0.50-50/${exp}/g" ${file}
			done
	done
