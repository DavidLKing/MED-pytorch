#!/usr/bin/env bash

for i in configs/artif-final-0*
	do 	
		exp=`basename $i`
		new_exp=${exp}-batches
		cp -r artif-final-template ${new_exp}
		for file in ${new_exp}/*
			do
                echo "file ${file}"
                echo "file basename `basename ${file}`"
                basefile=`basename ${file} .sbatch`
                works=`echo ${exp} | sed -e "s/artif-final/${basefile}/g"`
                echo "works ${works}"
				echo "exp ${exp}"
				echo "new_exp ${new_exp}"
				echo "file ${file}"
				sed -i "s/artif-final-0.50-50/${works}/g" ${file}
				sed -i "s/exp_string/${works}/g" ${file}
			done
	done