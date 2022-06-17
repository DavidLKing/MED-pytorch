#!/usr/bin/env bash

for dropout in 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
    do for network in 50 100 150 200 250 300
        do newname=artif-final-${dropout}-${network}
           cp -r artif-final ${newname}
                 for file in ${newname}/*.yml
                    do
                        sed -i "s/-0.5-50-testing/-${dropout}-${network}-testing/g" ${file}
                        sed -i "s/embed: 150/embed: ${network}/g" ${file}
                        sed -i "s/dropout: 0.5/dropout: ${dropout}/g" ${file}
                    done
        done
    done
