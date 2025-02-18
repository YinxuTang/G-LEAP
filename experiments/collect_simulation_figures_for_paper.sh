#!/bin/bash

SOURCE=./figures
CHINESE_SOURCE=./chinese_figures
DEST=../paper/figure/simulations
CHINESE_DEST=../chinese_figures/simulations

# Remove the previous directory for simulations figures
mkdir -p ${DEST}
rm -r ${DEST}
mkdir -p ${CHINESE_DEST}
rm -r ${CHINESE_DEST}

dir_list="comparison v h ht gamma b lv"
for dir in ${dir_list}
do
    mkdir -p ${DEST}/${dir}_simulations
    cp ${SOURCE}/${dir}_simulations/*.pdf ${DEST}/${dir}_simulations/
done
for dir in ${dir_list}
do
    mkdir -p ${CHINESE_DEST}/${dir}_simulations
    cp ${CHINESE_SOURCE}/${dir}_simulations/*.pdf ${CHINESE_DEST}/${dir}_simulations/
done
