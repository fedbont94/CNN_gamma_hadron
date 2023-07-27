#! /bin/sh

PYTHON=/hkfs/home/project/hk-project-pevradio/rn8463/virtual_env/bin/python3
SCRIPT=/hkfs/home/project/hk-project-pevradio/rn8463/gamma_hadron/cnn_gamma_hadron.py

# TODO Remember to change the output directory!! 
$PYTHON $SCRIPT \
    --inputDir "/hkfs/work/workspace/scratch/rn8463-lv3_Simulations/" \
    --outputDir "/hkfs/work/workspace/scratch/rn8463-gamma_hadron/dummy/" \
    --year 2012 \
    --energyStart 4.0 \
    --energyEnd 7.0 \
    --batchSize 128 \
    --numEpochs 1000
    