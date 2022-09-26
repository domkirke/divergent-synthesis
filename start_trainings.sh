RUNDIR="runs/"
python3 train_model.py --config-name div_alexnet_isfid rundir=${RUNDIR}
python3 train_model.py --config-name div_inception_isfid rundir=${RUNDIR}
python3 train_model.py --config-name div_mobilenet_isfid rundir=${RUNDIR} 
python3 train_model.py --config-name div_custom_isfid rundir=${RUNDIR} 

python3 train_model.py --config-name div_alexnet_mmd rundir=${RUNDIR} 
python3 train_model.py --config-name div_inception_mmd rundir=${RUNDIR}
python3 train_model.py --config-name div_mobilenet_mmd rundir=${RUNDIR}
python3 train_model.py --config-name div_custom_mmd rundir=${RUNDIR} 
