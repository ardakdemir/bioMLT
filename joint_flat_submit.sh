/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N  biomlt -e error_file -o out_file
cd ~/bioMLT
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --batch_size 12 --eval_batch_size 12 --squad_train_file $1 --squad_predict_file $2 --squad_dir .

