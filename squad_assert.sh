/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N assert_squad
cd ~/bioMLT
squad_train_file=train-v1.1.json
squad_predict_file=dev-v1.1.json
pretrained_name=bert-base-cased
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --batch_size 12 --eval_batch_size 12 --squad_train_file $squad_train_file --squad_predict_file $squad_predict_file  --learning_rate 0.00003 --init_bert --model_name_or_path $pretrained_name

