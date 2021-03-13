/usr/local/bin/nosh
ner_root_folder=${1}
save_root_folder=${2}
#QAS Model is given input of the embeddings

#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/data_selection.py --ner_root_folder ${ner_root_folder} --save_root_folder ${save_root_folder}
