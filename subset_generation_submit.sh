/usr/local/bin/nosh
ner_root_folder=$1
save_folder_pref=$2

#$ -cwd
#$ -l v100=1, s_vmem=100G, mem_req=100G
cd ~


singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/data_selection.py --ner_root_folder $ner_root_folder --save_folder_pref $save_folder_pref
