/usr/local/bin/nosh

save_folder=$1
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N tsne_generation

echo "Running tsne generation"
singularity exec ~/singularity/pt-cuda-tf python bioMLT/tsne_generation.py --save_folder $save_folder --limit 10 --with_pca 


