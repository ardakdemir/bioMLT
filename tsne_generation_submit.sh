/usr/local/bin/nosh

#$ -cwd
#$ -l os7
#$ -N tsne_generation

singularity exec ~/singularity/pt-cuda-tf python bioMLT/tsne_generation.py


