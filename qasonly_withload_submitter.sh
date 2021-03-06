#!/bin/bash

ner_folder=${1}
qas_output_dir=${2}
bioner_names=("linnaeus" "BC5CDR-disease" "BC4CHEMD" "BC5CDR-chem" "NCBI-disease" "JNLPBA" "s800" "BC2GM")

for name in "${bioner_names[@]}"
do
  ner_model_path=$ner_folder"/best_ner_model_on_"$name
  ner_vocab_path=$ner_folder"/best_ner_model_on_"$name"_vocab"
  task_name="qasonly_load_"${name}
  echo $ner_model_path
  echo $ner_vocab_path
  echo $task_name
  #qsub -N $task_name bioMLT/qasonly_withload_submit.sh ${qas_output_dir} $ner_model_path $ner_vocab_path $name
done



