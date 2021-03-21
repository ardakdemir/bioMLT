#!/bin/bash

ner_folder=${1}
qas_output_dir=${2}
exp_pref=${3}

dataset_name=All-entities
#bioner_names=("linnaeus" "BC5CDR-disease" "BC4CHEMD" "BC5CDR-chem" "NCBI-disease" "JNLPBA" "s800" "BC2GM" "All-entities")
#
#for name in "${bioner_names[@]}"
#do
#  ner_model_path=$ner_folder"/best_ner_model_on_"$name
#  ner_vocab_path=$ner_folder"/best_ner_model_on_"$name"_vocab"
#  task_name=$exp_pref"qasonly_load_"${name}
#  echo $ner_model_path
#  echo $ner_vocab_path
#  echo $task_name
#  qsub -N $task_name bioMLT/qasonly_withload_submit.sh ${qas_output_dir} $ner_model_path $ner_vocab_path $name
#done


for s in 2000 5000 10000 20000
do
  for x in bert-instance random
  do
    name=$x"_"$s"_"$dataset_name
    echo $name
    ner_model_path=$ner_folder"/best_ner_model_on_"$name
    ner_vocab_path=$ner_folder"/best_ner_model_on_"$name"_vocab"
    task_name=$exp_pref"qasonly_load_"${name}
    echo $ner_model_path
    echo $ner_vocab_path
    echo $task_name
    qsub -N $task_name bioMLT/qasonly_withload_submit.sh ${qas_output_dir} $ner_model_path $ner_vocab_path $name
  done
done




