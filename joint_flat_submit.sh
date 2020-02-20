echo "RUNNING SCRIPT FOR JOINT TRAINING OF NER AND QAS"

/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N  joint_flat_train_predict
cd ~/bioMLT

pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
finetuned_model=$1
question_type='factoid'

bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'

nbest_path='nbest_pred_'${model_save_name}
pred_path='pred_'${model_save_name}
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert'
n2bfactoid_path='./biocodes/transform_n2b_factoid.py'
n2byesno_path='./biocodes/transform_n2b_yesno.py'
myn2byesno_path='mytransformn2b_yesno.py'

if [ $question_type = 'yesno' ]
then
    echo ${question_type}' is yesno ?'
    converter=$myn2byesno_path
    input_for_converter=$pred_path
else
    converter=${BIOBERT_PATH}${n2bfactoid_path}
    input_for_converter=$nbest_path
fi
echo "CONVERTER : "${converter}
echo "INPUT PATH: "${input_for_converter}
out_for_bioasq_eval='converted_'${question_type}'_'${model_save_name}

singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --mode 'joint_flat' --squad_dir . --biobert_model_path ${pretrained_biobert_model_path} --load_model --load_model_path ${finetuned_model}


for test_num in 1 2 3 4 5
do
    
    squad_predict_yesno_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-'${question_type}'-6b-'${test_num}'-snippet.json'
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $output_dir"/"$model_save_name  --squad_dir . --nbest_path $nbest_path --load_model --squad_yes_no --squad_predict_yesno_file $squad_predict_yesno_file --overwrite_cache --pred_path $pred_path

    #rm $out_for_bioasq_eval
    python ${converter} --nbest_path $input_for_converter --output_path $out_for_bioasq_eval'_'${test_num}

    java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_for_bioasq_eval'_'${test_num} > result_for_${model_save_name}_${test_num}.txt
done
