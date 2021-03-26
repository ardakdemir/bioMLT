/usr/local/bin/nosh
output_dir='save_dir'
finetuned_model=$1
model_save_name=$2
question_type=$3
predict=$4
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=5
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'

nbest_path='nbest_pred_'${model_save_name}
pred_path='pred_'${model_save_name}
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert/'
n2bfactoid_path='biocodes/transform_n2b_factoid.py'
n2byesno_path='./biocodes/transform_n2b_yesno.py'
myn2byesno_path='mytransform_n2b_yesno.py'
myn2b_list_path='mytransform_n2b_list.py'
result_file='qas_result_'${model_save_name}
if [ $question_type = 'yesno' ]
then
    echo ${question_type}' is yesno ?'
    converter=$myn2byesno_path
    input_for_converter=$pred_path
    result_file=${result_file}"_yesno"
elif [ $question_type = 'factoid' ]
then
    converter=${BIOBERT_PATH}${n2bfactoid_path}
    input_for_converter=$nbest_path
    result_file=${result_file}"_factoid"
else
    converter=$myn2b_list_path
    input_for_converter=$nbest_path
    result_file=${result_file}"_list" 
fi
echo "CONVERTER : "${converter}
echo "INPUT PATH: "${input_for_converter}
out_for_bioasq_eval='converted_'${question_type}'_'${model_save_name}
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N squadfinetuned_train

cho $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo "Fine tuning : "${finetuned_model}
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT

pwd

if [ $predict = 0 ]
then
    echo " PREDICT IS "${predict}
    # First we train a model on the bioasq dataset and then test it on the test set!! 
    # Used for comparing the results for different pretrained models
    if [ $question_type = 'yesno' ]
    then
        echo 'yesno training'
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py  --squad_dir .  --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num  --overwrite_cache  --load_model_path $finetuned_model --squad_yes_no --qa_type yesno
    elif [ $question_type = 'factoid' ]
    then
        echo 'factoid training'
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py    --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num  --overwrite_cache  --load_model_path $finetuned_model --qa_type factoid
    else
        echo 'list training'
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py    --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num  --overwrite_cache  --load_model_path $finetuned_model --qa_type list
    fi
    load_model_path=${output_dir}"/"${model_save_name}
else
    load_model_path=${finetuned_model}
    echo "SKIPPING TRAINING  - MOVING TO PREDICTIONS"
fi
#rm $nbest_path
for test_num in 1 2 3 4 5
do
    
    squad_predict_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-'${question_type}'-6b-'${test_num}'-snippet.json'
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    if [ $question_type = 'yesno' ]
    then
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $load_model_path  --squad_dir . --nbest_path $nbest_path  --squad_yes_no --squad_predict_yesno_file $squad_predict_file --overwrite_cache --pred_path $pred_path 
    elif [ $question_type = 'factoid' ]
    then
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $load_model_path   --nbest_path $nbest_path  --squad_predict_factoid_file $squad_predict_file --overwrite_cache --pred_path $pred_path --qa_type $question_type
    else
        singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $load_model_path   --nbest_path $nbest_path  --squad_predict_list_file $squad_predict_file --overwrite_cache --pred_path $pred_path --qa_type $question_type
    fi
    #rm $out_for_bioasq_eval
    python ${converter} --nbest_path $input_for_converter --output_path $out_for_bioasq_eval'_'${test_num}

    java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_for_bioasq_eval'_'${test_num} > result_for_${model_save_name}_${test_num}.txt
done
echo "PREDICTIONS COMPLETED"
echo "STORING AVERAGE ALL BATCHES FOR "${question_type}" IN : "${result_file}

python get_average.py result_for_${model_save_name}_ >> ${result_file}


