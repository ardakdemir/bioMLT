/usr/local/bin/nosh
output_dir='save_dir'
finetuned_model=$1
model_save_name=$2
data_type=$3 
predict=$4
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=5
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'
bioasq_preprocessed_folder2='/home/aakdemir/biobert_data/BioASQ-7b/'

nbest_path='nbest_pred_'${model_save_name}
pred_path='pred_'${model_save_name}
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert/'
n2bfactoid_path='biocodes/transform_n2b_factoid.py'
n2byesno_path='./biocodes/transform_n2b_yesno.py'
my_n2b_yesno_path='mytransform_n2b_yesno.py'
my_n2b_list_path='mytransform_n2b_list.py'
my_n2b_factoid_path='mytransform_n2b_factoid.py'
result_file='qas_result_'${model_save_name}
init_result_file=$result_file
appended_folder='Appended-Snippet'
fullabs_folder='Full-Abstract'
snippet_folder='Snippet-as-is'
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N squad_allqa_train_predict

echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo "Fine tuning : "${finetuned_model}
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT
pwd


folder=$snippet_folder
if [ ${data_type} = 'fullabstract' ]
then
    folder=$fullabs_folder
elif [ ${data_type} = 'appended' ]
then 
    folder=$appended_folder
fi

if [ $predict = 0 ]
then
    echo " PREDICT IS "${predict}
    
    # First we train a model on the bioasq dataset and then test it on the test set!! 
    # Used for comparing the results for different pretrained models
    echo "Running training on all question types !! "
    echo "First experimenting with qas aloneee"
    echo "Best model will be saved in "${output_dir}/${model_save_name}
    echo "FOLDER TYPE "${folder}
    squad_train_yesno_file=${bioasq_preprocessed_folder}'train/'${folder}'/train_yesno_6b_'${data_type}'.json'
    squad_train_list_file=${bioasq_preprocessed_folder}'train/'${folder}'/train_list_6b_'${data_type}'.json'
    squad_train_factoid_file=${bioasq_preprocessed_folder}'train/'${folder}'/train_factoid_6b_'${data_type}'.json'
    squad_eval_yesno_file=${bioasq_preprocessed_folder2}'train/'${folder}'/train_yesno_7b_'${data_type}'.json'
    squad_eval_list_file=${bioasq_preprocessed_folder2}'train/'${folder}'/train_list_7b_'${data_type}'.json'
    squad_eval_factoid_file=${bioasq_preprocessed_folder2}'train/'${folder}'/train_factoid_7b_'${data_type}'.json'
    singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num   --load_model_path $finetuned_model  --squad_train_factoid_file ${squad_train_factoid_file} --squad_train_list_file ${squad_train_list_file} --squad_train_yesno_file ${squad_train_yesno_file} --squad_predict_factoid_file ${squad_eval_factoid_file} --squad_predict_list_file ${squad_eval_list_file} --squad_predict_yesno_file ${squad_eval_yesno_file}
    load_model_path=${output_dir}"/"${model_save_name}
else
    load_model_path=${finetuned_model}
    echo "SKIPPING TRAINING  - MOVING TO PREDICTIONS"
fi
#rm $nbest_path
echo "Loading model from "$load_model_path

out_for_bioasq_eval='converted_'${model_save_name}
for test_num in 1 2 3 4 5
do
    echo "FOLDER TYPE "${folder}" FOR PREDICTIONS"
    squad_predict_yesno_file=${bioasq_preprocessed_folder}'test/'${folder}'/test_yesno_6b_'${test_num}'.json'
    squad_predict_list_file=${bioasq_preprocessed_folder}'test/'${folder}'/test_list_6b_'${test_num}'.json'
    squad_predict_factoid_file=${bioasq_preprocessed_folder}'test/'${folder}'/test_factoid_6b_'${test_num}'.json'
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    singularity exec --nv ~/singularity/pt-cuda-tf python biomlt_alldata.py --predict --load_model_path $load_model_path  --squad_dir . --nbest_path $nbest_path --squad_predict_factoid_file $squad_predict_factoid_file --squad_predict_list_file $squad_predict_list_file --squad_predict_yesno_file $squad_predict_yesno_file  --pred_path $pred_path 
    #rm $out_for_bioasq_eval

    python ${my_n2b_yesno_path}  --nbest_path $pred_path'_yesno.json' --output_path $out_for_bioasq_eval'_yesno'
    python ${my_n2b_factoid_path} --nbest_path $nbest_path'_factoid.json' --output_path $out_for_bioasq_eval'_factoid'
    python ${my_n2b_list_path} --nbest_path $nbest_path'_list.json' --output_path $out_for_bioasq_eval'_list'
    echo "Storing all intermediate files with prefix "${out_for_bioasq_eval}
    python merge_predictions.py ${out_for_bioasq_eval}  'merged_json_'${data_type}"_"${test_num}
    java  -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  'merged_json_'${data_type}'_'${test_num} > result_for_${model_save_name}_${test_num}.txt
done
echo "PREDICTIONS COMPLETED"
echo "STORING AVERAGE ALL BATCHES FOR  IN : "${result_file}

python get_average.py result_for_${model_save_name}_ >> ${result_file}
