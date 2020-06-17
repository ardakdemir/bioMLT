/usr/local/bin/nosh
load_model_path=$1
save_name=$2
output_dir='output_for_'${save_name}
bioasq_dataset_folder='biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='biobert_data/BioASQ-6b/'

nbest_name='nbest_pred'
pred_name='preds'
EVAL_PATH='biobert_data/Evaluation-Measures'

out_for_bioasq_eval='input_for_bioasq'
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='biobert_data/biobert_v1.1_pubmed'

my_n2b_yesno_path='bioMLT/mytransformn2b_yesno.py'
my_n2b_list_path='bioMLT/mytransform_n2b_list.py'
my_n2b_factoid_path='bioMLT/mytransform_n2b_factoid.py'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N  predict_all -e predict_all_err  -o predict_all_out
echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
#echo "Fine tuning : "$pretrained_biobert_model_path
#echo "Model to be saved:  "$model_save_name
echo "Predicting with model "$load_model_path
echo $squad_predict_file
cd ~

pwd
# First we train a model on the bioasq dataset and then test it on the test set!! 
# Used for comparing the results for different pretrained models
#singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py  --squad_dir .  --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num

#rm $nbest_path
for test_num in 1 2 3 4 5
do
    
    squad_predict_factoid_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-factoid-6b-'${test_num}'-snippet.json'
    squad_predict_list_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-list-6b-'${test_num}'-snippet.json'
    squad_predict_yesno_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-yesno-6b-'${test_num}'-snippet.json'
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    nbest_path=${output_dir}'/nbest_pred_'${test_num}
    pred_path=${output_dir}'/preds_'${test_num}
    out_for_bioasq_eval=${output_dir}"/transformed_preds_"${test_num}
    out_json=${output_dir}'/merged_json_batch_'${test_num}
    singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --predict --load_model_path $load_model_path --squad_predict_list_file $squad_predict_list_file --squad_predict_yesno_file $squad_predict_yesno_file --squad_predict_factoid_file $squad_predict_factoid_file --output_dir ${output_dir} --pred_path $pred_path --nbest_path $nbest_path

    #rm $out_for_bioasq_eval
    python ${my_n2b_yesno_path}  --nbest_path $pred_path'_yesno.json' --output_path $out_for_bioasq_eval'_yesno'
    python ${my_n2b_factoid_path} --nbest_path $nbest_path'_factoid.json' --output_path $out_for_bioasq_eval'_factoid'
    python ${my_n2b_list_path} --nbest_path $nbest_path'_list.json' --output_path $out_for_bioasq_eval'_list'
    python bioMLT/merge_predictions.py ${out_for_bioasq_eval}  ${out_json}
    result_path=${output_dir}"/results_for_batch_"${test_num}
    echo "STORING RESULTS FOR BATCH "${test_num}" to "${result_path}
    java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_json > $result_path
done 

all_results_file=${output_dir}'/all_results_'${save_name}'.txt'
echo "Combining results for each batch to "$all_results_file
cat ${output_dir}"/results_for_batch_"* >> ${all_results_file}
