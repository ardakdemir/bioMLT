/usr/local/bin/nosh
output_dir='save_dir'
model_save_name=$1
#test_num=$2
#squad_train_file='../biobert_data/qas_train_split.json'
#squad_eval_file='../biobert_data/qas_dev_split.json'
epoch_num=10
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'

nbest_path='biomlt_train_pred_nbest_pred'
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'

BIOBERT_PATH='/home/aakdemir/bioasq-biobert'
n2bfactoid_path='./biocodes/transform_n2b_factoid.py'
out_for_bioasq_eval='input_for_bioasq'
# squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
#pretrained_biobert_model_path='../biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='../biobert_data/biobert_v1.1_pubmed'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N  biomlt -e bioasq_wopresquad_err  -o bioasq_wopresquad_out

echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo $gold_path
echo "Fine tuning : "$pretrained_biobert_model_path
echo "Model to be saved:  "$model_save_name
echo $squad_predict_file
cd ~/bioMLT

pwd
# First we train a model on the bioasq dataset and then test it on the test set!! 
# Used for comparing the results for different pretrained models
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py  --squad_dir .  --biobert_model_path $pretrained_biobert_model_path --model_save_name $model_save_name --output_dir $output_dir  --num_train_epochs $epoch_num

#rm $nbest_path
for test_num in 1 2 3 4 5
do
    
    squad_predict_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-factoid-6b-'${test_num}'-snippet.json'
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $output_dir"/"$model_save_name --squad_predict_file $squad_predict_file --squad_dir . --nbest_path $nbest_path --load_model

    #rm $out_for_bioasq_eval
    python ${BIOBERT_PATH}"/"${n2bfactoid_path} --nbest_path $nbest_path --output_path $out_for_bioasq_eval

    java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_for_bioasq_eval > result_for_${test_num}.txt
done 
