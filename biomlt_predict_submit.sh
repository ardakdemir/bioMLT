/usr/local/bin/nosh
load_model_path=$1
test_num=$2
EVAL_PATH='/home/aakdemir/biobert_data/Evaluation-Measures'
BIOBERT_PATH='/home/aakdemir/bioasq-biobert'
n2bfactoid_path='./biocodes/transform_n2b_factoid.py'
out_for_bioasq_eval='input_for_bioasq'
bioasq_dataset_folder='/home/aakdemir/biobert_data/datasets/QA/BioASQ/'
bioasq_preprocessed_folder='/home/aakdemir/biobert_data/BioASQ-6b/'
squad_predict_file=${bioasq_preprocessed_folder}'test/Snippet-as-is/BioASQ-test-factoid-6b-'${test_num}'-snippet.json'
#squad_predict_file=${bioasq_dataset_folder}'BioASQ-test-factoid-6b-'${test_num}'.json'
gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
pretrained_biobert_model_path='/home/aakdemir/biobert_data/biobert-squadv1.1_pretrained'
pretrained_biobert_model_path='/home/aakdemir/biobert_data/biobert-v1.1_pubmed'
nbest_path='save_dir/biomlt_pred_out'

#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N  biomlt -e pred_error_file -o pred_out_file
echo $EVAL_PATH
echo $BIOBERT_PATH
echo $n2bfactoid_path
echo $out_for_bioasq_eval
echo $gold_path
cd ~/bioMLT

pwd
singularity exec --nv ~/singularity/pt-cuda-tf python biomlt.py --predict --load_model_path $load_model_path --squad_predict_file $squad_predict_file --squad_dir . --nbest_path $nbest_path --biobert_model_path $pretrained_biobert_model_path
python ${BIOBERT_PATH}"/"${n2bfactoid_path} --nbest_path $nbest_path --output_path $out_for_bioasq_eval

java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_for_bioasq_eval

