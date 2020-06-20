/usr/local/bin/nosh
output_dir_pref=${1}
model_pref=${2}

EVAL_PATH='biobert_data/Evaluation-Measures'
my_n2b_yesno_path='bioMLT/mytransformn2b_yesno.py'
my_n2b_list_path='bioMLT/mytransform_n2b_list.py'
my_n2b_factoid_path='bioMLT/mytransform_n2b_factoid.py'
out_for_bioasq_eval='input_for_bioasq'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N joint_flat_predict_all
cd ~

ner_data_folder="biobert_data/datasets/NER"
for file in $(ls ${ner_data_folder})
do
    folder_path=$ner_data_folder"/"$file
    if [ -d $folder_path ]
    then
        train_file=$folder_path"/ent_train_dev.tsv"
        dev_file=$folder_path"/ent_test.tsv"
        echo "Training and testing NER respectively on "
        my_dir=${output_dir_pref}"_"$file
        model_path=${my_dir}"/"${model_pref}"_"${file}
        output_dir='output_for_'${my_dir}
        overleaf_table_file="overleaf_table_for_"${my_dir}
        echo "output directory : "$output_dir
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
            singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --predict --load_model_path $model_path --squad_predict_list_file $squad_predict_list_file --squad_predict_yesno_file $squad_predict_yesno_file --squad_predict_factoid_file $squad_predict_factoid_file --output_dir ${output_dir} --pred_path $pred_path --nbest_path $nbest_path

            #rm $out_for_bioasq_eval
            python ${my_n2b_yesno_path}  --nbest_path $pred_path'_yesno.json' --output_path $out_for_bioasq_eval'_yesno'
            python ${my_n2b_factoid_path} --nbest_path $nbest_path'_factoid.json' --output_path $out_for_bioasq_eval'_factoid'
            python ${my_n2b_list_path} --nbest_path $nbest_path'_list.json' --output_path $out_for_bioasq_eval'_list'
            python bioMLT/merge_predictions.py ${out_for_bioasq_eval}  ${out_json}
            result_path=${output_dir}"/results_for_batch_"${test_num}
            echo "STORING RESULTS FOR BATCH "${test_num}" to "${result_path}
            java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_json > $result_path
        done
        python result_converter.py ${output_dir} ${overleaf_table_file}
    fi
done

echo "All results are stored in folders "$output_dir_temp
