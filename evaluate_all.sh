save_name=$1
output_dir='output_for_'${save_name}
bioasq_dataset_folder='biobert_data/datasets/QA/BioASQ/'
for test_num in 1 2 3 4 5
do
    out_json=${output_dir}'/merged_json_batch_'${test_num}
    gold_path=${bioasq_dataset_folder}'6B'${test_num}'_golden.json'
    result_path=${output_dir}"/results_for_batch_"${test_num}
    java -Xmx10G -cp ${EVAL_PATH}/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 $gold_path  $out_json > $result_path
done

all_results_file=${output_dir}'/all_results_'${save_name}'.txt'
echo "Combining results for each batch to "$all_results_file
cat ${output_dir}"/results_for_batch_*" >> ${all_results_file}
