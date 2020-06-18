/usr/local/bin/nosh
output_dir_temp=${1}
epoch=5
baseline_model_path='biobert-squadv1.1_pretrained'
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N joint_flat_train
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
        echo $train_file
        echo $dev_file
        output_dir=output_dir_temp"_"$file
        echo "output directory : "$output_dir
        singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --biobert_model_path ${baseline_model_path} --mode joint_flat  --ner_train_file $train_file --ner_dev_file $dev_file  --output_dir $output_dir  --load_model --max_seq_length 256 --num_train_epochs $epoch
    fi
done

echo "All results are stored in "$output_dir
