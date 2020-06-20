/usr/local/bin/nosh
output_dir_temp=$1
pretrained_model_path=$2
epoch=30
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N joint_flat_train_1806
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
        output_dir=${output_dir_temp}"_"$file
        echo "output directory : "$output_dir
        save_name="best_model_"${file}
        singularity exec --nv ~/singularity/pt-cuda-tf python bioMLT/biomlt_alldata.py --mode joint_flat  --model_save_name ${save_name} --ner_train_file $train_file --ner_dev_file $dev_file  --output_dir $output_dir  --load_model_path ${pretrained_model_path} --max_seq_length 256 --num_train_epochs $epoch
    fi
done

echo "All results are stored in folders "$output_dir_temp
