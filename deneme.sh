ner_data_folder="ner_data/bioner_data"
for file in $(ls ${ner_data_folder})
do
    folder_path=$ner_data_folder"/"$file
    if [ -d $folder_path ]
    then
        train_file=$folder_path"/bio_"$file"_train.tsv"
        dev_file=$folder_path"/bio_"$file"_devel.tsv"
        echo $train_file
        echo $dev_file
    fi
done
