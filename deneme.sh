ner_data_folder="biobert_data/datasets/NER"
x=$(ls ${ner_data_folder})
echo $x
ind=0
for folder_1 in $x
do
  c=0
  for folder_2 in $x
  do
    if [ $ind -le $c ]
    then
      echo $folder_1 $folder_2
    fi
    c=$(($c + 1))
  done
  ind=$(($ind+1))
done