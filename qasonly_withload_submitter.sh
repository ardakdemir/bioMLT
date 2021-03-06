#!/bin/bash

ner_folder=${1}

bioner_names=("linnaeus" "BC5CDR-disease" "BC4CHEMD" "BC5CDR-chem" "NCBI-disease" "JNLPBA" "s800" "BC2GM")

for name in "${bioner_names[@]}"
do
  echo $name
done



