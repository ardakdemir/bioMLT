rm result_for_*
rm yesno_*
qsub predict_pubmed.sh pubmed_init_yesno6b_1902
qsub predict_bertinit.sh bert_init_yesno6b_1902
