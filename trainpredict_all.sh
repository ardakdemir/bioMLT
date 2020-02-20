rm result_for_*
rm yesno_*
rm pred_*
qsub trainbioasq_predict.sh pubmed_init_yesno6b_2002_1
qsub train_predictbertinit.sh bert_init_yesno6b_2002_1
