import os
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_path", default="../biobert_data/BioASQ-9b/BioASQ-task9bPhaseB-testset2", type=str, required=False,
        help="The path to load the model to continue training."
    )
    parser.add_argument(
        "--save_folder", default="../bioasq_tb2_2603", type=str, required=False,
        help="The path to save the predictions"
    )
    parser.add_argument(
        "--path_to_script", default="bioasq_pred_submit.sh", type=str, required=False,
        help="The path to submit script"
    )

    args = parser.parse_args()
    return args


model_name_map = {"UTokyo_qasonly":"UoT_baseline",
                  "UTokyo_load_All-entities":"UoT_allquestions",
                  "UTokyo_load_BC5CDR-disease":"UoT_multitask_learn",
                  "UTokyo_load_NCBI-disease":"UoT_bestfac",
                  "UTokyo_load_s800":"UoT_bestyesno"}

model_dict = {"UTokyo_qasonly": {"model_path": "../qaswithload_bioberts_1603/best_qas_model_woner"},
              "UTokyo_load_All-entities": {"model_path": "../qaswithload_bioberts_1603/best_qas_model_All-entities"},
              "UTokyo_load_NCBI-disease": {"model_path": "../qaswithload_bioberts_1603/best_qas_model_linnaeus"},# best full-dataset
              "UTokyo_load_s800": {"model_path": "../qaswithload_bioberts_1603/best_qas_model_bert-instance_1000_All-entities"}, ## with subset
              "UTokyo_load_BC5CDR-disease": { ## with subset-selection
                  "model_path": "../qaswithload_bioberts_1603/best_qas_model_bert-instance_2000_All-entities"}} # cuz- best yesno


def bioasq_pred_submitter():
    args = parse_args()
    test_path = args.test_path
    save_folder = args.save_folder
    path_to_script = args.path_to_script
    for sys_name, model in model_dict.items():
        print("Submitting {} {} ".format(sys_name, model))
        model_path = model["model_path"]
        cmd = "qsub {} {} {} {} {}".format(path_to_script, model_path, test_path, sys_name, save_folder)
        print("Calling: {}".format(cmd))
        subprocess.call(cmd, shell=True)


def main():
    bioasq_pred_submitter()


if __name__ == "__main__":
    main()
