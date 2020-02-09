import argparse
import os
import json
parser = argparse.ArgumentParser()

parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file).")
parser.add_argument(
        "--save_dir", default="save_dir", type=str, required=False, help="The directory to save the split")

    
parser.add_argument(
        "--dev_output_file", default="qas_dev_split.json", type=str, required=False, help="The output dev data file (a text file)."
    )
parser.add_argument(
        "--train_output_file", default="qas_train_split.json", type=str, required=False, help="The output training data file (a text file)."
    )
parser.add_argument(
        "--split_rate", default=0.8, type=float, required=False, help="The split rate for data"
    )
args = parser.parse_args()


def split_bioasq_dataset(args):
    bioasq_path = args.train_data_file
    split_rate = args.split_rate
    train_dataset = json.load(open(bioasq_path))
    data = train_dataset['data'][0]['paragraphs']
    train_dataset['data'][0]['paragraphs'] = data[:int(len(data)*split_rate)]
    test_dataset = {'version ': train_dataset['version']}
    test_dataset['data'] = [{}]
    test_dataset['data'][0]['title'] = train_dataset['data'][0]['title']
    test_dataset['data'][0]['paragraphs'] = data[int(len(data)*split_rate):]
    with open(os.path.join(args.save_dir,args.train_output_file),'w') as f:
        json.dump(train_dataset,f)
    with open(os.path.join(args.save_dir,args.dev_output_file),"w") as f2:
        json.dump(test_dataset,f2)
   

    #return train_dataset,test_dataset

if __name__ == "__main__":
        
    split_bioasq_dataset(args)

