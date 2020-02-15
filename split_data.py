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
        "--out_file", default="subset.json", type=str, required=False, help="The output dev data file (a text file)."
    )
parser.add_argument(
        "--train_output_file", default="qas_train_split.json", type=str, required=False, help="The output training data file (a text file)."
    )
parser.add_argument(
        "--question_num", default=10, type=int, required=False, help="The number of questions for getting subset"
    )
parser.add_argument(
        "--split_rate", default=0.8, type=float, required=False, help="The split rate for data"
    )
parser.add_argument(
        "--get_subset", action="store_true", required=False, help="To apply subsetting"
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

def get_subset(file,question_num):
    bioasq_path = file
    train_dataset = json.load(open(bioasq_path))
    data = train_dataset['data'][0]['paragraphs']
    #train_dataset['data'][0]['paragraphs'] = data[:int(len(data) * split_rate)]
    test_dataset = {'version ': train_dataset['version']}
    test_dataset['data'] = [{}]
    test_dataset['data'][0]['title'] = train_dataset['data'][0]['title']
    test_dataset['data'][0]['paragraphs'] = data[:question_num]
    with open(os.path.join(args.save_dir, args.out_file), "w") as f2:
        json.dump(test_dataset, f2)
    #return train_dataset,test_dataset

if __name__ == "__main__":

    if args.get_subset:
        get_subset(args.train_data_file,args.question_num)
    else:
        split_bioasq_dataset(args)

