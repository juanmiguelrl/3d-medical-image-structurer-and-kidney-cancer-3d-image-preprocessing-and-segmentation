
import os
import shutil
import random
import json


def split_dataset(input_dir,output_dir,train_proportion,test_proportion=0,json_file=False):
    if train_proportion > 1 or test_proportion > 1 or (train_proportion+test_proportion) > 1:
        print("The proportions must be between 0 and 1 and the sum of them must be less than 1")
        return
    val_proportion=1-train_proportion-test_proportion
    print("train_proportion: ",train_proportion)
    print("val_proportion: ",val_proportion)
    print("test_proportion: ",test_proportion)

    #takes the list of cases in the input directory and splits it in 3 lists randomly
    #this cases are the full path of the directories wich start with the string "case_"
    cases_list = os.listdir(input_dir)
    #check if the elements in the list start with "case_"
    cases_list = [case for case in cases_list if case.startswith("case_")]
    #check if the elements in the list are directories and adds the full path
    if json_file:
        #adds the full path to the cases
        cases_list = [os.path.join(input_dir,case) for case in cases_list if os.path.isdir(os.path.join(input_dir,case))]
    else:
        cases_list = [case for case in cases_list if os.path.isdir(os.path.join(input_dir,case))]
    random.shuffle(cases_list)
    train_list = cases_list[:int(len(cases_list)*train_proportion)]
    val_list = cases_list[int(len(cases_list)*train_proportion):int(len(cases_list)*(train_proportion+val_proportion))]
    test_list = cases_list[int(len(cases_list)*(train_proportion+val_proportion)):]
    #test_list = cases_list[int(len(cases_list)*train_proportion):int(len(cases_list)*(train_proportion+test_proportion))]

    #if json is true, creates a json file with the list of cases in each set
    if json_file:
        print(cases_list)
        #creates the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #creates the json files
        with open(os.path.join(output_dir,"train.json"), 'w') as outfile:
            json.dump(train_list, outfile)
        with open(os.path.join(output_dir,"test.json"), 'w') as outfile:
            json.dump(test_list, outfile)
        with open(os.path.join(output_dir,"val.json"), 'w') as outfile:
            json.dump(val_list, outfile)
    #else, it creates a directory for each set and copies the cases in the corresponding directory
    else:
        train_output_dir = os.path.join(output_dir,"train")
        test_output_dir = os.path.join(output_dir,"test")
        val_output_dir = os.path.join(output_dir,"val")
        #creates the output directory if it doesn't exist
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        if not os.path.exists(val_output_dir):
            os.makedirs(val_output_dir)
        #copies the cases in the corresponding directory
        for case in train_list:
            shutil.copytree(os.path.join(input_dir,case),os.path.join(train_output_dir,case),dirs_exist_ok=True)
        for case in test_list:
            shutil.copytree(os.path.join(input_dir,case),os.path.join(output_dir,"test",case),dirs_exist_ok=True)
        for case in val_list:
            shutil.copytree(os.path.join(input_dir,case),os.path.join(output_dir,"val",case),dirs_exist_ok=True)
    
    print(f"Train set: {len(train_list)} cases")
    print(f"Test set: {len(test_list)} cases")
    print(f"Validation set: {len(val_list)} cases")


