import os
import os.path as op
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys


BASE_DIR = op.dirname(op.dirname(op.abspath(__file__)))

NEEDED_TAGS = {
    'PatientID': '0010,0020',
    'PatientAge': '0010,1010',
    'PatientSex': '0010,0040',
    'PatientWeight': '0010,1030',	
    'PatientSize': '0010,1020',
    'PatientBirthDate': '0010,0030',
    'StudyDate': '0008,0020',
    'StudyTime': '0008,0030',
    'StudyID': '0020,0010',
    'StudyDescription': '0008,1030',
    'StudyInstanceUID': '0020,000D',
    'StudyComments': '0032,4000',
    'Modality': '0008,0060',
    'SeriesInstanceUID' : '0020,000E',
    'SeriesDescription': '0008,103E',
    'SeriesNumber': '0020,0011',
    'SeriesInstanceUID': '0020,000E',
    'ManufacturerModelName': '0008,1090',
    'SliceThickness': '0018,0050',
    'WindowWidth': '0028,1051',
}

PATIENT_TAGS = {
    'PatientID': '0010,0020',
    'PatientAge': '0010,1010',
    'PatientSex': '0010,0040',
    'PatientWeight': '0010,1030',
    'PatientSize': '0010,1020',
    'PatientBirthDate': '0010,0030',
}

STUDY_TAGS = {
    'StudyDate': '0008,0020',
    'StudyTime': '0008,0030',
    'StudyID': '0020,0010',
    'StudyDescription': '0008,1030',
    'StudyInstanceUID': '0020,000D',
    'StudyComments': '0032,4000',
}

SERIES_TAGS = {
    'SeriesDate': '0008,0021',
    'SeriesTime': '0008,0031',
    'SeriesNumber': '0020,0011',
    'SeriesDescription': '0008,103E',
    'SeriesInstanceUID': '0020,000E',
    'SeriesComments': '0032,4000',
    'Modality': '0008,0060',
    'ProtocolName': '0018,1030',
    'BodyPartExamined': '0018,0015',
    'Manufacturer': '0008,0070',
    'ManufacturerModelName': '0008,1090',
    'MagenticFieldStrength': '0018,0087',
    'EchoTime': '0018,0081',
    'RepetitionTime': '0018,0080',
    'InversionTime': '0018,0082',
    'FlipAngle': '0018,1314', 
    'RadiopharmaceuticalInformationSequence': '0054,0010',
    'Radiopharmaceutical': '0018,0031',
}

IMAGE_TAGS = {
    'ImageType': '0008,0008',
    'InstanceNumber': '0020,0013',
    'ImagePositionPatient': '0020,0032',
    'ImageOrientationPatient': '0020,0037',
    'SliceLocation': '0020,1041',
    'PixelSpacing': '0028,0030',
    'SliceThickness': '0018,0050',
    'Rows': '0028,0010',
    'Columns': '0028,0011',
    'WindowCenter': '0028,1050',
    'WindowWidth': '0028,1051',
    'RescaleIntercept': '0028,1052',
    'RescaleSlope': '0028,1053',
    'RescaleType': '0028,1054',
    'SOPInstanceUID': '0008,0018',
}

MODALITY_TAGS = {
    'PT': 'pet',
    'CT': 'ct',
    'MR': 'mr',
    'US': 'us',
    'XA': 'xa',
}

def guess_mri_sequence(ds):
    """ Find out MRI sequence by reading meaningful DICOM tags """

    et = ds.EchoTime
    rt = ds.RepetitionTime
    it = ds.InversionTime
    fa = ds.FlipAngle

    if et == 0.0 and rt == 0.0 and it == 0.0 and fa == 0.0:
        return 'T1'
    elif et == 0.0 and rt == 0.0 and it == 0.0 and fa != 0.0:
        return 'T2'
    elif et != 0.0 and rt != 0.0 and it == 0.0 and fa == 0.0:
        return 'FLAIR'
    elif et != 0.0 and rt != 0.0 and it == 0.0 and fa != 0.0:
        return 'PD'
    elif et != 0.0 and rt != 0.0 and it != 0.0 and fa != 0.0:
        return 'SWI'
    else:
        return 'UNKNOWN'

    
def get_series_modality(dcm_file):
    """ Get series modality """
    modality = pydicom.dcmread(dcm_file).Modality
    print("modality: ",modality)
    return

def get_study_instance_uid(dcm_file):
    """ Get study instance UID """
    uid = pydicom.dcmread(dcm_file).StudyInstanceUID
    print("study instance id: ",uid)
    return

def get_patient_id(dcm_file):
    """ Get patient ID """
    id = pydicom.dcmread(dcm_file).PatientID
    print("patient id: ",id)
    return id


def read_dicom2(dcm_file,dataset,dataset_name):
    ds = pydicom.dcmread(dcm_file)
    tags_values = [(tag, ds.data_element(tag).value) if tag in ds else (tag,'') for tag in NEEDED_TAGS]

    #check if dataset is empty, if so, add the tags and values to the dataset
    if dataset.empty:
        #the columns are defined by the tags
        dataset = pd.DataFrame(columns=[tag for tag, value in tags_values])
        dataset.loc[len(dataset.index)] = [value for tag, value in tags_values]
        #for tag, value in tags_values:
        #    dataset[tag] = value
    #store the tags and values in the dataset only if the patientid is not already in the pandas dataset
    #elif ds.data_element("PatientID").value not in dataset["PatientID"].values:
    elif (dataset.loc[(dataset["PatientID"] == ds.data_element("PatientID").value) & (dataset["StudyInstanceUID"] == ds.data_element("StudyInstanceUID").value) & (dataset["StudyID"] == ds.data_element("StudyID").value) & (dataset['SeriesInstanceUID'] == ds.data_element("SeriesInstanceUID").value)]).empty:
        #print rows where patientid is the same as the one in the dicom file
        #if (dataset.loc[dataset["PatientID"] == ds.data_element("PatientID").value & dataset["StudyInstanceUID"] == ds.data_element("SeriesInstanceUID").value & dataset["StudyID"] == ds.data_element("StudyID").value]).empty:
        #    print("aaaaaaaaaaaaa")     
        #print("ooooooooo")
        #print(dataset.loc[(dataset["PatientID"] == ds.data_element("PatientID").value) & (dataset["StudyInstanceUID"] == ds.data_element("StudyInstanceUID").value)])
        #dataset.loc[len(dataset.index)] = [value for tag, value in tags_values]
        #for tag, value in tags_values add the value to the dataset
        #dataset.append(tags_values)
        dataset.loc[len(dataset.index)] = [value for tag, value in tags_values]
    #dataset.loc[len(dataset.index)] = [value for tag, value in tags_values]
    #dataset = pd.concat([dataset, pd.DataFrame(columns=[tag for tag, value in tags_values])])
    
    #print(dataset)
    return dataset


def read_dicom(dcm_file,patient_list):
    """ Read DICOM tags of interest """
    
    ds = pydicom.dcmread(dcm_file)
    #print(f"DICOM metadata: \n {ds}")

    # Patient tags
    #for t in PATIENT_TAGS:
    #    print(ds.data_element(t).value)
    #print(ds.data_element("PatientID").value)
    #tags_values = [ds.data_element(tag).value for tag in PATIENT_TAGS.values()]
    #tags_values = [(tag,ds.data_element(tag).value) for tag in PATIENT_TAGS]
    tags_values = [(tag, ds.data_element(tag).value) if tag in ds else (tag,'') for tag in PATIENT_TAGS]

    patient_id = ds.data_element("PatientID").value
    study_id = ds.data_element("StudyID").value
    series_id = ds.data_element("SeriesInstanceUID").value

    #we suppose that the patient id is unique and that all files have the same patient id,study id and series id has the same useful information to change
    #TODO
    #this has to be changed to another function later as now is being used to store patient data in a csv file and json file

    if not patient_list:
        patient_list = [(patient_id,study_id,series_id)]
    elif (patient_id,study_id,series_id) not in patient_list:
        patient_list.append((patient_id,study_id,series_id))
    else:
        #print("Patient already in the list")
        return patient_list
    dict_values = dict(tags_values)
    df = pd.DataFrame.from_dict(dict_values, orient='index')
    df.to_csv(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.csv'.format(patient_id=patient_id)), header=False)
    #df.to_json(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.json'), orient='records')
    #df.to_csv(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.csv'), header=False)
    #df.to_json(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.json',orient='index'))
    #with open(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.json'), "w") as outfile:
    json_file = open(op.join(BASE_DIR, 'data', 'patient-{patient_id}-data.json'.format(patient_id=patient_id)), "w")
    json.dump(dict_values,json_file, indent=4)
    json_file.close()

    # Study tags
    [(tag, ds.data_element(tag).value) if tag in ds else (tag,'') for tag in STUDY_TAGS]
    
    dict_values = dict(tags_values)
    df = pd.DataFrame(tags_values)
    df.to_csv(op.join(BASE_DIR, 'data', 'study-{study_id}-data.csv'.format(study_id=study_id)), header=False)
    json_file = open(op.join(BASE_DIR, 'data', 'study-{study_id}-data.json'.format(study_id=study_id)), "w")
    json.dump(dict_values,json_file, indent=4)
    json_file.close()

    # Series tags
    [(tag, ds.data_element(tag).value) if tag in ds else (tag,'') for tag in SERIES_TAGS]
    
    dict_values = dict(tags_values)
    df = pd.DataFrame(tags_values)
    df.to_csv(op.join(BASE_DIR, 'data', 'series-{series_id}-data.csv').format(series_id=series_id), header=False)
    json_file = open(op.join(BASE_DIR, 'data', 'series-{series_id}-data.json'.format(series_id=series_id)), "w")
    json.dump(dict_values,json_file, indent=4)
    json_file.close()

    return patient_list


def grab_dataset_deatiled_info(dcm_file):
    """ Create a CSV file summarizing the relavant information for this dataset """

    TAGS = {
        'PID': 'PatientID',
        'IID': 'ImageID',
        'SID': 'DatasetID',
        'IMAGING': 'ImagesModalities',
        'ORIGIN': 'ImagesOrigin',
        'DESTINATION': 'ImagesDestination',
    }

    df = pd.DataFrame(columns=TAGS.keys())
    # TODO Fill whit the data


def plot_dicom(ds):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ds.pixel_array, cmap="gnuplot2")
    fig.savefig(op.join(BASE_DIR, 'data', 'pet-dicom.png'))


if __name__ == "__main__":
    dicom_files = os.listdir(op.join(BASE_DIR, 'data'))
    for dcm_file in dicom_files:
        dcm_file = op.join(BASE_DIR, 'data', dcm_file)
        read_dicom(dcm_file)
        # plot_dicom(dcm_file)


def list_files(directory, output):
    #creates in the output file a list of files and sub directories of the directory given
    #if there is output file it will write to it, else it will print to the console
    original_stdout = sys.stdout # Save a reference to the original standard output
    if output:
        f = open(output, 'w')
        sys.stdout = f # Change the standard output to the file
    #walk through the directory
    for dir, dirs, files in os.walk(directory, topdown=False):
    #print(dir)
        for name in files:
            #fwriteline = dir + '/' + name
            print(dir + '/' + name)
    f.close()
    sys.stdout = original_stdout # Reset the standard output to its original value

def read_list_files(directory, output):
    #gets only the name of the subdirectory
    output_name = os.path.basename(directory)
    output_dir_dataset = op.join(output, output_name)
    os.makedirs(output_dir_dataset, exist_ok=True)
    patient_list = list()
    #creates empty pandas dataframe
    df = pd.DataFrame()
    total_df = pd.DataFrame()

    #reads each subdirectory of the directory given and creates a folder with the name of the subdirectory at the output
    # Loop through each subdirectory in the input directory
    for subdir in os.listdir(directory):
        # Check if the subdirectory is a directory (not a file)
        if os.path.isdir(os.path.join(directory, subdir)):
            # Create a new directory with the same name as the subdirectory
            new_dir = os.path.join(output_dir_dataset, subdir)
            os.makedirs(new_dir, exist_ok=True)

        #transform the DICOM files in NIFTI files
        #os.sytem(f'dcm2niix -z y -f {op.basename(nifti_file)} -o {output_dir} {dicom_dir}')
        
        #reads the DICOM files from the list of files and sub directories of the directory given
        for dir, dirs, files in os.walk(directory, topdown=False):
        #print(dir)
            for name in files:
                #print(dir + '/' + name)
                #print("\n______----------------_______\n")
                #check if is a DICOM file
                if name.endswith(".dcm"):
                    #patient_list = read_dicom(dir + '/' + name,patient_list)
                    df = read_dicom2(dir + '/' + name,df,output_name)
                #print("\n**********\n\n**********\n\n**********\n")
        #stores the pandas dataframe in a csv file in the output directory named with the name of the last sub directory of the directory given
        df.to_csv(op.join(new_dir, output_name + '.csv'), header=True)
        #for each modality creates a csv file with the information of all the patients
        for series_description in df['SeriesDescription'].unique():
            df[df['SeriesDescription'] == series_description].to_csv(op.join(new_dir, series_description + '.csv'), header=True)
            df[df['SeriesDescription'] == series_description].to_json(op.join(new_dir, series_description + "_read" + '.json'),orient="records",lines=True)

        if total_df.empty:
            #the columns are defined by the tags
            total_df = df.copy()
        else:
            total_df = pd.concat([total_df, df], ignore_index=True)
    total_df.to_csv(op.join(output_dir_dataset, output_name + '.csv'), header=True)
    
    
def read_list_files2(directory, output):
    #gets only the name of the subdirectory
    output_name = os.path.basename(directory)
    output_dir_dataset = op.join(output, output_name)
    os.makedirs(output_dir_dataset, exist_ok=True)
    patient_list = list()
    #creates empty pandas dataframe
    df = pd.DataFrame()
    total_df = pd.DataFrame()

    #reads each subdirectory of the directory given and creates a folder with the name of the subdirectory at the output
    # Loop through each subdirectory in the input directory
    for subdir in os.listdir(directory):

        #transform the DICOM files in NIFTI files
        #os.sytem(f'dcm2niix -z y -f {op.basename(nifti_file)} -o {output_dir} {dicom_dir}')
        
        #reads the DICOM files from the list of files and sub directories of the directory given
        for dir, dirs, files in os.walk(directory, topdown=False):
        #print(dir)
            for name in files:
                #print(dir + '/' + name)
                #print("\n______----------------_______\n")
                #check if is a DICOM file
                if name.endswith(".dcm"):
                    #patient_list = read_dicom(dir + '/' + name,patient_list)
                    total_df = read_dicom2(dir + '/' + name,total_df,output_name)
                #print("\n**********\n\n**********\n\n**********\n")
        #if total_df.empty:
            #the columns are defined by the tags
        #    total_df = df.copy()
        #else:
            #total_df = pd.concat([total_df, df], ignore_index=True)
    #total_df = total_df.drop_duplicates()
    #df.loc[df.astype(str).drop_duplicates().index]
    total_df.to_csv(op.join(output_dir_dataset, output_name + '.csv'), header=True)

    #for each PatientID it creates a folder with the name of the PatientID
    for patientid in total_df["PatientID"].unique():
        new_dir = os.path.join(output_dir_dataset, patientid)
        os.makedirs(new_dir, exist_ok=True)
        #for each modality creates a csv file with the information of the patient
        df_patient = total_df[total_df["PatientID"] == patientid]
        for series_description in df_patient['SeriesDescription'].unique():
            df_patient[df_patient['SeriesDescription'] == series_description].to_csv(op.join(new_dir, series_description + '.csv'), header=True)
            df_patient[df_patient['SeriesDescription'] == series_description].to_json(op.join(new_dir, series_description + "_read" + '.json'),orient="records",lines=True)



def read_dicom3(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    tags_values = [(tag, ds.data_element(tag).value) if tag in ds else (tag,'') for tag in NEEDED_TAGS]
    dataset = pd.DataFrame(columns=[tag for tag, value in tags_values])
    dataset.loc[len(dataset.index)] = [value for tag, value in tags_values]
    return dataset



def read_list_files3(directory, output,filter,notdcm2nix,tojson):
    filter_list = filter.split(",")
    print(filter_list)
    #gets only the name of the subdirectory
    output_name = os.path.basename(directory)
    output_dir_dataset = op.join(output, output_name)
    os.makedirs(output_dir_dataset, exist_ok=True)
    patient_list = list()
    #creates empty pandas dataframe
    df = pd.DataFrame()
    total_df = pd.DataFrame()

    #reads each subdirectory of the directory given and creates a folder with the name of the subdirectory at the output
    # Loop through each subdirectory in the input directory
    for subdir in os.listdir(directory):

        #reads each file of in all the subdirectories of the directory given
        for dir, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                #check if is a DICOM file and if all the filters are in the path
                if name.endswith(".dcm") and all(x in (dir + '/' + name) for x in filter_list):
                    #reads the directory of the DICOM file, being the last directory the seriesid,the second to last the studyid and the third to last the patientid
                    #the directory is split by the character "/"
                    dir_split = os.path.normpath(dir).split(os.sep)
                    #the patientid is the third to last element of the list
                    patientid = dir_split[-3]
                    #the studyid is the second to last element of the list
                    studyid = dir_split[-2]
                    #the seriesid is the last element of the list
                    seriesid = dir_split[-1]
                    #total_df = read_dicom2(dir + '/' + name,total_df,output_name)
                    line = read_dicom3(dir + '/' + name)
                    line['PatientID'] = patientid
                    line['StudyID'] = studyid
                    line['SeriesID'] = seriesid
                    line["Path"] = dir
                    #total_df = total_df.append(line)
                    total_df = pd.concat([total_df, line])
                    total_df = total_df.loc[:,["PatientID","StudyID","SeriesID","Path"]]

    if not total_df.empty:
        print(total_df)
        #for each PatientID it creates a folder with the name of the PatientID
        for patientid in total_df["PatientID"].unique():
            new_dir = os.path.join(output_dir_dataset, patientid)
            os.makedirs(new_dir, exist_ok=True)

            #for each seriesid it stores in a csv file the information of the first dataframe line
            df = total_df.loc[(total_df["PatientID"] == patientid)]
            df = df.drop_duplicates(subset=['PatientID', 'SeriesID'], keep='first')
            df.to_csv(op.join(new_dir, "series.csv"), header=True)
            df.to_json(op.join(new_dir, 'series.json'),orient="records",lines=True)

            #creates a nifti file for each seriesid
            for seriesid in df['SeriesID'].unique():
                #takes the path of the seriesid
                path = df.loc[(df["SeriesID"] == seriesid), "Path"].iloc[0]
                #print(path)
                #print(new_dir)
                if not notdcm2nix:
                    #os.system(f'dcm2niix.exe -z y -f %p -o "{new_dir}" "{path}"')
                    os.system(f'dcm2niix -z y -f %p -o "{new_dir}" "{path}"')
                
        
        #

            
            total_df.to_csv(op.join(output_dir_dataset, output_name + '.csv'), header=True)
            if tojson:
                total_df.to_json(op.join(output_dir_dataset, output_name + '.json'),orient="records",lines=True)





def list_series(path,output):
    #for each directory and subdirectory of the path given it reads the first dicom file, stores it in a dataframe and prints it
    list_series = pd.DataFrame()
    for dir, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".dcm"):
                ds = pydicom.dcmread(dir + '/' + name)
                serie = ds.data_element("SeriesInstanceUID").value
                path = os.path.normpath(dir)
                list_series = list_series.append({'SeriesInstanceUID': serie, 'Path': path}, ignore_index=True)
                break
    print(list_series)
    #if output is a string
    if output != "":
        list_series.to_csv(op.join(output, 'serieslist' + '.csv') , header=True)