import typer
from typing_extensions import Annotated
import tools
import process


app = typer.Typer()


@app.callback()
def callback():
    """
    Manage users CLI app.

    Use it with the create command.

    A new user with the given NAME will be created.
    """


"""
Example of use:
main.py split-dataset "input_dir" "output_dir" --train-proportion 0.3 --test-proportion 0.3 --json-file
"""
@app.command()
def split_dataset(input_dir: str,
                output_dir: str,
                train_proportion: float = typer.Option(0.8, help="Proportion of the dataset that will be used for training"),
                test_proportion: float = typer.Option(0.0, help="Proportion of the dataset that will be used for testing"),
                json_file: bool = typer.Option(False, help="If this option is activated, json files with the paths of each set will be created in the output_dir")):
    """
    Split the dataset in three parts, one for training, one for validation and other for testing.
    The proportion of the dataset that will be used for training and testing can be specified.
    The rest of the dataset will be used for validation.
    The dataset will be split in a random way, storing the train, validation and test files in different directories in the output_dir,
    unless the json_file option is activated, in which case the files will be stored in output_dir as json_file storing the paths to each case directory.
    """
    tools.split_dataset(input_dir, output_dir, train_proportion, test_proportion, json_file)



@app.command()
def train(json_file: str,
            output_dir: str,
            ):
    """
    Train a model with the dataset and parameter specified in the json_file.
    """
    process.train(json_file, output_dir)


@app.command()
def evaluate_model(json_file: str
            ):
    """
    Evaluate a model with the dataset and parameters specified in the json_file.
    """
    process.evaluate_model(json_file)



"""
@app.command()
def process(input: str,output: str,
             filter: str = typer.Option("", help="List of words (separated b commas (,)) which must have the file path to be processed"),
             notdcm2nix: bool = typer.Option(False, help="If this option is activated, the dicom files won`t be transformed to nifti format"),
             tojson: bool = typer.Option(False, help="If this option is activated, json files with the same information as the csv files will be created")):
    """
    #Reads the different series of dicom files in the directory and its subdirectories and stores a list with data about the series in a csv file at the output directory with the files transformed to nifti format.
    #Also can store the list in a csv file
"""
    print(f"The input directory is {input}")
    print(f"The output directory is {output}")
    if filter != "":
        print(f"And the words to filter are {filter}")
    
    rdcm.read_list_files3(input,output,filter,notdcm2nix,tojson)
"""
@app.callback()
def main():
    """
    Manage users in the awesome CLI app.
    """

if __name__ == "__main__":
    app()
