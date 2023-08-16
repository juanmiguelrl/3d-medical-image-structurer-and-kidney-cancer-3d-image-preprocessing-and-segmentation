import typer
import read_dcm as rdcm

app = typer.Typer()


@app.callback()
def callback():
    """
    Manage users CLI app.

    Use it with the create command.

    A new user with the given NAME will be created.
    """

@app.command()
def listseries(input: str, output: str = typer.Option("", help="Output directory where the csv file will be stored")):
    """
    Shows a list with the series of the dicom files in the input directory.
    Also can store the list in a csv file
    """
    rdcm.list_series(input, output)


@app.command()
def process(input: str,output: str,
             filter: str = typer.Option("", help="List of words (separated b commas (,)) which must have the file path to be processed"),
             notdcm2nix: bool = typer.Option(False, help="If this option is activated, the dicom files won`t be transformed to nifti format"),
             tojson: bool = typer.Option(False, help="If this option is activated, json files with the same information as the csv files will be created")):
    """
    Reads the different series of dicom files in the directory and its subdirectories and stores a list with data about the series in a csv file at the output directory with the files transformed to nifti format.
    Also can store the list in a csv file
    """
    print(f"The input directory is {input}")
    print(f"The output directory is {output}")
    if filter != "":
        print(f"And the words to filter are {filter}")
    
    rdcm.read_list_files3(input,output,filter,notdcm2nix,tojson)

@app.callback()
def main():
    """
    Manage users in the awesome CLI app.
    """
    print("asda0 ")

if __name__ == "__main__":
    app()
