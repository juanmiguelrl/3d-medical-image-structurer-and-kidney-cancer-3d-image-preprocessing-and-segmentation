# U-NET from processing kidney medical images from KITS21

To use this program there are 3 options:



## split_dataset
This option divides the given dataset in different sets (either by storing them in different folders or by creating a json file with the paths to the images)
This option can be called as in the following example:

main.py split-dataset "input_dir" "output_dir" --train-proportion 0.3 --test-proportion 0.3 --json-file

The parameters to input are:
- input_dir: path to the folder containing the images
- output_dir: path to the folder where the images will be stored
- train-proportion: proportion of the dataset that will be used for training
- test-proportion: proportion of the dataset that will be used for testing
- json-file: if this flag is used, the program will create a json file with the paths to the images

## train
This option trains the model with the given dataset.It uses a json file to input all the parameters needed, which will be explained later .It can be called as in the following example:
main.py train ""input_json_with_parameters" "output_dir"

The parameters to input are:
- input_json_with_parameters: path to the json file containing the parameters
- output_dir: path to the folder where the model will be stored


## evaluate_model
This option evaluates the model with the given dataset.It uses a json file to input all the parameters needed, which will be explained later .It can be called as in the following example:
main.py evaluate-model "input_json_with_parameters"

The parameters to input are:
- input_json_with_parameters: path to the json file containing the parameters

# JSON file

## train

json_file: boolean that indicates if the dataset sets are in json files or is stored in different folders
label_name: name of the label file
### json_data: dictionary with the paths to the json files containing the paths to the images(in case json_file is true)
&emsp;train_json: path to the json file containing the paths to the images for training
&emsp;val_json: path to the json file containing the paths to the images for validation
&emsp;test_json: path to the json file containing the paths to the images for testing
### dir_data: dictionary with the paths to the folders containing the images (in case json_file is false)
&emsp;train_dir: path to the folder containing the images for training
&emsp;val_dir: path to the folder containing the images for validation
&emsp;test_dir: path to the folder containing the images for testing
### wandb_enabled: boolean that indicates if wandb will be used (at the moment it is not available the option to not use it)
### wandb: dictionary with the parameters for wandb
&emsp;project: name of the project in wandb
&emsp;description: description of the project in wandb
### config: dictionary with the parameters for the training
&emsp;cache_rate: proportion of the dataset that will be cached
&emsp;num_workers: number of workers for the dataloader
&emsp;train_batch_size: batch size for training
&emsp;val_batch_size: batch size for validation
&emsp;learning_rate: learning rate for the optimizer
&emsp;max_epochs: maximum number of epochs
&emsp;val_interval: interval between validations
&emsp;lr_scheduler: type of learning rate scheduler
&emsp;model_type: type of model (at the moment only unet is available)
### &emsp;model_params: dictionary with the parameters for the model
&emsp;&emsp;spatial_dims: number of dimensions of the images
&emsp;&emsp;in_channels: number of channels of the images
&emsp;&emsp;out_channels: number of channels of the output
&emsp;&emsp;channels: list with the number of channels for each layer
&emsp;&emsp;strides: list with the strides for each layer
&emsp;&emsp;num_res_units: number of residual units
&emsp;&emsp;norm: type of normalization
### eval: dictionary with the parameters for the evaluation
&emsp;model_path: path to the model
&emsp;case_num: number of the case to evaluate
&emsp;slice_num: number of the slice to evaluate
&emsp;cache_rate: proportion of the dataset that will be cached
&emsp;num_workers: number of workers for the dataloader
&emsp;batch_size: batch size
### &emsp;model_params: dictionary with the parameters for the model (same as in train)
&emsp;&emsp;spatial_dims: number of dimensions of the images
&emsp;&emsp;in_channels: number of channels of the images
&emsp;&emsp;out_channels: number of channels of the output
&emsp;&emsp;channels: list with the number of channels for each layer
&emsp;&emsp;strides: list with the strides for each layer
&emsp;&emsp;num_res_units: number of residual units
&emsp;&emsp;norm: type of normalization



## example of JSON file
    
```json
{
    "json_file": true,
    "label_name": "aggregated_MAJ_seg.nii.gz",
    "json_data": {
        "train_json": "C:\\Users\\Desktop\\folder\\test\\train.json",
        "val_json": "C:\\Users\\Desktop\\folder\\test\\val.json",
        "test_json": "C:\\Users\\Desktop\\folder\\test\\test.json"
    },
    "dir_data": {
        "train_dir": "C:\\Users\\Desktop\\folder\\test\\train",
        "val_dir": "C:\\Users\\Desktop\\folder\\test\\val",
        "test_dir": "C:\\Users\\Desktop\\folder\\test\\test"
    },

    "wandb_enabled": true,
    "wandb": {
        "project": "kidney_segmentation",
        "description": "Unet for 3D kidney segmentation"
    },

    "config": {
        "cache_rate": 1.0,
        "num_workers": 1,



        "train_batch_size": 1,
        "val_batch_size": 1,
        "learning_rate": 1e-3,
        "max_epochs": 5,
        "val_interval": 2,
        "lr_scheduler": "cosine_decay",




        "model_type": "unet",
        "model_params": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 4,
            "channels": [64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "INSTANCE"
        }

    },

    "eval": {
        "model_path": "C:\\Users\\Desktop\\folder\\out\\best_metric_model.pth",
        "case_num": 0,
        "slice_num": 60,
        "cache_rate": 1.0,
        "num_workers": 1,
        "batch_size": 1,
        "model_params": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 4,
            "channels": [64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "INSTANCE"
        }
    }
}
```


