import os
from glob import glob
import json
import wandb
import datetime

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    RandShiftIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    #GammaTransformd,
    RandZoomd,
    Orientationd,
    EnsureType,
    AsDiscrete,
)

from monai.data import Dataset, DataLoader,CacheDataset, decollate_batch
from monai.utils import first
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.inferers import sliding_window_inference

import numpy as np


#get datasets from its directories
def get_datasets_dir(train_dir,val_dir,test_dir=None,label_name="aggregated_MAJ_seg.nii.gz"):

    train_images = sorted(glob(os.path.join(train_dir, "case_*/imaging.nii.gz")))
    train_labels = sorted(glob(os.path.join(train_dir, "case_*/"+label_name)))

    val_images = sorted(glob(os.path.join(val_dir, "case_*/imaging.nii.gz")))
    val_labels = sorted(glob(os.path.join(val_dir, "case_*/"+label_name)))

    test_images = sorted(glob(os.path.join(test_dir, "case_*/imaging.nii.gz")))
    test_labels = sorted(glob(os.path.join(test_dir, "case_*/"+label_name)))

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


#gets datasets from json files
def get_datasets_json(train_json,val_json,test_json=None,label_name="aggregated_MAJ_seg.nii.gz"):

    with open(train_json) as f:
        train_list = json.load(f)
    with open(val_json) as f:
        val_list = json.load(f)
    with open(test_json) as f:
        test_list = json.load(f)

    train_images = [os.path.join(case,"imaging.nii.gz") for case in train_list]
    train_labels = [os.path.join(case,label_name) for case in train_list]

    val_images = [os.path.join(case,"imaging.nii.gz") for case in val_list]
    val_labels = [os.path.join(case,label_name) for case in val_list]

    test_images = [os.path.join(case,"imaging.nii.gz") for case in test_list]
    test_labels = [os.path.join(case,label_name) for case in test_list]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels



def get_datasets(data):
    if data["json_file"]:
        json_data=data["json_data"]
        train_images, train_labels, val_images, val_labels, test_images, test_labels = get_datasets_json(json_data["train_json"],json_data["val_json"],json_data["test_json"],data["label_name"])
    else:
        dir_data=data["json_data"]
        train_images, train_labels, val_images, val_labels, test_images, test_labels = get_datasets_dir(dir_data["train_dir"],dir_data["val_dir"],dir_data["test_dir"],data["label_name"])

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

#reads the parameters stored at a json file
def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def get_files(images, labels):
    files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(images, labels)]
    #val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
    #test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]

    return files


orig_transforms = Compose(

    [
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        
        ToTensord(keys=['image', 'label'])
    ]
)

train_transforms = Compose(

    [


        #load image and label
        LoadImaged(keys=['image', 'label']),
        #makes the image format to have furst the num_channel, like from (spatial_dim_1[, spatial_dim_2, …]) to (num_channels, spatial_dim_1[, spatial_dim_2, …])
        AddChanneld(keys=['image', 'label']),
        #Ensures that the data are in format that the channel is first -> (C,H,W) instead of (H,W,C)
        #EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
        #Must be used before any anisotropiic spatial transform, it assures that the images are in the standar RAS (right,anterior,superior) orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #Scales the intensity of the image to the given range (normalization)
        ScaleIntensityRanged(keys=["image"], a_min=-80, a_max=305,b_min=0.0, b_max=1.0, clip=True),
        #it removes the background
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        #Resizes the image to the given spatial size
        #Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
        #Randomly crops the image to the given spatial size taking into account the label
        RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(160, 160, 64),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        #Randomly elasticly deforms the image
        Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                sigma_range=(5, 8),
                magnitude_range=(50, 150),
                spatial_size=(160, 160, 64),
                translate_range=(10, 10, 5),
                rotate_range=(np.pi/36,np.pi/36, np.pi),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="zeros",
            ),
        #Randomly shifts the intensity of the image
        RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.25,
            ),
        #Randomly adds gaussian noise to the image
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),

        #these transformations randomly flip the images in different orientations
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
    
        #randomly rotates the images 90 degrees
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),

    #problema -> no hay una transformacion aleatoria de gamma -> o la creo o la aplico a todas??
        #GammaTransformd(keys=["image"], gamma_range=(0.7, 1.3)),
        
        #randomly makes a zoom to the image
        RandZoomd(keys=["image", "label"], prob=0.1, zoom_range=(0.9,1.1)),
        #it ensures that the input data is a pytorch tensor or a numpy array
        EnsureTyped(keys=["image", "label"]),
        #it transforms the data to a tensor
        ToTensord(keys=['image', 'label'])
    ]
)

#the transforms are described at the train_transforms block of code
val_transforms = Compose(

    [
        #load image and label
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        #Ensures that the data are in format that the channel is first -> (C,H,W) instead of (H,W,C)
        #EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
    
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #Scales the intensity of the image to the given range
        ScaleIntensityRanged(keys=["image"], a_min=-80, a_max=305,b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        EnsureTyped(keys=["image", "label"]),

        ToTensord(keys=['image', 'label'])
    ]
)

def get_dataloaders(files,transform,batch_size=2,cache_rate=1.0,num_workers=0,shuffle=True):
    #orig_ds = Dataset(data=train_files, transform=orig_transforms)
    #orig_loader = DataLoader(orig_ds, batch_size=2)

    #ds = Dataset(data=files, transform=transform)
    ds = CacheDataset(data=files, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
    #train_loader = DataLoader(ds, batch_size=2)
    #loader = DataLoader(ds, batch_size, shuffle=shuffle, num_workers=num_workers)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #val_ds = Dataset(data=val_files, transform=val_transforms)
    #val_loader = DataLoader(val_ds, batch_size=2)

    return loader, ds


def train(json_file,output_dir):
    data = read_json(json_file)
    config = data['config']

    train_images, train_labels, val_images, val_labels, test_images, test_labels = get_datasets(data)
    train_files = get_files(train_images, train_labels)
    val_files = get_files(val_images, val_labels)
    #test_files = get_files(test_images, test_labels)
    train_loader, train_ds = get_dataloaders(train_files,train_transforms,batch_size=config['train_batch_size'],cache_rate=config['cache_rate'],
                                    num_workers=config['num_workers'],shuffle=True)
    val_loader, val_ds = get_dataloaders(val_files,val_transforms,batch_size=config['val_batch_size'],cache_rate=config['cache_rate'],
                                    num_workers=config['num_workers'],shuffle=False)
    #test_loader = get_dataloaders(test_files,test_transforms,batch_size=2)


    



    #train and neural network parameters configuration

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**config['model_params']).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-9)

    #wandb configuration
    if data["wandb_enabled"]:
        #prepare for uploading the data to wandb
        # 🐝 initialize a wandb run
        wandb.init(
            project=data["wandb"]["project"],
            config=config
        )

        # 🐝 log gradients of the model to wandb
        wandb.watch(model, log_freq=5)

    max_epochs = config['max_epochs']
    val_interval = config['val_interval']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    #these post_pred and post_label are used in the evaluation of the model to see how well the model performs during the training
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=config["model_params"]["out_channels"], n_classes=config["model_params"]["out_channels"])])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=config["model_params"]["out_channels"], n_classes=config["model_params"]["out_channels"])])



    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
            
            # 🐝 log train_loss for each step to wandb
            wandb.log({"train/loss": loss.item()})
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # step scheduler after each epoch (cosine decay)
        scheduler.step()
        
        # 🐝 log train_loss averaged over epoch to wandb
        wandb.log({"train/loss_epoch": epoch_loss})
        
        # 🐝 log learning rate after each epoch to wandb
        wandb.log({"learning_rate": scheduler.get_lr()[0]})

        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160,160, 64)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # 🐝 aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # 🐝 log validation dice score for each validation round
                wandb.log({"val/dice_metric": metric})

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        output_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"\ntrain completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

    # 🐝 log best score and epoch number to wandb
    wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})

    # 🐝 Version your model
    best_model_path = os.path.join(output_dir, "best_metric_model.pth")
    model_artifact = wandb.Artifact(
                "unet", type="model",
                description=data["wandb"]["description"],
                metadata=dict(config['model_params']))
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)


    #evaluation of the model
    import plotly

    eval_num = val_interval

    # Crear figura y subfiguras
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Graficar pérdida promedio por iteración
    axs[0].plot([eval_num * (i + 1) for i in range(len(epoch_loss_values))], epoch_loss_values)
    axs[0].set_title("Iteration Average Loss")
    axs[0].set_xlabel("Iteration")

    # Graficar métrica promedio en validación por iteración
    axs[1].plot([eval_num * (i + 1) for i in range(len(metric_values))], metric_values)
    axs[1].set_title("Val Mean Dice")
    axs[1].set_xlabel("Iteration")

    # Mostrar gráfico
    plt.show()

    # Crear imagen para W&B
    #img = wandb.Image(fig)
    #store the plots in wandb
    wandb.log({"iteration average loss & val mean dice": wandb.Image(fig)})


    #store the model with date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "best_metric_model_" + current_time + ".pth"
    torch.save(model.state_dict(), os.path.join(output_dir, filename))


def evaluate_model(json_file):

    data = read_json(json_file)
    model_path = data["eval"]['model_path']
    case_num = data["eval"]['case_num']
    config = data['config']


    #wandb configuration
    if data["wandb_enabled"]:
        #prepare for uploading the data to wandb
        # 🐝 initialize a wandb run
        wandb.init(
            project=data["wandb"]["project"],
            config=config

        )


    _, _, val_images, val_labels, _, _ = get_datasets(data)
    #train_files = get_files(train_images, train_labels)
    val_files = get_files(val_images, val_labels)
    #test_files = get_files(test_images, test_labels)
    #train_loader, train_ds = get_dataloaders(train_files,train_transforms,data['cache_rate'],
    #                                num_workers=data['num_workers'],batch_size=data['train_batch_size'],shuffle=True)
    #_, val_ds = get_dataloaders(val_files,val_transforms,batch_size=data["eval"]['batch_size'],cache_rate=data["eval"]['cache_rate'],
    #                            num_workers=data["eval"]['num_workers'],shuffle=False)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=data["eval"]['cache_rate'], num_workers=data["eval"]['num_workers'])
    #test_loader = get_dataloaders(test_files,test_transforms,batch_size=2)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**data["eval"]['model_params']).to(device)

    model.load_state_dict(torch.load(os.path.join(model_path)))
    model.eval()
    with torch.no_grad():
        n = data["eval"]["slice_num"]
        #img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        val_labels = torch.unsqueeze(label, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)


        # creates the figure and subfigures
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        #plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        axs[0].set_title("image")
        axs[0].imshow(val_inputs.cpu().numpy()[0, 0, :, :, n], cmap="gray")
        plt.subplot(1, 3, 2)
        axs[1].set_title("label")
        axs[1].imshow(val_labels.cpu().numpy()[0, 0, :, :, n])
        plt.subplot(1, 3, 3)
        axs[2].set_title("output")
        axs[2].imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, n])
        plt.show()
        if data["wandb_enabled"]:
            wandb.log({"Label prediction made by the model": wandb.Image(fig)})
