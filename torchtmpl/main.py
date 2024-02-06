# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import tqdm
import pandas as pd

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    #device = torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        #wandb.login(key=wandb_config["key"]) # pour weight bias aaron team geolifeclef_aaron_julien_olivier
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info("Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_sizes, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_sizes, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])
    print(loss)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        print(logdir)
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    image_input_size = next(iter(train_loader))[0]["image"].shape
    feature_input_size = next(iter(train_loader))[0]["features"].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + "### Image model\n"
        + f"{torchinfo.summary(model.image_model, input_size=image_input_size)}\n\n"
        + "### Features model\n"
        + f"{torchinfo.summary(model.features_model, input_size=feature_input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)


def test(config):
    """
    Load le model torch.load(model_testing)
    Load le dataloader mais juste pour le testset -> recréer dataloader ou passer en arg que c'est le test.
    model.eval()
    model(donees)  = forward
    
    faire une fonction qui calcul le top 30 -> permet d'avoir une fct loss qu'on utilisera sur le validation set
    sortir un fichier avec le top 30
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    
    data_config = config["data"]
    test_loader, input_sizes, num_classes = data.get_test_dataloader(data_config, use_cuda)

    model_config = config["model"]
    model = models.build_model(config["model"], input_sizes, num_classes)
    model.load_state_dict(torch.load(model_config["path_to_test_model"]), strict=False)
    model.eval()
    model.to(device)

    top_30 = {"Id": [], "Predicted": []}

    for i, (inputs, observations) in (pbar := tqdm.tqdm(enumerate(test_loader))):

        image_inputs, features_inputs = inputs["image"].to(device), inputs["features"].to(device)

        # Compute the forward propagation
        output_batch = model((image_inputs, features_inputs))
        species_id = utils.get_top_30(output_batch)

        top_30["Id"] += observations
        top_30["Predicted"] += species_id.detach().cpu().numpy().tolist()
    
    top_30 = pd.DataFrame(top_30)
    top_30["Predicted"] = top_30["Predicted"].apply(lambda x: " ".join(map(str, x)))
    top_30.set_index('Id', inplace=True)
    top_30.to_csv('sample_submission_testings.csv')
    print("CSV done")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
