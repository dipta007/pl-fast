import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from dataloaders.dummy_datamodule import DummyDataModule
from models.dummy_model import DummyModel
import base_config as config
from utils import get_git_commit_hash
from pprint import pprint
import shutil
import wandb

wandb.login()

print("="*100)
print("===> Config")
config = config.get_config()
config.git_commit_hash = get_git_commit_hash()
pprint(vars(config))
print("="*100)


def main():
    L.seed_everything(config.seed)

    monitoring_metric = config.monitoring_metric
    monitoring_mode = config.monitoring_mode
    checkpoint_dir = f"{config.checkpoint_dir}/{config.exp_name}"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model_{}={{{}:.2f}}".format(monitoring_metric.replace("/", "_"), monitoring_metric),
            auto_insert_metric_name=False,
            monitor=f"{monitoring_metric}",
            mode=monitoring_mode,
            verbose=True,
            save_top_k=1,
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="model_final",
            verbose=True,
            save_top_k=-1,
            every_n_epochs=1,
            enable_version_counter=False,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(leave=True),
    ]

    if config.early_stopping_patience != -1:
        callbacks.append(
            EarlyStopping(
                monitor=monitoring_metric,
                patience=config.early_stopping_patience,
                mode=monitoring_mode,
                verbose=True,
            )
        )

    loggers = [
        CSVLogger(checkpoint_dir, name=config.exp_name)
    ]
    if not config.debug:
        wandb_logger = WandbLogger(
            entity=config.wandb_entity,
            project=config.wandb_project,
            log_model=False,
            name=config.exp_name if config.exp_name != "sweep" else None,
            config=vars(config),
        )
        wandb.experiment.define_metric("val/dummy", summary="max")
        loggers.append(wandb_logger)

    print("Loading data")
    datamodule = DummyDataModule(config)

    print("Loading model")
    model = DummyModel(config)

    print("Training")
    strategy = "ddp_find_unused_parameters_true" if config.ddp else 'auto'
    trainer = pl.Trainer(
        devices=config.devices,
        accelerator="auto",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        val_check_interval=config.validate_every,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.grad_accumulation_step,
        log_every_n_steps=1,
        overfit_batches=config.overfit_batches,
        precision=config.precision,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_val,
    )

    print("Fitting")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
