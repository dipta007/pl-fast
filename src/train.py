import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
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

    loggers = False
    if not config.debug:
        loggers = [
            WandbLogger(
                entity=config.wandb_entity,
                project=config.wandb_project,
                log_model=False,
                name=config.exp_name if config.exp_name != "sweep" else None,
                config=vars(config),
            )
        ]
        loggers[0].experiment.define_metric("val/dummy", summary="max")

    print("Loading data")
    datamodule = DummyDataModule(config)

    print("Loading model")
    model = DummyModel(config)

    print("Training")
    accumulate_grad_batches = (
        config.accumulate_grad_batches // config.batch_size
        if config.batch_size < config.accumulate_grad_batches and config.accumulate_grad_batches != -1
        else 1
    )

    strategy = "ddp_find_unused_parameters_true" if config.ddp else 'auto'
    trainer = pl.Trainer(
        accelerator="auto",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        val_check_interval=config.validate_every,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=1,
        overfit_batches=config.overfit if config.overfit != 0 else 0.0,
    )

    print("Fitting")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
