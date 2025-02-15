import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from mtn.data.data import FmriDataModule
from mtn.models.models import LitMLP
from mtn.utils import get_datetime, get_paths


@hydra.main(config_path="../configs", config_name="conf", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # paths
    paths = get_paths()
    expname = get_datetime(expname=config.expname)
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    # helpers
    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_auroc",
        filename="{epoch}-{val_auroc:.2f}",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
    )

    # data
    datamodule = FmriDataModule(
        batch_size=config.data.batch_size,
        quantile=config.data.quantile,
        rand_seed=config.data.rand_seed,
    )

    # model
    model = LitMLP(config)

    # fit wrapper
    trainer = L.Trainer(
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        max_epochs=config.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        # accelerator="cpu"
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
