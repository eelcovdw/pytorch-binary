import argparse
from pathlib import Path
import pytorch_lightning as pl

from vae import build_model
from data_utils import MNISTDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    x_size = 28 * 28
    log_dir = Path(args.output_dir) / "mnist" / f"{args.latent_type}-{args.latent_size}-{args.estimator}"
    
    data = MNISTDataModule(batch_size=args.batch_size, binarize=True)
    data.setup()
    model = build_model(x_size, args.latent_size, args.latent_type, args.estimator)

    # stopping_callback = EarlyStopping(monitor="val/log_marginal", mode="max")
    # chkpt_callback = ModelCheckpoint(monitor="val/log_marginal", mode="max")
    # callbacks=[chkpt_callback, stopping_callback]

    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, 
                         log_every_n_steps=10, progress_bar_refresh_rate=20, 
                         default_root_dir=log_dir)
    trainer.fit(model, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Lightning output directory", type=str, default="./runs/")
    parser.add_argument("--num_epochs", help="Number of epochs", type=int, default=200)
    parser.add_argument("--batch_size", help="Minibatch size", type=int, default=128)
    parser.add_argument("--latent_size", help="size of z", type=int, default=32)
    parser.add_argument("--latent_type", help="Type of prior and posterior.", default="normal")
    parser.add_argument("--estimator", help="Estimator used, pathwise|sfe|arm|disarm", default="pathwise")
    parser.add_argument("--num_gpus", help="number of GPUs", type=int, default=1)
    args = parser.parse_args()
    for arg in args.__dict__:
        print("{}: {}".format(arg, getattr(args, arg)))
    main(args)
