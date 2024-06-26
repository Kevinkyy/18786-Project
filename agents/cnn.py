import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale
import numpy as np
import time
from os import path
from typing import Dict, Optional, Tuple, Union
import torchmetrics

from utils import SuperTuxImages, load_model, save_model, set_seed, load_data, save_checkpoint, load_checkpoint

# Constants
PREFIX_TRAINING_PARAMS = "tr_par_"

def make_downsample_layer(n_input, n_output, stride):
    if stride != 1 or n_input != n_output:
        return nn.Sequential(
            nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(n_output)
        )
    return None

class BlockConv(nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=1, residual=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, bias=False),
            nn.BatchNorm2d(n_output),
            nn.ReLU(),
            nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=False),
            nn.BatchNorm2d(n_output),
            nn.ReLU()
        )
        self.residual = residual
        self.downsample = make_downsample_layer(n_input, n_output, stride)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv(x)
        return x + residual if self.residual else x

class BlockUpConv(nn.Module):
    def __init__(self, n_input, n_output, stride=1, residual=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(n_output),
            nn.ReLU(),
            nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=stride, output_padding=1, bias=False),
            nn.BatchNorm2d(n_output),
            nn.ReLU(),
            nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(n_output),
            nn.ReLU()
        )
        self.residual = residual
        self.upsample = None
        if stride != 1 or n_input != n_output:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(n_input, n_output, kernel_size=1, stride=stride, output_padding=1, bias=False),
                nn.BatchNorm2d(n_output)
            )

    def forward(self, x):
        if self.residual:
            identity = x if self.upsample is None else self.upsample(x)
            return self.net(x) + identity
        else:
            return self.net(x)

class KartCNN(nn.Module):
    def __init__(self, dim_layers=[16, 32, 64, 128], n_input=3, input_normalization=True, residual=True, input_dim=(96,128), hidden_dim=128, **kwargs):
        super().__init__()
        self.dict_model = locals().copy()
        del self.dict_model['self']
        n_output = n_input
        self.norm = nn.BatchNorm2d(n_input) if input_normalization else None
        self.min_size = 2 ** (len(dim_layers)) * 4

        c = dim_layers[0]
        list_encoder = [nn.Sequential(
            nn.Conv2d(n_input, c, kernel_size=7, padding=3, stride=4, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU()
        )]
        list_decoder = [nn.Sequential(
            nn.ConvTranspose2d(c, n_output, kernel_size=7, padding=3, stride=4, output_padding=3),
            nn.BatchNorm2d(n_output),
            nn.ReLU()
        )]
        for l in dim_layers[1:]:
            list_encoder.append(BlockConv(c, l, stride=2, residual=residual))
            list_decoder.insert(0, BlockUpConv(l, c, stride=2, residual=residual))
            c = l
        
        in_linear_dim = (np.asarray(input_dim) // (2 ** (len(dim_layers) - 1)) // 4).astype(int).tolist()
        in_linear_size = np.prod(in_linear_dim) * c

        list_encoder.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_linear_size, hidden_dim),
            nn.ReLU(),
        ))
        list_decoder.insert(0, nn.Sequential(
            nn.Linear(hidden_dim, in_linear_size),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=[c] + in_linear_dim),
        ))

        self.encoder = nn.Sequential(*list_encoder)
        self.decoder = nn.Sequential(*list_decoder)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)

        h, w = x.shape[2:]
        if h < self.min_size or w < self.min_size:
            resize = torch.zeros(x.size(0), x.size(1), max(self.min_size, h), max(self.min_size, w), device=x.device)
            resize[:, :, :h, :w] = x
            x = resize

        x = self.encoder(x)
        x = self.decoder(x)
        return x[:, :, :h, :w]

    def train_model(
        self,
        save_name:str = None,
        dataset_path: str = 'data',
        checkpoint_path:str = None,
        seed:int = 1234,
        log_dir: str = './logs',
        save_path: str = './saved/ae',
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        n_epochs: int = 60,
        batch_size: int = 32,
        num_workers: int = 4,
        scheduler_mode: str = 'min_val_loss',
        debug_mode: bool = False,
        steps_save: int = 1,
        use_cpu: bool = False,
        device=None,
        scheduler_patience: int = 10,
        train_transform=Compose([
            Grayscale(),
            RandomHorizontalFlip(0.5),
        ]),
        test_transform=Grayscale(),
    ):
        model = self
        dict_model = self.dict_model

        # cpu or gpu used for training if available (gpu much faster)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
        print(device)

        # Set seed
        set_seed(seed)

        # Tensorboard
        # dictionary of training parameters
        dict_param = {f"{PREFIX_TRAINING_PARAMS}{k}": v for k, v in locals().items() if k in [
            "seed",
            "lr",
            "optimizer_name",
            "n_epochs",
            "batch_size",
            "num_workers",
            "scheduler_mode",
            "steps_save",
            "scheduler_patience",
            "train_transform",
            "test_transform",
        ]}

        # dictionary to set model name
        # name_dict = dict_model.copy()
        # name_dict.update(dict_param)
        # model name
        # name_model = '/'.join([
        #     str(name_dict)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
        # ])

        valid_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), f"valid_{save_name}"), flush_secs=1)
        train_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), f"train_{save_name}"), flush_secs=1)
        # valid_logger = train_logger
        # global_step = 0

        # Model
        dict_model.update(dict_param)
        model = model.to(device)

        # Loss
        loss = torch.nn.MSELoss().to(device)

        # load data
        dataset = SuperTuxImages(path=dataset_path, train_transform=train_transform, test_transform=test_transform)
        train_loader, val_loader, _ = load_data(dataset, batch_size, num_workers)

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise Exception("Optimizer not configured")

        if scheduler_mode in ["min_loss", 'min_val_loss']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)
        elif scheduler_mode in []:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience)
        else:
            raise Exception("Optimizer not configured")

        # Load checkpoint if given
        checkpoint = {}
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(
                path=checkpoint_path,
                optimizer=optimizer
            )
        old_epoch = checkpoint.get('epoch',-1) + 1
        global_step = old_epoch
        n_epochs += old_epoch

        # Initialize epoch timer
        tic = time.time()
        epoch_time_metric = torchmetrics.MeanMetric()

        for epoch in range(old_epoch, n_epochs):
            # for epoch in (p_bar := trange(n_epochs, leave = True)):
            # p_bar.set_description(f"{name_model} -> best in {dict_model['epoch']}: {dict_model['val_acc']}")

            # train_loss = []
            train_mse_metric = torchmetrics.MeanSquaredError(compute_on_cpu=True)
            train_mse_metric.to(device)

            # Start training: train mode
            model.train()
            dataset.test = False
            
            for d in train_loader:
                x,y = [k.to(device) for k in d]

                # Compute loss on training and update parameters
                pred = model(x)
                loss_train = loss(pred, y)

                # Do back propagation
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # Add train loss and accuracy
                # train_loss.append(loss_train.cpu().detach().numpy())
                train_mse_metric.update(pred,y)

            # Evaluate the model
            # val_loss = []
            val_mse_metric = torchmetrics.MeanSquaredError(compute_on_cpu=True)
            val_mse_metric.to(device)

            # Set evaluation mode
            model.eval()
            dataset.test = True

            with torch.no_grad():
                for d in val_loader:
                    x,y = [k.to(device) for k in d]

                    pred = model(x)

                    # Add loss and accuracy
                    # val_loss.append(loss(pred, y).cpu().detach().numpy())
                    val_mse_metric.update(pred, y)

            # calculate mean metrics
            # train_loss = np.mean(train_loss)
            train_loss = train_mse_metric.compute()
            train_mse_metric.reset()
            # val_loss = np.mean(val_loss)
            val_loss = val_mse_metric.compute()
            val_mse_metric.reset()
            
            # calculate time/epoch
            toc = time.time()
            epoch_time_metric.update(toc - tic, weight=epoch)   # give more weight to recent values
            epoch_time = epoch_time_metric.compute()
            tic = toc

            # Step the scheduler to change the learning rate
            # is_better = False
            # scheduler_info = SCHEDULER_MODES[scheduler_mode]
            # met =
            is_better = False
            if scheduler_mode == "min_loss":
                met = train_loss
                if (best_met := dict_model.get('train_loss', None)) is not None:
                    is_better = met < best_met
                else:
                    dict_model['train_loss'] = met
                    is_better = True
            elif scheduler_mode == "min_val_loss":
                met = val_loss
                if (best_met := dict_model.get('val_loss', None)) is not None:
                    is_better = met < best_met
                else:
                    dict_model['val_loss'] = met
                    is_better = True
            elif scheduler_mode is None:
                met = None
            else:
                raise Exception("Unknown scheduler mode")

            if met is not None:
                scheduler.step(met)

            # log metrics
            global_step += 1
            if train_logger is not None:
                # train log
                suffix = 'train'
                train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
                # log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)

                # validation log
                suffix = 'val'
                valid_logger.add_scalar(f'loss_{suffix}', val_loss, global_step=global_step)
                # log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)

                # learning rate log
                train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            # Save the model
            if (is_periodic := epoch % steps_save == steps_save - 1) or is_better:
                # d = dict_model if is_better else dict_model.copy()
                d = dict_model.copy()

                # training info
                d["epoch"] = epoch
                d['last_epoch_sec'] = epoch_time
                # metrics
                d["train_loss"] = train_loss
                d["val_loss"] = val_loss

                # name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
                if save_name is None:
                    name_path = int((np.random.rand() + train_loss) * 10000)
                else:
                    name_path = save_name    

                if is_better:
                    save_model(model, save_path, f"{name_path}_best", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_best", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"New best {epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.5f}, val loss {val_loss:0.5f}")

                # if periodic save, then include epoch
                if is_periodic:
                    save_model(model, save_path, f"{name_path}_{epoch + 1}", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_{epoch + 1}", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"{epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.5f}, val loss {val_loss:0.5f}")


MODEL_CLASS = dict(
    cnn=KartCNN,
)


if __name__ == '__main__':
    cnn = KartCNN(n_input=3)
    cnn.train_model(
        dataset_path='data',
        save_name="color1", 
        use_cpu=False, 
        num_workers=0,
        scheduler_patience=10,
        train_transform=Compose([
            RandomHorizontalFlip(0.5),
        ]),
        test_transform=None,
        steps_save=np.inf,
        batch_size=32,
        lr=1e-3,
        n_epochs=400,
        # checkpoint_path='./saved/ae/color2_best',
    )