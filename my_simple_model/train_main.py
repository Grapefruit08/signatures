import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import yaml

from data_modules.signature_data_module import SignatureDataModule
from model_modules.mobilenet_model import MobileNetModel
from model_modules.simple_model import SimpleBaseModel

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    # load config
    experiment_name = "ninth_experiment"  # YOLO model experiment
    config_path = f"experiments\\{experiment_name}\\config.yaml"
    config = load_config(config_path)

    # create output directory next to config file
    config_dir = Path(config_path).parent
    output_dir = config_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Data module
    data_module = SignatureDataModule(
        train_csv=config['data']['train_csv'],
        test_csv=config['data']['test_csv'],
        images_dir=config['data']['images_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['model']['input_size'])
    )
    
    # Model selection based on config
    model_type = config['model'].get('model_type', 'simple')
    if model_type == 'mobilenet':
        model = MobileNetModel(
            learning_rate=config['training']['learning_rate'],
            pretrained=config['model'].get('pretrained', True),
            freeze_backbone=config['model'].get('freeze_backbone', True)
        )
        print("Using MobileNetV3 Large Model")
    else:
        model = SimpleBaseModel(
            learning_rate=config['training']['learning_rate']
        )
        print("Using Simple Baseline Model")

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=config['output']['experiment_name']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=2,
        filename='{epoch:02d}-{val_loss:.4f}',
        save_last=False
    )
    
    # NO early stopping for fifth experiment - let it run full 10 epochs
    early_stopping = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=10,  # Much more patience for learning
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    
    print("Starting training")
    
    # Fit model
    trainer.fit(model, data_module)
    
    # Save final model
    final_model_path = output_dir / config['output']['experiment_name'] / f"version_{trainer.logger.version}" / "final_model.pt"
    trainer.save_checkpoint(final_model_path)
    
    print(f"\n Training completed!")
    print(f" Best model: {checkpoint_callback.best_model_path}")
    print(f" Final model: {final_model_path}")
    print(f" TensorBoard logs: {logger.log_dir}")
    print(f" To visualize, run: tensorboard --logdir {output_dir}")


if __name__ == '__main__':
    main()
