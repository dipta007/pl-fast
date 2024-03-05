import argparse

def none_or_str(value):
    if value == "None":
        return None
    return value

def array_or_str(value):
    if value == "auto":
        return value
    return [int(x) for x in value.split(",")]

def none_or_float(value):
    if value == "None":
        return None
    return float(value)

def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)

def add_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, required=False, default=False, help='debug?')
    parser.add_argument("--seed", type=int, default=42, help="value for reproducibility") 
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use CUDA?")
    parser.add_argument("--checkpoint_dir", type=str, default="/nfs/ada/ferraro/users/sroydip1/dummy/checkpoints/", help="Checkpoint directory")
    parser.add_argument("--wandb_entity", type=str, default="gcnssdvae", help="Wandb entity")
    parser.add_argument("--wandb_project", type=str, default="dummy", help="Wandb project")
    return parser

def add_data_args(parent_parser):
    parser = parent_parser.add_argument_group("Data Config")
    parser.add_argument("--data_dir", type=str, default="./data/steps_", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size?")
    parser.add_argument("--grad_accumulation_step", default=2, type=int, help="Number of accumulation of grad batches")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers?")
    return parent_parser

def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model Config")
    parser.add_argument("--exp_name", type=str, default="dummy", help="Experiement name?", required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name?")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay?")
    return parent_parser

def add_trainer_args(parent_parser):
    parser = parent_parser.add_argument_group("Trainer Config")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Number of maximum epochs", )
    parser.add_argument("--validate_every", type=int_or_float, default=100, help="Number of maximum epochs", )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--overfit_batches", default=0, type=int_or_float, help="Overfit batches")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience? -1 if no early stopping")
    parser.add_argument("--monitoring_metric", type=str, default="valid/loss", help="Monitoring metric")
    parser.add_argument("--monitoring_mode", type=str, default="min", help="Monitoring mode")
    parser.add_argument("--devices", type=array_or_str, default="auto", help="Devices")
    parser.add_argument("--ddp", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use DDP?")
    parser.add_argument("--precision", type=str, default="32-true", help="Trainer Precision?")
    parser.add_argument("--gradient_clip_algorithm", type=none_or_str, default=None, help="Gradient clip algorithm")
    parser.add_argument("--gradient_clip_val", type=none_or_float, default=None, help="Gradient clip value")
    return parent_parser

def get_config():
    parser = add_program_args()
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_trainer_args(parser)
    cfg    = parser.parse_args()
    return cfg