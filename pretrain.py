import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from transformers import BertTokenizer

from data import BertPretrainingDataset, BertPretrainingCollator
from trainer import Trainer
from bert_modelling import BertForPretraining
from utils import load_config
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, random_split


def main():
    """Main function to train and validate the transformer model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config_module = load_config(args.config)
    config = config_module.Config()  # instantiate your Config class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
    
    # Load dataset
    hf_dataset = load_from_disk(config.dataset_path)
    
    dataset = BertPretrainingDataset(
        hf_dataset,
        tokenizer,
        max_seq_length=config.max_len,
        mlm_prob=config.mlm_prob,
        mask_prob=config.mask_prob,
        random_prob=config.random_prob,
        same_prob=config.same_prob,
        batched=True,
        num_proc=config.num_proc,
        device=device,
    )
    # Define split sizes
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    dataset_collator = BertPretrainingCollator(
        pad_token_id=tokenizer.pad_token_id,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        collate_fn=dataset_collator,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        shuffle=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        collate_fn=dataset_collator,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    
    # Initialize model
    model = BertForPretraining(
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        d_ff=config.d_ff,
        max_len=config.max_len,
        pd=config.pd,
        norm_eps=config.norm_eps,
        initialize=config.initialize,
        add_pooler=config.add_pooler,
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.optim_eps,
        weight_decay=config.l2_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.warmup_steps,
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        checkpoint_dir=config.checkpoint_path,
        save_best_only=True,
        early_stopping_patience=None,
        metric_fn=None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # Start training
    trainer.train()
        
if __name__ == "__main__":
    main()