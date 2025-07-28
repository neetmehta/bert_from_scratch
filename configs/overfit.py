class Config:
    def __init__(self):
        # General
        self.max_len = 64  # Smaller for faster overfitting and lower memory

        # Tokenizer
        self.tokenizer_name = "bert-base-uncased"

        # Data
        self.dataset_path = "F:\\bert_from_scratch\\overfit"
        self.mlm_prob = 0.15
        self.mask_prob = 0.8
        self.random_prob = 0.1
        self.same_prob = 0.1
        self.num_proc = 1  # Fewer processes for debugging
        self.train_split = 0.02  # Use all for training
        self.val_samples = 0  # Skip validation entirely
        self.train_batch_size = 1  # Small batch to see quick updates
        self.val_batch_size = 1
        self.workers = 0  # No multiprocessing for debugging
        self.pin_memory = False  # Disable unless needed

        # Model (Smaller BERT to overfit quickly)
        self.num_blocks = 2  # Fewer layers
        self.num_heads = 2
        self.d_model = 128
        self.vocab_size = 30522
        self.d_ff = 512
        self.max_len = 64
        self.pd = 0.0  # No dropout
        self.norm_eps = 1e-12
        self.initialize = True
        self.add_pooler = True

        # Optimizer (AdamW)
        self.l2_weight_decay = 0.0  # Disable weight decay
        self.optim_eps = 1e-6
        self.betas = (0.9, 0.999)
        self.lr = 1e-3  # Higher LR to force quick overfit

        # Training
        self.num_epochs = 1000  # Overfit thoroughly
        self.checkpoint_path = "./overfit_ckpt.pth"
        self.resume = False
        self.save_after_steps = 100000  # Effectively disables saving
        self.warmup_steps = 0  # No warmup for overfitting
        self.gradient_accumulation_steps = 1
        self.overfit = True
