class Config:
    def __init__(self):
        # General
        self.max_len = 128  # Reasonable for pretraining with limited GPU memory

        # Tokenizer
        self.tokenizer_name = "bert-base-uncased"

        # Data
        self.dataset_path = "F:\\bert_from_scratch\\news_2025-07-25"
        self.mlm_prob = 0.15  # Masking probability
        self.mask_prob = 0.8
        self.random_prob = 0.1
        self.same_prob = 0.1
        self.num_proc = 2  # Lowered to reduce CPU contention
        self.train_split = 0.9
        self.val_samples = -1  # Use full val set if available
        self.train_batch_size = 8  # Best tested value for 6GB GPU
        self.val_batch_size = 4
        self.workers = 2
        self.pin_memory = True

        # Model (BERT Base)
        self.num_blocks = 12
        self.num_heads = 12
        self.d_model = 768
        self.vocab_size = 30522
        self.d_ff = 3072
        self.max_len = 512
        self.pd = 0.1
        self.norm_eps = 1e-12
        self.initialize = True
        self.add_pooler = True

        # Optimizer (AdamW)
        self.l2_weight_decay = 0.01
        self.optim_eps = 1e-6  # Corrected based on BERT paper
        self.betas = (0.9, 0.999)
        self.lr = 5e-5  # Safe starting point for pretraining

        # Training
        self.num_epochs = 3
        self.checkpoint_path = "./best_ckpt_base.pth"
        self.resume = True
        self.save_after_steps = 1000
        self.warmup_steps = 1000  # For small dataset, reduce warmup
        self.gradient_accumulation_steps = 4  # Simulate larger batch size
        self.overfit = False
