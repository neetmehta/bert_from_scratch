class Config:
    def __init__(self):
        
        # General
        self.max_len = 128
        
        # tokenizer
        self.tokenizer_name = "bert-base-uncased"

        # Data
        self.dataset_path = "F:\\bert_from_scratch\\news_2025-07-25"
        self.mlm_prob = 0.15
        self.mask_prob = 0.8
        self.random_prob = 0.1
        self.same_prob = 0.1
        self.num_proc = 8
        self.train_split = 0.8
        self.val_samples = -1
        self.train_batch_size = 2
        self.val_batch_size = 1
        self.workers = 1
        self.pin_memory = False

        # Model
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

        # Optimizer and loss
        self.l2_weight_decay = 0.01
        self.optim_eps = 1e-9
        self.betas = (0.9, 0.999)
        self.lr = 0.1

        # Training
        self.num_epochs = 3
        self.checkpoint_path = "./best_ckpt_base.pth"
        self.resume = True
        self.save_after_steps = 1000
        self.warmup_steps = 10000
        self.gradient_accumulation_steps = 4
        self.overfit = False