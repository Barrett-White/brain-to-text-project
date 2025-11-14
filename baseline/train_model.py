from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

if __name__ == "__main__":
    args = OmegaConf.load("rnn_args_sampled.yaml")
    trainer = BrainToTextDecoder_Trainer(args)
    metrics = trainer.train()
