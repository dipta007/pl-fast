import torch
from torch import nn
import lightning as pl
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


THRESHOLD = 0.5


class DummyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, text):
        return self.model(**text).logits

    def training_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        loss = self.criterion(y_hat.view(-1), label.view(-1))
        log_dict = self.get_metrics(label.view(-1), torch.sigmoid(y_hat.view(-1)), "train")
        log_dict["train/loss"] = loss
        self.log_dict(log_dict, prog_bar=True, batch_size=self.config.batch_size, sync_dist=self.config.ddp)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        loss = self.criterion(y_hat.view(-1), label.view(-1))
        log_dict = self.get_metrics(label.view(-1), torch.sigmoid(y_hat.view(-1)), "valid")
        log_dict["valid/loss"] = loss
        self.log_dict(log_dict, prog_bar=True, batch_size=self.config.batch_size, sync_dist=self.config.ddp)
        return loss

    def predict_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        return torch.sigmoid(y_hat.view(-1)), label.view(-1)

    def get_metrics(self, y, y_hat, mode):
        y_hat = y_hat > THRESHOLD
        acc = accuracy_score(y.cpu(), y_hat.cpu())

        log_dict = {}
        if mode in ["train", "valid"]:
            log_dict[f"{mode}/acc"] = acc
        return log_dict

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)


if __name__ == "__main__":
    from src.base_config import get_config
    config = get_config()
    input_names = ['Sentence']
    output_names = ['yhat']
    model = DummyModel(config=config)
    dummy_text = {
        "text": {
            'input_ids': torch.randint(0, 100, (2, 128)),
            'attention_mask': torch.randint(0, 2, (2, 128))
        }
    }
    torch.onnx.export(model, dummy_text, 'rnn.onnx', input_names=input_names, output_names=output_names)