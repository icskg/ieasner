import os
import sys

import time
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.albert_industry_dataset import ALBertIndustryDataset
from data.collate_functions import bert_industry_collate
from models.albert_industry import ALBertIndustry
from evaluate.bert_industry_evaluate import evaluate_metrics
from utils.common import save_json, mask_focal_loss, get_gpu_memory_occupation, get_best_score
from utils.logs import Logger
from scripts.dataset_ratio_traindata import sample_train_data
from utils.early_stopping import EarlyStopping


bert_dim_settings = {
    "albert-base-v2": 768,
    "albert-large-v2": 1024,
    "albert-xlarge-v2": 2048,
    "albert-xxlarge-v2": 4096
}


class ALBertIndustryTrainer:
    def __init__(self):
        super(ALBertIndustryTrainer, self).__init__()
        self.args = self.parse_args()
        self.train_iter, self.test_iter, self.dev_iter, self.rel_dict = self.load_data_iterators(self.args.dataset,
                                                                                                 self.args.bert_name,
                                                                                                 self.args.max_len,
                                                                                                 self.args.batch_size,
                                                                                                 self.args.ratio)
        self.device = self.choose_device()
        self.args.bert_dim = bert_dim_settings[self.args.bert_name]
        self.model = self.init_model(self.args.bert_name, self.args.bert_dim, self.args.num_hiddens, len(self.rel_dict),
                                     self.args.load_weights, self.args.dataset + '_ALBertIndustry.pth', self.args.gamma,
                                     self.args.delta)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.args.lr_decay)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.args.lr_decay, patience=10)
        self.early_stopping = EarlyStopping(patience=15, verbose=False, dataset_name=self.args.dataset, delta=1e-6)
        self.metrics = []
        self.warnup_epochs = 20

    def train(self):
        best_f1_score = 0.0

        self.model.to(self.device)

        Logger.info(
            f'model hyper-parameters, alpha:{self.args.alpha}, beta:{self.args.beta}, gamma:{self.args.gamma}, delta:{self.args.delta}')

        for epoch in range(self.args.num_epochs):
            self.model.train()

            start_clock = time.time()
            self.train_epoch(self.model, self.optimizer, self.train_iter, self.device)
            end_clock = time.time()
            train_epoch_times = end_clock - start_clock

            self.model.eval()
            dev_precision, dev_recall, dev_f1_score = evaluate_metrics(self.model, self.dev_iter, self.device, self.rel_dict, metric_type=self.args.metric)
            Logger.info(
                f'Epoch {epoch + 1}, lr {self.optimizer.param_groups[0]["lr"]} ,dev {self.args.metric}-f1 score {dev_f1_score * 100:4.2f}%, dev precision {dev_precision * 100:4.2f}%, dev recall {dev_recall * 100:4.2f}%')

            start_clock = time.time()
            test_precision, test_recall, test_f1_score = evaluate_metrics(self.model, self.test_iter, self.device, self.rel_dict)
            end_clock = time.time()
            test_epoch_times = end_clock - start_clock

            print("train time cost:{}, test time cost:{}".format(train_epoch_times, test_epoch_times))

            if test_f1_score > best_f1_score:
                best_f1_score = test_f1_score
                if self.args.save_weights:
                    self.save_weights(self.model, self.args.dataset + '_albert_industry.pth', self.device)
                    Logger.info(
                        f'saving model, best {self.args.metric}-f1 score {test_f1_score * 100:4.2f}%, best precision {test_precision * 100:4.2f}%, best recall {test_recall * 100:4.2f}%')


            Logger.info(
                f'Epoch {epoch + 1}, test {self.args.metric}-f1 score {test_f1_score * 100:4.2f}%, test precision {test_precision * 100:4.2f}%, test recall {test_recall * 100:4.2f}%')

            self.scheduler.step(test_f1_score)

            self.metrics.append(
                {'epoch': epoch + 1,
                 'f1': round(test_f1_score, 5),
                 'precision': round(test_precision, 5),
                 'recall': round(test_recall, 5),
                 'train_epoch_times': train_epoch_times,
                 'test_epoch_times': test_epoch_times,
                 "type":self.args.metric,
                 "ratio":self.args.ratio
                 }
            )
            if epoch > self.warnup_epochs:
                self.early_stopping(test_f1_score, self.model, self.optimizer, path=None)
                if self.early_stopping.early_stop:
                    Logger.info("Early stopping")
                    break

        result_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', self.args.dataset + '_albert_metrics_{}_{}.json'.format(self.args.num_epochs, time.time()))
        save_json(self.metrics, result_path)

        best_score = get_best_score(result_path, 'f1')
        Logger.info("best score:".format(best_score))

    @staticmethod
    def save_weights(model: torch.nn.Module, weights_name: str, device: torch.device):
        model.to('cpu')
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', weights_name)
        torch.save(model.state_dict(), weights_path)
        model.to(device)

    def train_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_iter: DataLoader,
                    device: torch.device):
        epoch_loss, num_batches = 0.0, 0
        loop = tqdm(train_iter, ncols=100)
        for batch in loop:
            # torch.cuda.empty_cache()
            input_ids, attention_mask, token_type_ids, start_logits, end_logits, span_logits, entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            start_logits = start_logits.to(device)
            end_logits = end_logits.to(device)
            span_logits = span_logits.to(device)
            num_classes = span_logits.shape[1]

            pred_start_logits, pred_end_logits, pred_span_logits = model(input_ids, attention_mask, token_type_ids)

            start_loss = mask_focal_loss(pred_start_logits, start_logits, attention_mask)
            end_loss = mask_focal_loss(pred_end_logits, end_logits, attention_mask)
            span_loss = mask_focal_loss(pred_span_logits, span_logits,
                                        attention_mask.unsqueeze(1).repeat(1, num_classes, 1))

            loss = self.args.alpha * (start_loss + end_loss) + self.args.beta * span_loss

            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            num_batches += 1
            loop.set_postfix(train_loss=epoch_loss / num_batches)

    def init_model(self, bert_name: str, bert_dim: int, num_hiddens: int, num_classes: int, load_weights: bool,
                   weights_name: str, gamma: float = 1.0, delta: float = 1.0):
        model = ALBertIndustry(bert_name, bert_dim, num_hiddens, num_classes, gamma, delta)
        if load_weights:
            weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', weights_name)
            model.load_state_dict(torch.load(weights_path))
            Logger.info(f'finish loading weights from {weights_path}')

        Logger.info(
            f'finish initiating ALBertIndustry, parameters:{self.args}')
        return model

    def choose_device(self):
        if self.args.device.isdigit() and torch.cuda.is_available():
            device_id = int(self.args.device) if torch.cuda.device_count() > int(self.args.device) else 0
            device = f'cuda:{device_id}'
        else:
            device = 'cpu'
        Logger.info(f'finish choosing device, device: {device}')
        return torch.device(device)

    @staticmethod
    def load_data_iterators(dataset: str, bert_name: str, max_len: int, batch_size: int, ratio:float=1.0):
        sample_train_data(ratio, dataset=dataset)
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', dataset)
        rel_path = os.path.join(dataset_path, 'rel.json')
        train_path = os.path.join(dataset_path, f'train_{ratio}.json')
        test_path = os.path.join(dataset_path, 'test.json')
        dev_path = os.path.join(dataset_path, 'dev.json')
        train_dataset = ALBertIndustryDataset(train_path, bert_name, rel_path, max_len, is_test=False)
        test_dataset = ALBertIndustryDataset(test_path, bert_name, rel_path, max_len, is_test=True)
        dev_dataset = ALBertIndustryDataset(dev_path, bert_name, rel_path, max_len, is_test=True)

        train_iter = DataLoader(dataset=train_dataset, collate_fn=bert_industry_collate, batch_size=batch_size,
                                shuffle=True)
        test_iter = DataLoader(dataset=test_dataset, collate_fn=bert_industry_collate, batch_size=1, shuffle=False)
        dev_iter = DataLoader(dataset=dev_dataset, collate_fn=bert_industry_collate, batch_size=1, shuffle=False)

        Logger.info(
            f'finish loading data iterators, train_samples: {len(train_dataset)}, test_samples: {len(test_dataset)}, dev_samples: {len(dev_dataset)}')
        return train_iter, test_iter, dev_iter, train_dataset.rel_dict

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Model Controller')
        parser.add_argument('--dataset', type=str, default='ACE2004')
        parser.add_argument('--bert_name', type=str, default='albert-base-v2')
        parser.add_argument('--max_len', type=int, default=180)
        parser.add_argument('--bert_dim', type=int, default=768)
        parser.add_argument('--num_hiddens', type=int, default=128)
        parser.add_argument('--device', type=str, default='1')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=2e-5)
        parser.add_argument('--save_weights', type=bool, default=True)
        parser.add_argument('--load_weights', type=bool, default=False)
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--lr_decay', type=float, default=0.85)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=0.3)
        parser.add_argument('--ratio', type=float, default=0.1)
        parser.add_argument('--metric', type=str, default='micro')
        return parser.parse_args()


if __name__ == '__main__':
    max_rounds = 20
    for index in range(max_rounds):
        Logger.info(
            f'Start the {index}/{max_rounds} round of random initialization tests for ALBertIndustry model')
        trainer = ALBertIndustryTrainer()
        trainer.train()

