import os
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer


def micro_metric(model, data_iter, device, rel_dict, threshold=0.5):
    correct_num, predict_num, gold_num = 0, 0, 0
    model_path = os.path.join(os.path.dirname(__file__), '..', 'huggingface', model.bert_name)
    if 'albert' in model.bert_name and "chinese" not in model.bert_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path)
    rev_rel_dict = {v: k for k, v in rel_dict.items()}

    loop = tqdm(data_iter, ncols=150)
    for batch in loop:
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, start_logits, end_logits, span_logits, entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            pred_start_logits, pred_end_logits, pred_span_logits = model(input_ids, attention_mask, token_type_ids)

            pred_span_logits = (pred_span_logits + pred_span_logits.transpose(2, 3)) / 2

            pred = (pred_span_logits > threshold)

            pred_entities = []
            ent_matches = torch.where(pred[0] == 1)

            for i in range(len(ent_matches[0])):
                cls = rev_rel_dict[ent_matches[0][i].item()]
                ent_head = ent_matches[1][i].item()
                ent_tail = ent_matches[2][i].item()

                if ent_head > ent_tail:
                    continue
                if model.is_chinese:
                    ent = ''.join(tokenizer.decode(input_ids[0][ent_head: ent_tail + 1]).split())
                else:
                    ent = tokenizer.decode(input_ids[0][ent_head: ent_tail + 1])
                pred_entities.append((ent, cls))

            pred_entities = set(pred_entities)
            gold_entities = set(entities[0])

            correct_num += len(pred_entities & gold_entities)
            predict_num += len(pred_entities)
            gold_num += len(gold_entities)

        loop.set_postfix(correct_num=correct_num, predict_num=predict_num, gold_num=gold_num)

    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1_score


def metric_v1(model, data_iter, device, rel_dict, start_threshold=0.5, end_threshold=0.5, span_threshold=0.5):
    correct_num, predict_num, gold_num = 0, 0, 0
    model_path = os.path.join(os.path.dirname(__file__), '..', 'huggingface', model.bert_name)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    rev_rel_dict = {v: k for k, v in rel_dict.items()}

    loop = tqdm(data_iter)
    for batch in loop:
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, start_logits, end_logits, span_logits, entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            pred_start_logits, pred_end_logits, pred_span_logits = model(input_ids, attention_mask, token_type_ids)
            pred_span_logits = (pred_span_logits + pred_span_logits.transpose(2, 3)) / 2

            seq_len = pred_start_logits.shape[1]
            num_classes = pred_span_logits.shape[1]

            pred_start_logits = pred_start_logits.unsqueeze(-1).unsqueeze(1).repeat(1, num_classes, 1, seq_len)
            pred_end_logits = pred_end_logits.unsqueeze(1).unsqueeze(1).repeat(1, num_classes, seq_len, 1)

            pred = (pred_start_logits > start_threshold) * (pred_end_logits > end_threshold) * (
                        pred_span_logits > span_threshold)

            pred_entities = []
            ent_matches = torch.where(pred[0] == 1)

            for i in range(len(ent_matches[0])):
                cls = rev_rel_dict[ent_matches[0][i].item()]
                ent_head = ent_matches[1][i].item()
                ent_tail = ent_matches[2][i].item()

                if ent_head > ent_tail:
                    continue
                if model.is_chinese:
                    ent = ''.join(tokenizer.decode(input_ids[0][ent_head: ent_tail + 1]).split())
                else:
                    ent = tokenizer.decode(input_ids[0][ent_head: ent_tail + 1])
                pred_entities.append((ent, cls))

            pred_entities = set(pred_entities)
            gold_entities = set(entities[0])

            correct_num += len(pred_entities & gold_entities)
            predict_num += len(pred_entities)
            gold_num += len(gold_entities)

        loop.set_postfix(correct_num=correct_num, predict_num=predict_num, gold_num=gold_num)

    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1_score

def macro_metric(model, data_iter, device, rel_dict, threshold=0.5):

    correct_num, predict_num, gold_num = 0, 0, 0
    model_path = os.path.join(os.path.dirname(__file__), '..', 'huggingface', model.bert_name)
    if 'albert' in model.bert_name and "chinese" not in model.bert_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path)
    rev_rel_dict = {v: k for k, v in rel_dict.items()}

    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    loop = tqdm(data_iter, ncols=150)
    for batch in loop:
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, start_logits, end_logits, span_logits, entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            pred_start_logits, pred_end_logits, pred_span_logits = model(input_ids, attention_mask, token_type_ids)

            pred_span_logits = (pred_span_logits + pred_span_logits.transpose(2, 3)) / 2

            pred = (pred_span_logits > threshold)

            pred_entities = []
            ent_matches = torch.where(pred[0] == 1)

            for i in range(len(ent_matches[0])):
                cls = rev_rel_dict[ent_matches[0][i].item()]
                ent_head = ent_matches[1][i].item()
                ent_tail = ent_matches[2][i].item()

                if ent_head > ent_tail:
                    continue
                if model.is_chinese:
                    ent = ''.join(tokenizer.decode(input_ids[0][ent_head: ent_tail + 1]).split())
                else:
                    ent = tokenizer.decode(input_ids[0][ent_head: ent_tail + 1])
                pred_entities.append((ent, cls))

            pred_entities = set(pred_entities)
            gold_entities = set(entities[0])

            pred_by_class = defaultdict(set)
            gold_by_class = defaultdict(set)
            for ent, cls in pred_entities:
                pred_by_class[cls].add(ent)
            for ent, cls in gold_entities:
                gold_by_class[cls].add(ent)

            all_classes = set(pred_by_class.keys()) | set(gold_by_class.keys())
            for cls in all_classes:
                tp = len(pred_by_class[cls] & gold_by_class[cls])
                fp = len(pred_by_class[cls] - gold_by_class[cls])
                fn = len(gold_by_class[cls] - pred_by_class[cls])
                class_stats[cls]['tp'] += tp
                class_stats[cls]['fp'] += fp
                class_stats[cls]['fn'] += fn

        loop.set_postfix(correct_num=correct_num, predict_num=predict_num, gold_num=gold_num)

    precisions, recalls, f1s = [], [], []
    for cls, stats in class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    print('-'*50)
    print(f'performance scores per class:')
    for precision, recall, f1 in zip(precisions, recalls, f1s):
        print(f'precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1}')
    print('-' * 50)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1_score = sum(f1s) / len(f1s)

    return precision, recall, f1_score

def evaluate_metrics(model, data_iter, device, rel_dict, threshold=0.5, metric_type='micro'):

    if metric_type == 'macro':
        return macro_metric(model, data_iter, device, rel_dict, threshold=threshold)
    elif metric_type == 'micro':
        return micro_metric(model, data_iter, device, rel_dict, threshold=threshold)
    else:
        raise NotImplementedError(f"metric type {metric_type} is not implemented")
