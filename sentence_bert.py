import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from tokenizer import BertTokenizer
from pooling import Pooling
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    load_multitask_data_nli
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask, model_eval_multitask_nli

from torch.utils.tensorboard import SummaryWriter

TQDM_DISABLE=False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskSentenceBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        self.cos = torch.nn.CosineSimilarity(dim=-1)

        self.pooling = Pooling(config.hidden_size, config.pooling_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.project_sentiment = nn.Linear(config.hidden_size, config.num_labels)
        self.project_para = nn.Linear(config.hidden_size, 1)            # original BERT classification
        cat_len = 4 if args.four_cat else 3
        self.project_para_s = nn.Linear(cat_len * config.hidden_size, 1)      # SBERT classification
        self.project_sts = None
        self.project_inf = nn.Linear(3 * config.hidden_size, 3)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)
    
    def predict_sentiment(self, input_ids, attention_mask):
        output = self.forward(input_ids, attention_mask)
        output = output["pooler_output"]
        output = self.dropout(output)
        output = self.project_sentiment(output)
        return output
    
    # used for original BERT para and STS tasks
    def get_similarity_embeddings(self,
                                  input_ids_1, attention_mask_1,
                                  input_ids_2, attention_mask_2):
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.size()[0], 1)

        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)
        return self.forward(input_id, attention_mask)
    
    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        if args.use_bert_para:
            # original BERT paraphrase classification
            output = self.get_similarity_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            output = output["pooler_output"]
            output = self.dropout(output)
            output = self.project_para(output)
        else:
            # SBERT paraphrase classification
            u, v = self.get_sentence_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            diff = torch.abs(u - v)
            if args.four_cat:
                prod = u * v
                output = torch.cat((u, v, diff, prod), dim=1)
            else:
                output = torch.cat((u, v, diff), dim=1)
            # output = self.dropout(output)
            output = self.project_para_s(output)
        return output
    
    def get_sentence_embeddings(self,
                                input_ids_1, attention_mask_1,
                                input_ids_2, attention_mask_2):
        u = self.forward(input_ids_1, attention_mask_1)
        u = self.pooling(u, attention_mask_1)["sentence_embedding"]
        v = self.forward(input_ids_2, attention_mask_2)
        v = self.pooling(v, attention_mask_2)["sentence_embedding"]
        return (u, v)
    
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        u, v = self.get_sentence_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        return self.cos(u, v)
    
    def predict_inference(self,
                          input_ids_1, attention_mask_1,
                          input_ids_2, attention_mask_2):
        u, v = self.get_sentence_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        diff = torch.abs(u - v)
        output = torch.cat((u, v, diff), dim=1)
        output = self.project_inf(output)
        return output


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train SentenceBERT.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    if args.should_train_nli or args.debug_nli:
        sst_train_data, num_labels, para_train_data, sts_train_data, nli_train_data = load_multitask_data_nli(args.sst_train, args.para_train, args.sts_train, args.nli_train, split ='train')
        sst_dev_data, num_labels, para_dev_data, sts_dev_data, nli_dev_data = load_multitask_data_nli(args.sst_dev, args.para_dev, args.sts_dev, args.nli_dev, split ='dev')
    else:
        sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split ='train')
        sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split ='dev')

    # Sentiment classification: SST dataset
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # Paraphrase detection: Quora dataset
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # Semantic textual similarity (STS): SemEval dataset
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    # Natural Language Inference (NLI): SNLI dataset
    if args.should_train_nli or args.debug_nli:
        nli_train_data = SentencePairDataset(nli_train_data, args)
        nli_dev_data = SentencePairDataset(nli_dev_data, args)

        nli_train_dataloader = DataLoader(nli_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=nli_train_data.collate_fn)
        nli_dev_dataloader = DataLoader(nli_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=nli_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(num_labels),
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'pooling_mode': args.pooling_mode}

    config = SimpleNamespace(**config)

    model = MultitaskSentenceBERT(config)
    if args.load_model:
        assert args.existing_model_path is not None
        saved = torch.load(args.existing_model_path)
        model.load_state_dict(saved['model'])

        print(f"Loaded model to continue training on {args.existing_model_path}")
    
    model = model.to(device)

    tensorboard_path = "sbert" if args.use_gpu else "sbert_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    sst_iter = 0
    para_iter = 0
    sts_iter = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss_sst = 0
        num_batches_sst = 0
        train_loss_para = 0
        num_batches_para = 0
        train_loss_sts = 0
        num_batches_sts = 0
        train_loss_nli = 0
        num_batches_nli = 0

        # train on SST data
        if args.should_train_sst:
            for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss_sst += loss.item()
                num_batches_sst += 1

                sst_iter += 1

                # if sst_iter == 0:
                writer.add_scalar('loss-sst/train', train_loss_sst / num_batches_sst, sst_iter)

            train_loss_sst = train_loss_sst / num_batches_sst

        # train on paraphrase data
        if args.should_train_para:
            for batch in tqdm(para_train_dataloader, desc=f'train-para-{epoch}', disable=TQDM_DISABLE):
                if np.random.randint(1, 100) < args.batch_skip:
                    continue
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                                batch['attention_mask_1'],
                                                                batch['token_ids_2'],
                                                                batch['attention_mask_2'],
                                                                batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                b_labels = b_labels.float()

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels, reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss_para += loss.item()
                num_batches_para += 1

                para_iter += 1
                
                if para_iter % args.log_every == 0:
                    writer.add_scalar('loss-para/train', train_loss_para / num_batches_para, para_iter)

            print(f'Number of para batches: {num_batches_para}')
            train_loss_para = train_loss_para / num_batches_para

        # train on STS data
        if args.should_train_sts:
            for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                                batch['attention_mask_1'],
                                                                batch['token_ids_2'],
                                                                batch['attention_mask_2'],
                                                                batch['labels'])
                
                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                # scale labels [0.0, 5.0] -> [0.0, 1.0]
                b_labels = b_labels.float() / 5.0

                optimizer.zero_grad()
                cos_similarity = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                # loss = F.cosine_embedding_loss(u, v, b_labels, reduction='sum') / args.batch_size
                loss = F.mse_loss(cos_similarity, b_labels, reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss_sts += loss.item()
                num_batches_sts += 1

                # if sts_iter == 0:
                writer.add_scalar('loss-sts/train', train_loss_sts / num_batches_sts, sts_iter)

            train_loss_sts = train_loss_sts / num_batches_sts

        # train on NLI data
        if args.should_train_nli:
            for batch in tqdm(nli_train_dataloader, desc=f'train-nli-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                                  batch['attention_mask_1'],
                                                                  batch['token_ids_2'],
                                                                  batch['attention_mask_2'],
                                                                  batch['labels'])
                
                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                # b_labels = b_labels.float()

                optimizer.zero_grad()
                logits = model.predict_inference(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss_nli += loss.item()
                num_batches_nli += 1

            train_loss_nli = train_loss_nli / num_batches_nli

        if args.should_train_nli or args.debug_nli:
            train_sst_acc, _, _, train_para_acc, _, _, train_sts_corr, _, _, train_nli_acc, _, _ = model_eval_multitask_nli(sst_train_dataloader,
                                                                                                                            para_train_dataloader,
                                                                                                                            sts_train_dataloader,
                                                                                                                            nli_train_dataloader,
                                                                                                                            model,
                                                                                                                            device)
            dev_sst_acc, _, _, dev_para_acc, _, _, dev_sts_corr, _, _, dev_nli_acc, _, _ = model_eval_multitask_nli(sst_dev_dataloader,
                                                                                                                    para_dev_dataloader,
                                                                                                                    sts_dev_dataloader,
                                                                                                                    nli_dev_dataloader,
                                                                                                                    model,
                                                                                                                    device)
        else:
            train_sst_acc, _, _, train_para_acc, _, _, train_sts_corr, _, _ = model_eval_multitask(sst_train_dataloader,
                                                                                                para_train_dataloader,
                                                                                                sts_train_dataloader,
                                                                                                model,
                                                                                                device)
            dev_sst_acc, _, _, dev_para_acc, _, _, dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader,
                                                                                            para_dev_dataloader,
                                                                                            sts_dev_dataloader,
                                                                                            model,
                                                                                            device)

        # select model based on leaderboard overall performance
        overall_score = (dev_sst_acc + (dev_sts_corr + 1) / 2 + dev_para_acc) / 3
        if args.score == "overall" and overall_score > best_dev_acc:
            best_dev_acc = overall_score
            save_model(model, optimizer, args, config, args.filepath)
        elif args.score == "sst" and dev_sst_acc > best_dev_acc:
            best_dev_acc = dev_sst_acc
            save_model(model, optimizer, args, config, args.filepath)
        elif args.score == "para" and dev_para_acc > best_dev_acc:
            best_dev_acc = dev_para_acc
            save_model(model, optimizer, args, config, args.filepath)
        elif args.score == "sts" and dev_sts_corr > best_dev_acc:
            best_dev_acc = dev_sts_corr
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss sst :: {train_loss_sst :.3f}, train acc sst :: {train_sst_acc :.3f}, dev acc sst :: {dev_sst_acc :.3f}")
        print(f"Epoch {epoch}: train loss para :: {train_loss_para :.3f}, train acc para :: {train_para_acc :.3f}, dev acc para :: {dev_para_acc :.3f}")
        print(f"Epoch {epoch}: train loss sts :: {train_loss_sts :.3f}, train corr sts:: {train_sts_corr :.3f}, dev corr sts :: {dev_sts_corr :.3f}")
        if args.should_train_nli or args.debug_nli:
            print(f"Epoch {epoch}: train loss nli :: {train_loss_nli :.3f}, train acc nli :: {train_nli_acc :.3f}, dev acc nli :: {dev_nli_acc :.3f}")
        if args.score == "overall": print(f"Epoch {epoch}: dev overall score :: {overall_score :.3f}")

        writer.add_scalars('acc/train',
                           {'sst':train_sst_acc,
                            'para':train_para_acc},
                           epoch+1)
        writer.add_scalar('sts/train', train_sts_corr, epoch+1)

        writer.add_scalars('acc/dev',
                           {'sst':dev_sst_acc,
                            'para':dev_para_acc},
                           epoch+1)
        writer.add_scalar('sts/dev', dev_sts_corr, epoch+1)
        writer.add_scalar('overall/dev', overall_score, epoch+1)


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskSentenceBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # baselines
    parser.add_argument("--use_bert_para", action="store_true")
    parser.add_argument("--pooling_mode", type=str, default="mean")
    parser.add_argument("--should_train_sst", action="store_true")
    parser.add_argument("--should_train_para", action="store_true")
    parser.add_argument("--should_train_sts", action="store_true")
    parser.add_argument("--should_train_nli", action="store_true")

    # evaluation
    parser.add_argument("--score", type=str, default="overall", choices=("overall", "sst", "para", "sts"))

    # nli dataset
    parser.add_argument("--nli_train", type=str, default="data/snli-train.csv")
    parser.add_argument("--nli_dev", type=str, default="data/snli-dev.csv")
    parser.add_argument("--nli_test", type=str, default="data/snli-test.csv")

    parser.add_argument("--nli_dev_out", type=str, default="predictions/nli-dev-output.csv")
    parser.add_argument("--nli_test_out", type=str, default="predictions/nli-test-output.csv")

    # debug
    parser.add_argument("--debug_nli", action="store_true")

    # logging
    parser.add_argument("--log_every", type=int, default=10)

    # continue training model
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--existing_model_path", type=str)

    # training batch skip for large datasets
    parser.add_argument("--batch_skip", type=int, default=0)

    # different paraphrase concatenation method
    parser.add_argument("--four_cat", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)

