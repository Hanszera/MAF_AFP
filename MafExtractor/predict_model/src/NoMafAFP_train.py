import torch
import os
import numpy as np
import argparse, ast
import yaml
from torch.optim.lr_scheduler import CyclicLR
from MafExtractor.predict_model.src.warmupScheduler import warmupLR
from MafExtractor.predict_model.src.Radam import RAdam
from MafExtractor.predict_model.src.lookahead import Lookahead
from tqdm import tqdm
from torch import device
from torch.cuda import is_available
from esm.tokenization import EsmSequenceTokenizer
from MafExtractor.predict_model.utils.loadDataset import XYDataset
from esm.sdk.api import ESMProtein
import pandas as pd
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from MafExtractor.predict_model.utils.checkdir_time import create_numbered_subfolder
import warnings
import json
DEVICE = device("cuda:0" if is_available() else "cpu")

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def configure_optimizers(model, method, learning_rate, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if 'sigma' in fpn:
                no_decay.add(fpn)

            if (pn.endswith('bias') or pn.startswith('bias')):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add('pos_embed')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = None
    if method == 'Adam':
        optimizer = torch.optim.Adam(optim_groups,lr=learning_rate)
    elif method == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate)
    elif method == 'SDG':
        optimizer = torch.optim.SDG(optim_groups,lr=learning_rate)
    elif method == 'Ranger':
        optimizer_inner = RAdam(optim_groups,lr=learning_rate)
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    return optimizer

def configure_scheduler(optimizer, learning_rate, total_epochs,warmup_epochs,steps_per_epoch):
    warmup_epochs=[warmup_epochs]
    init_lr = [learning_rate*1e-2]
    max_lr = [learning_rate]
    final_lr = [learning_rate*1e-2]
    total_epochs=[total_epochs]
    scheduler_ = 'WarmupLR'#
    if scheduler_ == 'CyclicLR':
        if warmup_epochs[0] < 1.0:
            step_size_up = int(steps_per_epoch*warmup_epochs)
            step_size_down = steps_per_epoch - step_size_up
        else:
            step_size_up = int(steps_per_epoch*warmup_epochs)
            step_size_down = step_size_up
        scheduler = CyclicLR(optimizer=optimizer,
                    base_lr=init_lr*2,
                    max_lr=max_lr*2,
                    step_size_up=step_size_up,
                    step_size_down=step_size_down,
                    mode='exp_range',
                    gamma=0.99991,
                    scale_fn=None,
                    scale_mode='cycle',
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1)

    if scheduler_ == 'WarmupLR':
        scheduler = warmupLR(
                    optimizer=optimizer,
                    warmup_epochs=warmup_epochs*2,
                    total_epochs=total_epochs*2,
                    steps_per_epoch=steps_per_epoch,
                    init_lr=init_lr*2,
                    max_lr=max_lr*2,
                    final_lr=final_lr*2)
    if scheduler is None:
        raise RuntimeError(f'{scheduler_} is not a available scheduler')
    return scheduler



def esm_encoder_seq(sequences,max_length) -> torch.Tensor:
    tokenizer = get_esmc_model_tokenizers()
    pad = tokenizer.pad_token_id

    assert pad is not None
    encode_se = stack_variable_length_tensors(
        [
            tokenize_sequence(x, tokenizer, max_length=max_length,add_special_tokens=True)
            for x in sequences.sequence['sequences']
        ],
        constant_value=pad,
    )
    return encode_se
def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    max_length:int,
    add_special_tokens: bool = True

) -> torch.Tensor:
    sequence = sequence.replace('_', sequence_tokenizer.mask_token)
    sequence_tokens = sequence_tokenizer.encode(
        sequence, add_special_tokens=add_special_tokens,padding="max_length", truncation=True, max_length=max_length
    )
    sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
    return sequence_tokens


def get_esm_embedding(seq,max_length) -> _BatchedESMProteinTensor:
    protein_tensor = _BatchedESMProteinTensor(sequence=esm_encoder_seq(seq,max_length).to(DEVICE))

    return protein_tensor


from esm.tokenization import (
    get_esmc_model_tokenizers,
)

def train_mlp(cfg,t_data_path,v_data_path):
    from MafExtractor.predict_model.model.MafAFP.NoMAFModel import EsmcClassifier
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = f'{root_path}/esmc/save_model/mlp/'
    save_path_timestamp = create_numbered_subfolder(save_path)
    data_dic = {"train": t_data_path, "val": v_data_path}
    train_dataset = XYDataset(pd.read_csv(t_data_path))
    train_data_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True, shuffle=True,
                                                    num_workers=1)
    val_dataset = XYDataset(pd.read_csv(v_data_path))
    val_data_loader = DataLoader(val_dataset, batch_size=4, pin_memory=True, shuffle=False,
                                                    num_workers=1)

    model_args = cfg["esm_model"]


    model = EsmcClassifier.from_esm_pretrained(**model_args)

    model.eval()

    print(f"Model has {model.num_params:,} parameters.")


    model.freeze_base()

    model.unfreeze_last_k_encoder_layers(cfg["train"]["esm_train_layers"])

    print(f"Number of trainable parameters: {model.num_trainable_parameters:,}")
    logs = {"dataset":[data_dic],"classfier_params":[{"train_layers ":cfg["train"]["esm_train_layers"]}],"test": []}

    model = model.to(DEVICE)
    loss_function = BCEWithLogitsLoss()

    model.train()
    lr = cfg["train"]["learning_rate"]
    best_AUC = ast.literal_eval(cfg["train"]["best_AUC"])

    optimizer = AdamW(model.parameters(), lr=ast.literal_eval(cfg["train"]["learning_rate"]))


    logger = SummaryWriter(f'{save_path_timestamp}')
    max_padding_length = 100

    for epoch in range(cfg["train"]["starting_epoch"], cfg["train"]["epochs"]):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0
        x_steps = 0
        for index, datas in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch}")):
            text = datas['sequences']
            y = datas['labels'].float().to(DEVICE)
            y = y.unsqueeze(-1)
            protein = ESMProtein(sequence=datas)
            protein_tensor = get_esm_embedding(protein,max_padding_length)

            y_pred = model.forward(sequence_tokens=protein_tensor.sequence,text_tokens=text)

            loss = loss_function(y_pred, y)

            scaled_loss = loss / cfg["train"]["gradient_accumulation_steps"]

            scaled_loss.backward()

            total_cross_entropy += loss.item()
            total_batches += 1
            x_steps += 1

            if x_steps % cfg["train"]["gradient_accumulation_steps"] == 0:

                norm = clip_grad_norm_(model.parameters(), cfg["train"]["max_gradient_norm"])

                optimizer.step()
                # if lr_decay:
                #     scheduler.step()

                # model.zero_grad()
                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = 0.0
        if total_steps > 0:
            average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.5f}",
        )
        from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score,matthews_corrcoef
        all_probs, all_labels = [], []
        if epoch % 1 == 0:
            model.eval()

            for index, datas in enumerate(tqdm(val_data_loader, desc=f"Epoch {epoch}_val", leave=False)):
                text = datas['sequences']
                y = datas['labels'].float().to(DEVICE)
                y = y.unsqueeze(-1)
                protein = ESMProtein(sequence=datas)
                protein_tensor = get_esm_embedding(protein,max_padding_length)

                with torch.no_grad():
                    y_pred = model.forward(sequence_tokens=protein_tensor.sequence,text_tokens=text)

                    y_prob = torch.sigmoid(y_pred)
                all_probs.append(y_prob.cpu())
                all_labels.append(y.cpu())


            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()
            best_thr, best_f1 = 0.5, 0
            for thr in np.linspace(0.05, 0.95, 91):
                preds = (all_probs >= thr).astype(int)
                f1 = f1_score(all_labels, preds)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr

            final_preds = (all_probs >= best_thr).astype(int)
            auc = roc_auc_score(all_labels, all_probs)
            sn = recall_score(all_labels, final_preds)
            precision = precision_score(all_labels, final_preds)
            acc = accuracy_score(all_labels, final_preds)
            f1 = f1_score(all_labels, final_preds)
            mcc = matthews_corrcoef(all_labels, final_preds)

            logger.add_scalar("F1 Score", f1, epoch)
            logger.add_scalar("Precision", precision, epoch)
            logger.add_scalar("AUC", auc, epoch)
            logger.add_scalar("Sn", sn, epoch)
            logger.add_scalar("Acc", acc, epoch)
            logger.add_scalar("MCC", mcc, epoch)

            print(
                f"Epoch {epoch} | "
                f"AUC: {auc:.3f}, "
                f"Sn: {sn:.3f}, "
                f"Acc: {acc:.3f}, "
                f"F1: {f1:.3f}, "
                f"MCC: {mcc:.3f}, " 
                f"Precision: {precision:.3f}, "
                f"BestThr: {best_thr:.2f}"
            )

            logs['test'].append({
                "epoch": epoch,
                "AUC": f'{auc:.3f}',
                "Sn": f'{sn:.3f}',
                "Acc": f'{acc:.3f}',
                "F1": f'{f1:.3f}',
                "MCC": f'{mcc:.3f}',
                "Precision": f'{precision:.3f}',
                "BestThr": f'{best_thr:.2f}'
            })

            if auc > best_AUC:
                best_AUC = auc
                torch.save(model.state_dict(), os.path.join(save_path_timestamp, 'best_AUC.pkl'))
                print(f"Best model saved_{best_AUC:.3f}")
                all_probs = all_probs.ravel()  # [N, 1] -> [N]
                all_labels = all_labels.ravel()  # [N, 1] -> [N]
                np.savez(
                    os.path.join(save_path_timestamp, f"best_AUC_val_epoch.npz"),
                    labels=all_labels,
                    probs=all_probs,
                    auc=np.array([best_AUC], dtype=np.float32),
                )

            model.train()

        save_json(logs, os.path.join(save_path_timestamp, 'logs.json'))
    print("Done!")

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default=f"{root_path}\esmc\config.yaml")
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)


    t_data_type = "mulit_peptide_train_esmc"
    v_data_type = "mulit_peptide_val_esmc"
    t_data_path =f'{root_path}/dataset/iAFPs-Mv-BiTCN/{t_data_type}.csv'
    v_data_path =f'{root_path}/dataset/iAFPs-Mv-BiTCN/{v_data_type}.csv'

    train_mlp(cfg,t_data_path,v_data_path)
