import torch
import os
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch import device
from torch.cuda import is_available
from MafExtractor.predict_model.utils.loadDataset import XYDataset_MAF
from esm.sdk.api import ESMProtein
import pandas as pd
from MafExtractor.predict_model.utils.get_embedding import get_esm_embedding
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from MafExtractor.predict_model.utils.checkdir_time import create_numbered_subfolder
import warnings
from MafExtractor.maf.models.maf_extractor import MafExtractor
import json
from MafExtractor.predict_model.src.optimistic import build_llrd_param_groups
from MafExtractor.predict_model.utils.interpretation import run_interpretation,run_global_interpretation


DEVICE = device("cuda:0" if is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def collate_padding(batch):

    lab_list = []
    maf_X_list = []
    seq_list = []

    for sample in batch:

        sequences = sample['sequences']
        label = sample['labels']
        maf_X = sample['maf_X']
        lab_list.append(label)
        maf_X_list.append(maf_X)
        seq_list.append(sequences)

    maxL = 100
    D = maf_X_list[0].size(1)
    X = torch.zeros(len(maf_X_list), maxL, D)
    for i, x in enumerate(maf_X_list):
        X[i, :x.size(0)] = x

    lab = data.dataloader.default_collate(lab_list)
    maf_Xs = X
    seqs = data.dataloader.default_collate(seq_list)
    return {"sequences": seqs, "labels": lab, "maf_Xs": maf_Xs}


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    save_path = f'{root_path}/MafExtractor/predict_model/save_model/mlp_feature/'
    save_path_timestamp = create_numbered_subfolder(save_path)
    maf_path = f"{root_path}/MafExtractor/maf/output/g_if/best_maf.pt"

    t_data_type = "mulit_peptide_train_esmc"
    v_data_type = "mulit_peptide_val_esmc"
    t_data_path =f'{root_path}/datasets/Antifp_DS1/{t_data_type}.csv'
    v_data_path =f'{root_path}/datasets/Antifp_DS1/{v_data_type}.csv'
    data_dic = {"train": t_data_path, "val": v_data_path}
    train_dataset = XYDataset_MAF(pd.read_csv(t_data_path))
    train_data_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True, shuffle=True,
                                                    persistent_workers=True, num_workers=1,collate_fn=collate_padding)
    val_dataset = XYDataset_MAF(pd.read_csv(v_data_path))
    val_data_loader = DataLoader(val_dataset, batch_size=4, pin_memory=True, shuffle=False,
                                                    persistent_workers=True, num_workers=1,collate_fn=collate_padding)


    from MafExtractor.predict_model.model.MafAFP.MafAFPmodel import MafAFPClassifier,clasfier_params
    model_args = {
        "model_name": "esmc_600m",
        "classifier_hidden_ratio": 1,
        "id2label": {"pos": 1, "neg": 0},
        "use_flash_attention": True,
    }


    maf = MafExtractor()
    maf.load_state_dict(torch.load(maf_path, map_location='cuda'))
    maf.eval()
    for p in maf.parameters():
        p.requires_grad = False
    model_name = "esmc_600m"
    model = MafAFPClassifier.from_esm_pretrained(**model_args)
    model.eval()
    print(f"Model has {model.num_params:,} parameters.")
    train_layer = 1
    model.freeze_base()

    model.unfreeze_last_k_encoder_layers(train_layer)

    print(f"Number of trainable parameters: {model.num_trainable_parameters:,}")
    logs = {"dataset":[data_dic],"train_layer":train_layer,"classfier_params":[],"test": []}
    logs['classfier_params'].append(
        clasfier_params)
    precision_metric = BinaryPrecision().to(DEVICE)
    recall_metric = BinaryRecall().to(DEVICE)
    model = model.to(DEVICE)
    labels = train_dataset.dataFrame["label"].values
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    pos_weight = torch.tensor([neg / pos], device=DEVICE, dtype=torch.float32)
    loss_function = BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    gradient_accumulation_steps = 16
    max_gradient_norm=1.0
    starting_epoch=0
    num_epochs=10

    base_lr = 5e-5
    head_lr_mult = 5.0
    weight_decay = 1e-4
    lr_decay = 0.8

    param_groups = build_llrd_param_groups(
        model,
        base_lr=base_lr,
        head_lr_mult=head_lr_mult,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
    )
    optimizer = AdamW(param_groups)
    best_AUC = 1e-9
    logger = SummaryWriter(f'{save_path_timestamp}')
    max_padding_length = 100
    checkpoint_interval = 100
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(starting_epoch, num_epochs):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0
        x_steps = 0
        for index, datas in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch}")):
            text = datas['sequences']
            with torch.no_grad():  # 修改点
                maf_X = datas['maf_Xs']
                out = maf(maf_X, text)
                maf_feature = out["global_feat"].to(DEVICE)
            y = datas['labels'].float().to(DEVICE)
            y = y.unsqueeze(-1)
            protein = ESMProtein(sequence=datas)
            protein_tensor = get_esm_embedding(protein,max_padding_length)
            with torch.amp.autocast('cuda'):
                out = model.forward_backbone_only(sequence_tokens=protein_tensor.sequence)

            with torch.amp.autocast('cuda',enabled=False):
                y_pred = model.forward_classifier_only(out, maf_feature)
                loss = loss_function(y_pred, y)


            loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            total_cross_entropy += loss.item()
            total_batches += 1
            x_steps += 1

            if x_steps % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                norm = clip_grad_norm_(model.parameters(), max_gradient_norm)

                scaler.step(optimizer)
                scaler.update()
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
                with torch.no_grad():
                    maf_X = datas['maf_Xs']
                    out = maf(maf_X, text)
                    maf_feature = out["global_feat"].to(DEVICE)
                    protein = ESMProtein(sequence=datas)
                    protein_tensor = get_esm_embedding(protein,max_padding_length)
                    # y_pred = model.forward(sequence_tokens=protein_tensor.sequence,maf=maf_feature)
                    with torch.amp.autocast('cuda'):
                        out = model.forward_backbone_only(sequence_tokens=protein_tensor.sequence)
                    with torch.amp.autocast('cuda',enabled=False):
                        y_pred = model.forward_classifier_only(out, maf_feature)
                        y_prob = torch.sigmoid(y_pred)
                    all_probs.append(y_prob.cpu())
                    all_labels.append(y.cpu())
                # if index == 0:
                #     seq_batch = protein_tensor.sequence[:1]
                #     maf_batch = maf_feature[:1]
                #     run_interpretation(
                #         model=model,
                #         seq_tensor=seq_batch,
                #         maf_tensor=maf_batch,
                #         seq_labels=[f"AA{i}" for i in range(seq_batch.size(1))],
                #         maf_features=["hydropathy", "charge", "mass", "polarity", "size", "aromatic"],
                #         output_dir=save_path_timestamp,
                #         epoch=epoch
                #     )

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
            precision_metric.reset()
            recall_metric.reset()


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

        if epoch == num_epochs-1:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # torch.save(checkpoint, f"{save_path_timestamp}/{model_name}_epoch_{epoch}_{average_cross_entropy:.2f}.pkl" )

            print("Checkpoint saved")

            run_global_interpretation(
                model=model,
                maf_model = maf,
                val_loader=val_data_loader,
                maf_features=["hydropathy", "charge", "mass", "polarity", "size", "aromatic"],
                device=DEVICE,
                output_dir=f"{save_path_timestamp}"
            )

        save_json(logs, os.path.join(save_path_timestamp, 'logs.json'))

    print("Done!")