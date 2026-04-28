import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import joyful
from joyful.fusion_methods import AutoFusion


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd, cwd):
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout


def eval_checkpoint(dataset, modalities, data_dir, ckpt_path, device):
    data = joyful.utils.load_pkl(os.path.join(data_dir, dataset, f"data_{dataset}.pkl"))
    ckpt = torch.load(ckpt_path)
    args = ckpt["args"]
    args.device = device
    model = ckpt["modelN_state_dict"].to(device)
    modelF = ckpt["modelF_state_dict"].to(device)
    model.eval()
    modelF.eval()
    testset = joyful.Dataset(data["test"], modelF, False, args)

    golds = []
    preds = []
    embeddings = []
    with torch.no_grad():
        for idx in range(len(testset)):
            batch = testset[idx]
            golds.append(batch["label_tensor"])
            for k, v in batch.items():
                if k != "utterance_texts":
                    batch[k] = v.to(device)
            y_hat = model(batch, False)
            pred = y_hat.detach().cpu()
            rep, _, _ = model.get_rep(batch, False)
            embeddings.append(rep.detach().cpu())
            preds.append(pred)

    golds = torch.cat(golds, dim=-1).numpy()
    preds = torch.cat(preds, dim=-1).numpy()
    embeddings = torch.cat(embeddings, dim=0).numpy()
    return {
        "acc": float(accuracy_score(golds, preds)),
        "wf1": float(f1_score(golds, preds, average="weighted")),
        "golds": golds,
        "preds": preds,
        "embeddings": embeddings,
    }


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(results, title, path):
    labels = [r["name"] for r in results]
    wf1 = [r["mean_wf1"] for r in results]
    acc = [r["mean_acc"] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, wf1, width, label="WF1")
    plt.bar(x + width / 2, acc, width, label="ACC")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_heatmap(grid_rows, alphas, betas, path):
    z = np.zeros((len(alphas), len(betas)))
    for row in grid_rows:
        ai = alphas.index(row["alpha"])
        bi = betas.index(row["beta"])
        z[ai, bi] = row["mean_wf1"]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(z, cmap="YlGn", aspect="auto")
    plt.colorbar(im, label="WF1")
    plt.xticks(np.arange(len(betas)), [str(v) for v in betas])
    plt.yticks(np.arange(len(alphas)), [str(v) for v in alphas])
    plt.xlabel("beta (cl_loss_weight)")
    plt.ylabel("alpha (mf_loss_weight)")
    plt.title("Alpha-Beta Sensitivity (WF1)")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_confusion(golds, preds, label_names, path):
    cm = confusion_matrix(golds, preds)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="YlGn")
    plt.colorbar(im, label="Number of samples")
    plt.xticks(np.arange(len(label_names)), label_names)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_tsne(embeddings, labels, label_names, path):
    tsne = TSNE(n_components=2, random_state=24, init="pca", learning_rate="auto")
    xy = tsne.fit_transform(embeddings)
    plt.figure(figsize=(7, 5))
    for label_id, label_name in enumerate(label_names):
        idx = labels == label_id
        if np.sum(idx) == 0:
            continue
        plt.scatter(xy[idx, 0], xy[idx, 1], s=12, alpha=0.7, label=label_name)
    plt.legend()
    plt.title("Node Representation t-SNE")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def aggregate(items):
    accs = [x["acc"] for x in items]
    wf1s = [x["wf1"] for x in items]
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_wf1": float(np.mean(wf1s)),
        "std_wf1": float(np.std(wf1s)),
    }


def one_setting(setting_name, setting_args, common_args, repeats, run_dir):
    rows = []
    eval_records = []
    for seed in range(1, repeats + 1):
        exp_name = f"{setting_name}_seed{seed}"
        cmd = [
            sys.executable,
            "train.py",
            "--dataset",
            common_args["dataset"],
            "--modalities",
            common_args["modalities"],
            "--epochs",
            str(common_args["epochs"]),
            "--batch_size",
            str(common_args["batch_size"]),
            "--device",
            common_args["device"],
            "--data_dir_path",
            common_args["data_dir"],
            "--seed",
            str(seed),
            "--from_begin",
            "--cl_loss_weight",
            str(setting_args.get("cl_loss_weight", common_args["cl_loss_weight"])),
            "--mf_loss_weight",
            str(setting_args.get("mf_loss_weight", common_args["mf_loss_weight"])),
            "--augment_view1",
            setting_args.get("augment_view1", common_args["augment_view1"]),
            "--augment_view2",
            setting_args.get("augment_view2", common_args["augment_view2"]),
            "--wp",
            str(setting_args.get("wp", common_args["wp"])),
            "--wf",
            str(setting_args.get("wf", common_args["wf"])),
        ]
        if setting_args.get("disable_gcl", False):
            cmd.append("--disable_gcl")
        run_cmd(cmd, cwd=os.path.dirname(__file__))

        ckpt_name = f"{common_args['dataset']}_best_dev_f1_model_{common_args['modalities']}.pt"
        ckpt_src = os.path.join(os.path.dirname(__file__), "model_checkpoints", ckpt_name)
        ckpt_dst = os.path.join(run_dir, f"{exp_name}.pt")
        shutil.copy2(ckpt_src, ckpt_dst)

        metrics = eval_checkpoint(
            dataset=common_args["dataset"],
            modalities=common_args["modalities"],
            data_dir=common_args["data_dir"],
            ckpt_path=ckpt_dst,
            device=common_args["device"],
        )
        row = {
            "setting": setting_name,
            "seed": seed,
            "acc": metrics["acc"],
            "wf1": metrics["wf1"],
        }
        row.update(setting_args)
        rows.append(row)
        eval_records.append(metrics)
    return rows, eval_records


def main():
    parser = argparse.ArgumentParser(description="One-click JOYFUL experiment runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap_4",
        help="Fixed to iemocap_4 in this script.",
    )
    parser.add_argument("--modalities", type=str, default="atv")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./experiment_outputs")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--skip_alpha_beta", action="store_true", default=False)
    parser.add_argument("--skip_window", action="store_true", default=False)
    parser.add_argument("--skip_ablation", action="store_true", default=False)
    parser.add_argument("--skip_augmentation", action="store_true", default=False)
    args = parser.parse_args()

    if args.dataset != "iemocap_4":
        raise ValueError("This script currently supports only dataset=iemocap_4.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_tag if args.run_tag else f"{args.dataset}_{timestamp}"
    run_dir = os.path.abspath(os.path.join(args.out_dir, run_name))
    fig_dir = os.path.join(run_dir, "figures")
    ensure_dir(run_dir)
    ensure_dir(fig_dir)

    common_args = {
        "dataset": args.dataset,
        "modalities": args.modalities,
        "data_dir": args.data_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cl_loss_weight": 0.2,
        "mf_loss_weight": 0.05,
        "augment_view1": "fm+ep",
        "augment_view2": "fm+gp",
        "wp": 8,
        "wf": 8,
    }

    all_rows = []
    plot_sources = {}

    if not args.skip_augmentation:
        aug_settings = {
            "fm+ep": {"augment_view1": "fm", "augment_view2": "ep"},
            "fm+gp": {"augment_view1": "fm", "augment_view2": "gp"},
            "ep+gp": {"augment_view1": "ep", "augment_view2": "gp"},
            "mixed(fm+ep, fm+gp)": {"augment_view1": "fm+ep", "augment_view2": "fm+gp"},
        }
        summary = []
        for name, setting in aug_settings.items():
            rows, eval_records = one_setting(name, setting, common_args, args.repeats, run_dir)
            all_rows.extend(rows)
            agg = aggregate(rows)
            agg["name"] = name
            summary.append(agg)
            plot_sources[f"aug::{name}"] = eval_records
        save_csv(
            os.path.join(run_dir, "augmentation_results.csv"),
            all_rows,
            fieldnames=sorted(set().union(*(r.keys() for r in all_rows))),
        )
        plot_bar(summary, "Augmentation Pair Comparison", os.path.join(fig_dir, "fig4A_augmentation_bar.png"))

    if not args.skip_alpha_beta:
        grid_rows = []
        alpha_values = [0.02, 0.05, 0.08, 0.10]
        beta_values = [0.1, 0.2, 0.3, 0.5]
        for alpha, beta in product(alpha_values, beta_values):
            setting_name = f"alpha{alpha}_beta{beta}"
            setting = {"mf_loss_weight": alpha, "cl_loss_weight": beta}
            rows, eval_records = one_setting(setting_name, setting, common_args, args.repeats, run_dir)
            all_rows.extend(rows)
            agg = aggregate(rows)
            grid_rows.append({"alpha": alpha, "beta": beta, **agg})
            plot_sources[f"ab::{setting_name}"] = eval_records
        save_csv(
            os.path.join(run_dir, "alpha_beta_results.csv"),
            grid_rows,
            fieldnames=["alpha", "beta", "mean_acc", "std_acc", "mean_wf1", "std_wf1"],
        )
        plot_heatmap(grid_rows, alpha_values, beta_values, os.path.join(fig_dir, "fig4B_alpha_beta_heatmap.png"))

    if not args.skip_window:
        window_rows = []
        for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            setting_name = f"window{w}"
            setting = {"wp": w, "wf": w}
            rows, eval_records = one_setting(setting_name, setting, common_args, args.repeats, run_dir)
            all_rows.extend(rows)
            agg = aggregate(rows)
            window_rows.append({"window_size": w, **agg})
            plot_sources[f"w::{setting_name}"] = eval_records
        save_csv(
            os.path.join(run_dir, "window_results.csv"),
            window_rows,
            fieldnames=["window_size", "mean_acc", "std_acc", "mean_wf1", "std_wf1"],
        )
        plt.figure(figsize=(8, 5))
        plt.plot([x["window_size"] for x in window_rows], [x["mean_wf1"] for x in window_rows], marker="o")
        plt.ylim(0, 1)
        plt.xlabel("Window size")
        plt.ylabel("WF1")
        plt.title("Window Size Sensitivity")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig4C_window_curve.png"), dpi=220)
        plt.close()

    if not args.skip_ablation:
        ablation_settings = {
            "Full JOYFUL": {},
            "w/o MF": {"mf_loss_weight": 0.0},
            "w/o GCL": {"disable_gcl": True},
            "single aug (fm,fm)": {"augment_view1": "fm", "augment_view2": "fm"},
        }
        summary = []
        for name, setting in ablation_settings.items():
            rows, eval_records = one_setting(name, setting, common_args, args.repeats, run_dir)
            all_rows.extend(rows)
            agg = aggregate(rows)
            agg["name"] = name
            summary.append(agg)
            plot_sources[f"abl::{name}"] = eval_records
        save_csv(
            os.path.join(run_dir, "ablation_results.csv"),
            summary,
            fieldnames=["name", "mean_acc", "std_acc", "mean_wf1", "std_wf1"],
        )
        plot_bar(summary, "Ablation Study", os.path.join(fig_dir, "ablation_bar.png"))

    best_key = None
    best_wf1 = -1
    for k, records in plot_sources.items():
        avg_wf1 = np.mean([r["wf1"] for r in records]) if records else -1
        if avg_wf1 > best_wf1:
            best_wf1 = avg_wf1
            best_key = k

    if best_key:
        best_record = plot_sources[best_key][0]
        label_dict = {"iemocap_4": ["hap", "sad", "neu", "ang"], "iemocap": ["hap", "sad", "neu", "ang", "exc", "fru"]}
        labels = label_dict[args.dataset]
        plot_confusion(
            best_record["golds"],
            best_record["preds"],
            labels,
            os.path.join(fig_dir, "fig4D_confusion_matrix.png"),
        )
        plot_tsne(
            best_record["embeddings"],
            best_record["golds"],
            labels,
            os.path.join(fig_dir, "tsne_embeddings.png"),
        )

    if all_rows:
        save_csv(
            os.path.join(run_dir, "all_runs_raw.csv"),
            all_rows,
            fieldnames=sorted(set().union(*(r.keys() for r in all_rows))),
        )

    meta = {
        "run_name": run_name,
        "args": vars(args),
        "generated_at": datetime.now().isoformat(),
        "outputs": {
            "run_dir": run_dir,
            "figures": fig_dir,
        },
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()
