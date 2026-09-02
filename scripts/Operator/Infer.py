import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .ResultBuilder import ResultBuilder


class ModelInfer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.get_config("General", "device")
        self.quant_topk = int(args.get_config("Infer", "quant_topk", default=6))
        model_path = args.get_config("Infer", "checkpoint_path")
        if os.path.exists(model_path) and model_path.endswith(".pt"):
            self.model.load(model_path)
            if torch.cuda.device_count() > 1:
                logging.info("Found %d GPUs, use DataParallel", torch.cuda.device_count())
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            logging.info("Loaded model from %s", model_path)
        else:
            raise ValueError(f"Invalid model path: {model_path}")

    def infer(self, data_loader, xic_datas, basename):
        results_df = self.infer_model(data_loader)
        xic_data_map = {}
        for xic_data in xic_datas:
            key = self.format_modified_sequence(
                xic_data["pre"]["peptide"], xic_data["pre"]["modification"]
            ) + "_" + str(xic_data["pre"]["charge"])
            xic_data_map[key] = xic_data

        quantities = []
        for _, row in results_df.iterrows():
            key = row["modified_peptide"] + "_" + str(int(row["charge"]))
            xic_data = xic_data_map.get(key)
            if xic_data is None:
                logging.warning("Precursor key %s not found; quantity is NaN", key)
                quantities.append(np.nan)
            else:
                quantities.append(self.quant_precursor(xic_data, self.quant_topk))
        results_df["quant_result"] = quantities
        return ResultBuilder(self.args).build(results_df, basename)

    def infer_model(self, data_loader):
        self.model.eval()
        result = {
            "label": [], "score": [], "feature_distance": [],
            "cos_similarity": [], "modified_peptide": [], "charge": [],
        }
        with torch.inference_mode():
            for data in tqdm(data_loader, desc="Inference", total=len(data_loader)):
                output = self.model(data)
                result["label"].extend(data["label"].cpu().tolist())
                result["score"].extend(output.get(
                    "score", torch.full_like(data["label"], -1)
                ).cpu().tolist())
                result["feature_distance"].extend((
                    -torch.norm(output["peptide_feature"] - output["spec_feature"], dim=-1)
                ).cpu().tolist())
                result["cos_similarity"].extend(F.cosine_similarity(
                    output["peptide_feature"], output["spec_feature"], dim=-1
                ).cpu().tolist())
                result["modified_peptide"].extend(data["modified_peptide"])
                result["charge"].extend(data["charge"].cpu().tolist())
        return pd.DataFrame(result)

    @staticmethod
    def format_modified_sequence(sequence, modifications):
        return "".join(
            amino_acid + (f"({modifications[index]})" if index in modifications else "")
            for index, amino_acid in enumerate(sequence)
        )

    @staticmethod
    def quant_precursor(data, topk=6):
        chrom = np.asarray(data["frag"]["chrom"], dtype=float)
        rt = np.asarray(data["frag"]["RT"], dtype=float)
        if chrom.ndim != 2 or rt.ndim != 1 or chrom.shape[1] != len(rt):
            raise ValueError("Invalid fragment chromatogram/RT shape")
        if len(rt) < 2:
            return 0.0
        areas = np.trapezoid(chrom, x=rt, axis=1)
        nonzero = areas[areas > 0]
        if nonzero.size == 0:
            return 0.0
        return float(np.mean(np.sort(nonzero)[-min(topk, nonzero.size):]))
