import logging
import os
import re

import pandas as pd

from .FDR import FDRUtils


LOGGER = logging.getLogger(__name__)


class ResultBuilder:
    MODIFICATION_TO_UNIMOD = {
        "carbamidomethyl": "UniMod:4",
        "phosphorylation": "UniMod:21",
        "oxidation": "UniMod:35",
        "pyroglu": "UniMod:27",
        "pyrocarbamidomethyl": "UniMod:26",
        "deamidation": "UniMod:7",
        "acetyl": "UniMod:1",
    }

    METADATA_COLUMNS = [
        "Run.Index", "Run", "Precursor.Id", "Modified.Sequence",
        "Stripped.Sequence", "Precursor.Charge", "Decoy", "Proteotypic",
        "Precursor.Mz", "Protein.Ids", "Protein.Group", "Protein.Names",
        "Genes", "RT", "iRT", "Predicted.RT", "Predicted.iRT", "IM",
        "iIM", "Predicted.IM", "Predicted.iIM", "RT.Start", "RT.Stop",
        "FWHM",
    ]
    DIACLIP_COLUMNS = [
        "DIAClip.Score", "DIAClip.Q.Value", "DIAClip.Feature.Distance",
        "DIAClip.Cosine.Similarity", "DIAClip.Quantity", "DIAClip.Passed",
    ]

    def __init__(self, args):
        self.out_dir = str(args.get_config("General", "out_dir"))
        self.report_path = os.path.join(self.out_dir, "diann", "all_report.parquet")
        self.fdr = float(args.get_config("Infer", "fdr", default=0.01))

    @classmethod
    def to_unimod_sequence(cls, sequence):
        sequence = str(sequence)

        def replace(match):
            name = match.group(1).strip()
            if name.lower().startswith("unimod:"):
                return f"({name})"
            unimod = cls.MODIFICATION_TO_UNIMOD.get(name.lower())
            if unimod is None:
                raise ValueError(f"Unsupported modification in DIA-CLIP output: {name}")
            return f"({unimod})"

        return re.sub(r"\(([^()]*)\)", replace, sequence)

    @staticmethod
    def stripped_sequence(modified_sequence):
        return re.sub(r"\([^()]*\)", "", str(modified_sequence))

    def prepare_scores(self, results_df):
        required = {
            "label", "score", "feature_distance", "cos_similarity",
            "modified_peptide", "charge", "quant_result",
        }
        missing = sorted(required.difference(results_df.columns))
        if missing:
            raise ValueError(f"Missing inference result columns: {missing}")

        result = results_df.copy()
        result["Modified.Sequence"] = result["modified_peptide"].map(self.to_unimod_sequence)
        result["Stripped.Sequence"] = result["Modified.Sequence"].map(self.stripped_sequence)
        result["Precursor.Charge"] = result["charge"].astype(int)
        result["Precursor.Id"] = result["Modified.Sequence"] + result["Precursor.Charge"].astype(str)
        result["DIAClip.Score"] = result["score"].astype(float)
        result["DIAClip.Q.Value"] = FDRUtils.calculate_q_values(result["score"], result["label"])
        result["DIAClip.Feature.Distance"] = result["feature_distance"].astype(float)
        result["DIAClip.Cosine.Similarity"] = result["cos_similarity"].astype(float)
        result["DIAClip.Quantity"] = result["quant_result"].astype(float)
        result["DIAClip.Passed"] = (
            (result["label"].astype(int) == 1)
            & (result["DIAClip.Q.Value"] <= self.fdr)
        )
        return result

    @staticmethod
    def resolve_duplicate_precursors(metadata):
        kept, skipped, resolved = [], [], []
        for precursor_id, group in metadata.groupby("Precursor.Id", sort=False, dropna=False):
            if len(group) == 1:
                kept.append(group.iloc[0])
                continue
            if group["Decoy"].nunique(dropna=False) > 1:
                decoy_rows = group[group["Decoy"] == 1]
                if len(decoy_rows) == 1:
                    kept.append(decoy_rows.iloc[0])
                    resolved.append(str(precursor_id))
                    continue
            skipped.append((str(precursor_id), sorted(group["Decoy"].unique().tolist()), len(group)))

        if resolved:
            LOGGER.warning(
                "Resolved %d duplicate Precursor.Id values by keeping Decoy=1",
                len(resolved),
            )
        for precursor_id, decoys, rows in skipped:
            LOGGER.warning(
                "Skipping ambiguous Precursor.Id=%s (Decoy=%s, rows=%d)",
                precursor_id, decoys, rows,
            )
        output = pd.DataFrame(kept, columns=metadata.columns)
        if output["Precursor.Id"].duplicated().any():
            raise RuntimeError("Duplicate Precursor.Id remained after metadata resolution")
        return output, skipped

    def load_metadata(self):
        if not os.path.isfile(self.report_path):
            raise FileNotFoundError(f"DIA-NN report does not exist: {self.report_path}")
        metadata = pd.read_parquet(self.report_path)
        required = set(self.METADATA_COLUMNS)
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"Missing DIA-NN metadata columns: {missing}")
        return self.resolve_duplicate_precursors(metadata[self.METADATA_COLUMNS].copy())[0]

    def build(self, results_df, basename):
        os.makedirs(self.out_dir, exist_ok=True)
        scored = self.prepare_scores(results_df)
        all_columns = [
            "Precursor.Id", "Modified.Sequence", "Stripped.Sequence",
            "Precursor.Charge", "label", *self.DIACLIP_COLUMNS,
        ]
        all_result = scored[all_columns].copy()
        all_path = os.path.join(self.out_dir, f"{basename}.diaclip.all.parquet")
        all_result.to_parquet(all_path, index=False)

        passed = scored[scored["DIAClip.Passed"]].copy()
        metadata = self.load_metadata()
        final = passed[["Precursor.Id", *self.DIACLIP_COLUMNS]].merge(
            metadata,
            on="Precursor.Id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        unmatched = int((final["_merge"] != "both").sum())
        LOGGER.info(
            "FDR rows=%d, metadata matched=%d, unmatched=%d",
            len(final), len(final) - unmatched, unmatched,
        )
        final = final.drop(columns="_merge")
        final = final[[*self.METADATA_COLUMNS, *self.DIACLIP_COLUMNS]]
        parquet_path = os.path.join(self.out_dir, f"{basename}.diaclip.fdr.parquet")
        tsv_path = os.path.join(self.out_dir, f"{basename}.diaclip.fdr.tsv")
        final.to_parquet(parquet_path, index=False)
        final.to_csv(tsv_path, sep="\t", index=False)
        LOGGER.info("Saved DIA-CLIP results: %s, %s, %s", all_path, parquet_path, tsv_path)
        return all_result, final
