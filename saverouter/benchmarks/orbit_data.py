import os
import logging
import json
import tarfile
from typing import Any, Optional, Tuple,Dict,Type
import re
import pandas as pd
import numpy as np
from datasets import load_dataset
import pickle
import requests
from huggingface_hub import hf_hub_download, snapshot_download


def _safe_extract_tar(archive_path: str, destination: str) -> None:
    """Extract a tar archive while rejecting traversal and link entries."""
    destination_real = os.path.realpath(destination)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Archive contains an unsupported link: {member.name}")
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath([destination_real, target]) != destination_real:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination, members=members)


def whole_group_holdout_split(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete groups while matching the requested train size.

    A dynamic program finds the subset of groups whose number of queries is
    closest to the target. Randomized group order provides deterministic
    seed-dependent tie breaking without ever splitting a group.
    """
    df = pd.DataFrame(dataframe).copy()
    if group_column not in df:
        raise ValueError(f"Missing held-out group column: {group_column}")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must lie strictly between zero and one.")
    group_values = df[group_column].astype(str)
    counts = group_values.value_counts(sort=False)
    if len(counts) < 2:
        raise ValueError("Whole-group holdout requires at least two groups.")

    rng = np.random.default_rng(int(seed))
    names = counts.index.to_numpy()[rng.permutation(len(counts))]
    target = int(round(float(train_ratio) * len(df)))
    # Map reachable query counts to one tuple of selected group names.
    reachable = {0: ()}
    for name in names:
        count = int(counts[name])
        additions = {
            total + count: chosen + (str(name),)
            for total, chosen in list(reachable.items())
        }
        for total, chosen in additions.items():
            reachable.setdefault(total, chosen)

    valid = [
        (abs(total - target), total, chosen)
        for total, chosen in reachable.items()
        if chosen and len(chosen) < len(names)
    ]
    _, _, selected = min(valid, key=lambda item: (item[0], item[1]))
    train_mask = group_values.isin(selected)
    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[~train_mask].reset_index(drop=True)
    if set(train_df[group_column].astype(str)) & set(test_df[group_column].astype(str)):
        raise AssertionError("Held-out groups leaked across train and test.")
    return train_df, test_df

class BaseDatasetLoader:
    """
    Base class for dataset loaders.
    Subclasses should set `hf_id` and `default_target_path`.
    """

    hf_id: str = ""
    revision: Optional[str] = None
    default_target_path: str = "./data"
    name: str = "base"
    args: dict

    def __init__(self, target_path: Optional[str], args: dict):
        self.target_path = target_path or self.default_target_path
        self.args = args

    def download(self) -> Any:
        """Download or load dataset. Default uses Hugging Face `load_dataset`."""
        logging.info("[utils.data.py] Loading dataset %s from HF id %s", self.name, self.hf_id)
        dataset = load_dataset(
            self.hf_id,
            revision=self.revision,
            cache_dir=self.target_path,
        )
        return dataset

    def normalize_cost_dataframe(self, df: pd.DataFrame, min_bound: float = 0.1, epsilon: float = 0.1) -> pd.DataFrame:
        cost_columns = [col for col in df.columns if "cost" in col]

        cost_values = df[cost_columns].values
        log_values = np.log1p(cost_values)

        row_mins = log_values.min(axis=1)[:, np.newaxis]
        row_maxs = log_values.max(axis=1)[:, np.newaxis]
        row_ranges = row_maxs - row_mins

        soft_min = row_mins - epsilon * row_ranges
        soft_max = row_maxs + epsilon * row_ranges
        soft_ranges = soft_max - soft_min

        normalized_values = np.where(
            soft_ranges > 0,
            min_bound + (1 - 2 * min_bound) * (log_values - soft_min) / soft_ranges,
            0.5,
        )

        df[cost_columns] = normalized_values
        return df

    def clean_df(self, df: pd.DataFrame, col_idx: int) -> pd.DataFrame:
        """
        Clean the DataFrame by:
        1. Removing rows with NaN values in the specified column
        2. Removing rows that have data in columns after the specified column
        """
        valid_mask = df[df.columns[col_idx]].notna() & df.iloc[:, col_idx + 1 :].isna().all(axis=1)
        return df[valid_mask].reset_index(drop=True)

    def process(self, dataset: Any) -> Any:
        """
        Optional processing step.
        By default returns the dataset unchanged.
        Subclasses may return either:
          - a DataFrame (or dataset-like), or
          - a tuple (model_list, dataframe)
        """
        return dataset

    def split(self, dataset: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple train/test split according to args["dataset"]["split"]["ratios"].
        Returns: (train_df, test_df)
        """
        split_mode = self.args["dataset"]["split"]["mode"]
        if split_mode == "in-domain":
            return self.split_in(dataset)
        return self.split_out(dataset)

    def split_in(self, dataset: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """In-domain split: convert to pandas and return train/test DataFrames."""
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            df = pd.DataFrame(dataset)

        ratios = self.args["dataset"]["split"]["ratios"]
        train_ratio = ratios["train"]
        seed = self.args["seed"]

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_train = int(len(df) * train_ratio)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df = df.iloc[n_train:].reset_index(drop=True)
        return train_df, test_df

    def split_out(self, dataset: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Out-of-domain split with disjoint ``eval_name`` groups."""
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            df = pd.DataFrame(dataset)
        return whole_group_holdout_split(
            df,
            group_column="eval_name",
            train_ratio=self.args["dataset"]["split"]["ratios"]["train"],
            seed=self.args["seed"],
        )

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
        """
        Run full pipeline: download -> process -> split.
        process() may return either:
          - dataframe-like, or
          - (model_list, dataframe)
        Returns: (train_df, test_df, model_list_or_None)
        """
        dataset = self.download()
        processed = self.process(dataset)

        if isinstance(processed, tuple) and len(processed) == 2:
            model_list, df = processed
        else:
            model_list = None
            df = processed

        train_df, test_df = self.split(df)
        return train_df, test_df, model_list


class RouterbenchLoader(BaseDatasetLoader):
    name = "Routerbench"
    hf_id = "withmartian/routerbench"
    revision = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
    default_target_path = "./data/routerbench"

    def download(self) -> str:
        """Download routerbench_raw.pkl via snapshot_download and return its local path."""
        from huggingface_hub import snapshot_download

        os.makedirs(self.target_path, exist_ok=True)

        pickle_path = os.path.join(self.target_path, "routerbench_raw.pkl")
        if os.path.isfile(pickle_path):
            logging.info("[utils.data.py] routerbench_raw.pkl already exists, skip downloading → %s", pickle_path)
            return pickle_path

        snapshot_download(
            repo_id=self.hf_id,
            revision=self.revision,
            repo_type="dataset",
            local_dir=self.target_path,
            local_dir_use_symlinks=False,
            allow_patterns=["**/routerbench_raw.pkl", "routerbench_raw.pkl"],
        )

        if not os.path.isfile(pickle_path):
            # in case it was downloaded into a nested folder
            for root, _, files in os.walk(self.target_path):
                if "routerbench_raw.pkl" in files:
                    os.replace(os.path.join(root, "routerbench_raw.pkl"), pickle_path)
                    break

        if not os.path.isfile(pickle_path):
            raise FileNotFoundError("routerbench_raw.pkl not found after download.")

        logging.info("[utils.data.py] Saved routerbench_raw.pkl → %s", pickle_path)
        return pickle_path

    def process(self, dataset: Any) -> Any:
        """
        Load the pickle, normalize columns, pivot to wide format.
        Returns: (model_list, result_dataframe)
        """

        model_list = [
            "WizardLM/WizardLM-13B-V1.2",
            "claude-instant-v1",
            "claude-v1",
            "claude-v2",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "meta/code-llama-instruct-34b-chat",
            "meta/llama-2-70b-chat",
            "mistralai/mistral-7b-chat",
            "mistralai/mixtral-8x7b-chat",
            "zero-one-ai/Yi-34B-Chat",
        ]
        logging.info("[utils.data.py] Model List: %s", model_list)

        candidate_paths = [
            os.path.join(self.target_path, "routerbench_raw.pkl"),
            "./routerbench_raw.pkl",
            "./data/routerbench/routerbench_raw.pkl",
        ]
        pickle_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if pickle_path is None:
            raise FileNotFoundError(f"routerbench_raw.pkl not found. Searched: {candidate_paths}")

        with open(pickle_path, "rb") as f:
            raw_data = pickle.load(f)

        df = pd.DataFrame(raw_data)

        original_col_by_lower = {col.lower(): col for col in df.columns}

        def resolve_column(name_candidates):
            for cand in name_candidates:
                key = cand.lower()
                if key in original_col_by_lower:
                    return original_col_by_lower[key]
            return None

        sample_id_col = resolve_column(["sample_id", "sampleid", "id", "sample"])
        model_name_col = resolve_column(["model_name", "model", "model_name_str", "modelId"])
        prompt_col = resolve_column(["prompt", "input", "instruction", "query", "context", "prompt_text"])
        eval_name_col = resolve_column(["eval_name", "evaluation", "eval", "task", "eval_name_str"])
        performance_col = resolve_column(["performance", "score", "eval_score"])
        cost_col = resolve_column(["cost", "latency", "time", "inference_cost", "cost_usd"])
        response_col = resolve_column(["model_response", "response", "output", "model_output", "answer"])

        df[model_name_col] = df[model_name_col].astype(str)
        df = df[df[model_name_col].isin(model_list)].copy()

        selected_cols = [sample_id_col, model_name_col, performance_col, cost_col]
        if prompt_col:
            selected_cols.append(prompt_col)
        if eval_name_col:
            selected_cols.append(eval_name_col)
        if response_col:
            selected_cols.append(response_col)

        df = df[selected_cols].copy()
        # RouterBench pickles can expose numeric columns as object dtype.
        df[performance_col] = pd.to_numeric(df[performance_col], errors="coerce")
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

        perf_table = df.pivot_table(index=sample_id_col, columns=model_name_col, values=performance_col, aggfunc="first")
        cost_table = df.pivot_table(index=sample_id_col, columns=model_name_col, values=cost_col, aggfunc="first")
        resp_table = df.pivot_table(index=sample_id_col, columns=model_name_col, values=response_col, aggfunc="first")

        perf_table = perf_table.reindex(columns=model_list)
        cost_table = cost_table.reindex(columns=model_list)
        resp_table = resp_table.reindex(columns=model_list)

        perf_table.columns = [f"model_{i}_performance" for i in range(len(perf_table.columns))]
        cost_table.columns = [f"model_{i}_cost" for i in range(len(cost_table.columns))]
        resp_table.columns = [f"model_{i}_response" for i in range(len(resp_table.columns))]

        extra_series = []
        if prompt_col:
            prompt_series = df.groupby(sample_id_col)[prompt_col].first().rename("prompt")
            extra_series.append(prompt_series)
        if eval_name_col:
            eval_series = df.groupby(sample_id_col)[eval_name_col].first().rename("eval_name")
            extra_series.append(eval_series)

        result_parts = extra_series + [perf_table, cost_table, resp_table]
        result_df = pd.concat(result_parts, axis=1).reset_index()
        return model_list, result_df

    def split_out(self, dataset: Any) -> Any:
        df = pd.DataFrame(dataset)

        ratios = self.args["dataset"]["split"]["ratios"]
        train_ratio = ratios["train"]
        seed = self.args["seed"]
        np.random.seed(seed)

        eval_sizes = df.groupby("eval_name").size()
        eval_names = list(eval_sizes.index)
        eval_counts = list(eval_sizes.values)

        indices = np.arange(len(eval_names))
        np.random.shuffle(indices)

        train_eval_names = []
        train_count = 0
        target_count = len(df) * train_ratio
        test_ratio = 1 - train_ratio
        test_threshold = len(df) * test_ratio * 0.8

        for idx in indices:
            eval_name = eval_names[idx]
            count = eval_counts[idx]

            if count > test_threshold:
                train_eval_names.append(eval_name)
                train_count += count
                continue

            if train_count + count <= target_count * 1.1:
                train_eval_names.append(eval_name)
                train_count += count

        train_df = df[df["eval_name"].isin(train_eval_names)].reset_index(drop=True)
        test_df = df[~df["eval_name"].isin(train_eval_names)].reset_index(drop=True)
        return train_df, test_df

class  MixinstructLoader(BaseDatasetLoader):
    name = "Mixinstruct"
    hf_id = "llm-blender/mix-instruct"
    revision = "c7f77a4ef0515a99d1752c6387c22366d6922da6"
    default_target_path = "./data/mixinstruct"

    def download(self) -> Any:
        """Prefer already downloaded JSONL files before contacting the Hub."""
        local_files = {
            split: os.path.join(self.target_path, f"{split}_data_prepared.jsonl")
            for split in ("train", "val", "test")
        }
        if all(os.path.isfile(path) for path in local_files.values()):
            logging.info(
                "[utils.data.py][Mixinstruct] Loading local JSONL files from %s",
                self.target_path,
            )
            return load_dataset("json", data_files=local_files)
        return super().download()

    def process(self, dataset: Any) -> Any:
        logging.info("[utils.data.py][Mixinstruct] Starting processing...")

        model_list = [
            "alpaca-native",
            "chatglm-6b",
            "dolly-v2-12b",
            "flan-t5-xxl",
            "koala-7B-HF",
            "llama-7b-hf-baize-lora-bf16",
            "moss-moon-003-sft",
            "mpt-7b",
            "mpt-7b-instruct",
            "oasst-sft-4-pythia-12b-epoch-3.5",
            "stablelm-tuned-alpha-7b",
            "vicuna-13b-1.1",
        ]
        logging.info("[utils.data.py][Mixinstruct] Model list (%d): %s", len(model_list), model_list)
        model_costs = {
            0: 13, 1: 6, 2: 12, 3: 11,
            4: 7, 5: 7, 6: 16, 7: 7,
            8: 7, 9: 12, 10: 7, 11: 13,
        }
        max_cost = max(model_costs.values())
        model_costs_norm = {i: cost / max_cost for i, cost in model_costs.items()}
        rows = []
        logging.info("[utils.data.py][Mixinstruct] Extracting performance, cost and response data...")
        for split_name in dataset:
            split_ds = dataset[split_name]
            for item in split_ds:
                eval_name = "unknown"
                raw_id = item.get("id", None)
                if raw_id is not None:
                    id_str = str(raw_id)
                    # Mixinstruct IDs are namespaced as ``source/sample_id``.
                    # Digits are valid parts of a source name (for example,
                    # ``unified_chip2``), so truncating at the first digit turns
                    # sample IDs into thousands of spurious "tasks".
                    eval_name = id_str.split("/", 1)[0].strip() or "unknown"
                row = {
                    "sample_id": item["id"],
                    "prompt": item["instruction"] + " " + item.get("input", ""),
                    "eval_name": eval_name,
                }
                for i in range(len(model_list)):
                    row[f"model_{i}_performance"] = None
                    row[f"model_{i}_response"] = None
                    row[f"model_{i}_cost"] = None

                for cand in item.get("candidates", []):
                    model_name = cand.get("model")
                    if model_name in model_list:
                        model_idx = model_list.index(model_name)
                        score = cand.get("scores", {}).get("bertscore")
                        response = cand.get("text")
                        row[f"model_{model_idx}_performance"] = score
                        row[f"model_{model_idx}_response"] = response
                for i, cost in model_costs_norm.items():
                    row[f"model_{i}_cost"] = cost

                rows.append(row)
        result_df = pd.DataFrame(rows)

        logging.info(
            "[utils.data.py][Mixinstruct] Final dataframe shape: %s | columns: %d",
            result_df.shape, len(result_df.columns)
        )
        return model_list, result_df


    def split_out(self, dataset: Any) -> Any:
        ds = dataset
        if hasattr(ds, "to_pandas"):
            df = ds.to_pandas()
        elif isinstance(ds, pd.DataFrame):
            df = ds.copy()
        else:
            df = pd.DataFrame(ds)

        ratios = self.args["dataset"]["split"]["ratios"]
        train_ratio = ratios["train"]
        seed = self.args["seed"]
        np.random.seed(seed)

        eval_counts = df.groupby('eval_name').size()
        eval_names = list(eval_counts.index)
        counts = list(eval_counts.values)

        indices = np.arange(len(eval_names))
        np.random.shuffle(indices)

        train_eval_names = []
        train_count = 0
        target_count = len(df) * train_ratio
        test_ratio = 1 - train_ratio
        test_threshold = len(df) * test_ratio * 0.8

        for idx in indices:
            eval_name = eval_names[idx]
            count = counts[idx]

            if count > test_threshold:
                train_eval_names.append(eval_name)
                train_count += count
                continue

            if train_count + count <= target_count * 1.1:
                train_eval_names.append(eval_name)
                train_count += count

        train_df = df[df['eval_name'].isin(train_eval_names)]
        test_df = df[~df['eval_name'].isin(train_eval_names)]

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, test_df

class MMRBenchLoader(BaseDatasetLoader):
    name = "MMRBench"
    hf_id = "gh0stHunter/MMR-Bench"
    revision = "b2619d8389935171022f0a7e8bee45f80ca249dd"
    default_target_path = "./data/MMRBench"

    def download(self) -> str:
        """
        Download `images.tar.gz` and `MMR-Bench.csv` to `self.target_path`,
        extract the tar.gz into `self.target_path` (expecting multiple folders),
        then delete the tar.gz.
        """
        import tarfile

        os.makedirs(self.target_path, exist_ok=True)

        tar_path = os.path.join(self.target_path, "images.tar.gz")
        csv_path = os.path.join(self.target_path, "MMR-Bench.csv")

        # already prepared: csv exists and there is at least one directory (the extracted folders)
        if os.path.isfile(csv_path) and any(
            os.path.isdir(os.path.join(self.target_path, x)) for x in os.listdir(self.target_path)
        ):
            return self.target_path

        # 1) download only required files
        if not (os.path.isfile(tar_path) and os.path.isfile(csv_path)):
            snapshot_download(
                repo_id=self.hf_id,
                revision=self.revision,
                repo_type="dataset",
                local_dir=self.target_path,
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "**/images.tar.gz", "images.tar.gz",
                    "**/MMR-Bench.csv", "MMR-Bench.csv",
                ],
            )

            # move up if nested
            if not os.path.isfile(tar_path):
                for root, _, files in os.walk(self.target_path):
                    if "images.tar.gz" in files:
                        os.replace(os.path.join(root, "images.tar.gz"), tar_path)
                        break

            if not os.path.isfile(csv_path):
                for root, _, files in os.walk(self.target_path):
                    if "MMR-Bench.csv" in files:
                        os.replace(os.path.join(root, "MMR-Bench.csv"), csv_path)
                        break

        if not os.path.isfile(tar_path):
            raise FileNotFoundError("images.tar.gz not found after download.")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("MMR-Bench.csv not found after download.")

        _safe_extract_tar(tar_path, self.target_path)

        try:
            os.remove(tar_path)
        except OSError:
            pass

        return self.target_path

    def process(self, dataset: Any) -> Any:
        model_dict = {
            0: "Qwen2.5-VL-3B-Instruct",
            1: "Qwen2.5-VL-72B-Instruct",
            2: "InternVL3-78B",
            3: "Gemma3-4B",
            4: "Qwen2.5-VL-7B-Instruct",
            5: "Claude3-7V_Sonnet",
            6: "gpt-5-nano-2025-08-07",
            7: "GeminiPro2-5",
            8: "gpt-5-2025-08-07",
            9: "GeminiFlash2-5"
        }

        model_list = [model_dict[i] for i in range(len(model_dict))]
        logging.info(f"[utils.data.py] Model List: {model_list}")

        data_dir = self.target_path

        csv_files = os.path.join(data_dir, "MMR-Bench.csv")

        df = pd.read_csv(csv_files)

        def get_image_path(dataset_idx):
            match = re.search(r'^(.*)_([^_]+)$', dataset_idx)
            subdir = match.group(1)
            qid = match.group(2)
            if subdir == "SEEDBench2_Plus":
                subdir = "SEEDBenchv2Plus"

            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = f"{subdir}/{qid}{ext}"
                full_path = f"{data_dir}/{img_path}"
                if os.path.exists(full_path):
                    return subdir, full_path.replace('\\', '/')

        eval_name, image_path = zip(*df['dataset_idx'].apply(get_image_path))
        df['image_path'] = image_path
        df['eval_name'] = eval_name

        logging.info(f"[utils.data.py] Combined DataFrame shape: {df.shape}")

        new_columns = ['sample_id', 'prompt', 'answer', 'eval_name', 'image_path']
        for model_id,model_name in model_dict.items():
            new_columns.append(f'model_{model_id}_performance')
            new_columns.append(f'model_{model_id}_cost')
            new_columns.append(f'model_{model_id}_prediction')

        new_df = pd.DataFrame(columns=new_columns)
        column_mapping = {
            'sample_id': 'dataset_idx',
            'prompt': 'question',
            'answer': 'answer',
            'eval_name': 'eval_name',
            'image_path': 'image_path'
        }
        for new_col, old_col in column_mapping.items():
            new_df[new_col] = df[old_col]
        for model_id, model_name in model_dict.items():
            perf_col = f'{model_name}_correct'
            cost_col = f'{model_name}_cost'
            pred_col = f'{model_name}_prediction'
            raw_performance = df[perf_col]
            lowered = raw_performance.astype(str).str.strip().str.lower()
            numeric_performance = pd.to_numeric(raw_performance, errors="coerce")
            boolean_performance = lowered.map({"true": 1.0, "false": 0.0})
            new_df[f'model_{model_id}_performance'] = numeric_performance.fillna(
                boolean_performance
            )
            new_df[f'model_{model_id}_cost'] = pd.to_numeric(
                df[cost_col], errors="coerce"
            )
            new_df[f'model_{model_id}_prediction'] = df[pred_col]
        logging.info(f"[utils.data.py] Final DataFrame shape: {new_df.shape}")
        return model_list, new_df

    def split_out(self, dataset: Any) -> Any:
        ds = dataset
        if hasattr(ds, "to_pandas"):
            df = ds.to_pandas()
        elif isinstance(ds, pd.DataFrame):
            df = ds.copy()
        else:
            df = pd.DataFrame(ds)

        ratios = self.args["dataset"]["split"]["ratios"]
        train_ratio = ratios["train"]
        seed = self.args["seed"]
        np.random.seed(seed)

        eval_counts = df.groupby('eval_name').size()
        eval_names = list(eval_counts.index)
        counts = list(eval_counts.values)

        indices = np.arange(len(eval_names))
        np.random.shuffle(indices)

        train_eval_names = []
        train_count = 0
        target_count = len(df) * train_ratio
        test_ratio = 1 - train_ratio
        test_threshold = len(df) * test_ratio * 0.8

        for idx in indices:
            eval_name = eval_names[idx]
            count = counts[idx]

            if count > test_threshold:
                train_eval_names.append(eval_name)
                train_count += count
                continue

            if train_count + count <= target_count * 1.1:
                train_eval_names.append(eval_name)
                train_count += count

        train_df = df[df['eval_name'].isin(train_eval_names)]
        test_df = df[~df['eval_name'].isin(train_eval_names)]

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, test_df

class RouterEvalLoader(BaseDatasetLoader):
    name = "RouterEval"
    hf_id = "linggm/Routereval"
    revision = "0ee4de2661854b588474c211bebe97dc1e434237"
    default_target_path = "./data/routerEval"

    def download(self) -> str:
        """
        Download `router_dataset.zip`, unzip to `router_dataset/`,
        move its contents to `self.target_path`, then delete the zip and folder.
        """
        import zipfile
        import shutil

        os.makedirs(self.target_path, exist_ok=True)

        zip_path = os.path.join(self.target_path, "router_dataset.zip")
        extracted_dir = os.path.join(self.target_path, "router_dataset")

        # already prepared: has pkl files and no zip pending
        if os.path.isdir(self.target_path) and any(f.endswith(".pkl") for f in os.listdir(self.target_path)):
            return self.target_path

        # 1) download only router_dataset.zip
        if not os.path.isfile(zip_path):
            snapshot_download(
                repo_id=self.hf_id,
                revision=self.revision,
                repo_type="dataset",
                local_dir=self.target_path,
                local_dir_use_symlinks=False,
                allow_patterns=["**/router_dataset.zip", "router_dataset.zip"],
            )
            if not os.path.isfile(zip_path):
                for root, _, files in os.walk(self.target_path):
                    if "router_dataset.zip" in files:
                        os.replace(os.path.join(root, "router_dataset.zip"), zip_path)
                        break

        if not os.path.isfile(zip_path):
            raise FileNotFoundError("router_dataset.zip not found after download.")

        # 2) unzip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.target_path)

        if not os.path.isdir(extracted_dir):
            raise FileNotFoundError("router_dataset/ not found after unzip.")

        # 3) move contents up
        for name in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, name)
            dst = os.path.join(self.target_path, name)
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            shutil.move(src, dst)

        # 4) cleanup
        os.remove(zip_path)
        shutil.rmtree(extracted_dir, ignore_errors=True)

        return self.target_path

    def process(self, dataset: Any) -> Any:
        """
        Complete RouterEval Dataset Processing
        return (model_list, dataframe)
        """
        logging.info("[Routereval] Starting full processing pipeline...")
        logging.info("[Routereval] Step 1: Extracting common models by group...")

        base_path = dataset
        results = self._extract_all_common_models_by_group(base_path)

        group_a_data = {}
        group_b_data = {}

        if results["group_a"]["total_common"] > 0:
            group_a_data = {
                "group_name": "Group_A",
                "files": results["group_a"]["files"],
                "total_common_models": results["group_a"]["total_common"],
                "common_models": sorted(results["group_a"]["common_models"]),
            }

        if results["group_b"]["total_common"] > 0:
            group_b_data = {
                "group_name": "Group_B",
                "files": results["group_b"]["files"],
                "total_common_models": results["group_b"]["total_common"],
                "common_models": sorted(results["group_b"]["common_models"]),
            }

        logging.info("[Routereval] Step 2: Extracting performance data...")

        data_df_a = pd.DataFrame()
        data_df_b = pd.DataFrame()

        # _extract_performance_cost_data indexes every performance/cost column
        # by sorted model name; keep returned names aligned with those columns.
        models_a = sorted(results["group_a"]["common_models"])
        models_b = sorted(results["group_b"]["common_models"])

        if results["group_a"]["total_common"] > 0:
            data_df_a = self._extract_performance_cost_data(group_a_data)
            logging.info("[Routereval] Group A: %d models, %d samples", len(models_a), len(data_df_a))

        if results["group_b"]["total_common"] > 0:
            # data_df_b = self._extract_performance_cost_data(group_b_data)
            logging.info("[Routereval] Group B: %d models, %d samples", len(models_b), len(data_df_b))

        logging.info("[Routereval] Step 3: Merging and normalizing data...")

        data_df_a = self.normalize_cost_dataframe(data_df_a)
        # data_df_b = self.normalize_cost_dataframe(data_df_b)

        return models_a, data_df_a

    def split_out(self, dataset: Any) -> Any:
        ds = dataset
        if hasattr(ds, "to_pandas"):
            df = ds.to_pandas()
        elif isinstance(ds, pd.DataFrame):
            df = ds.copy()
        else:
            df = pd.DataFrame(ds)

        ratios = self.args["dataset"]["split"]["ratios"]
        train_ratio = ratios["train"]
        seed = self.args["seed"]
        np.random.seed(seed)

        eval_counts = df.groupby("eval_name").size()
        eval_names = list(eval_counts.index)
        counts = list(eval_counts.values)

        indices = np.arange(len(eval_names))
        np.random.shuffle(indices)

        train_eval_names = []
        train_count = 0
        target_count = len(df) * train_ratio
        test_ratio = 1 - train_ratio
        test_threshold = len(df) * test_ratio * 0.8

        for idx in indices:
            eval_name = eval_names[idx]
            count = counts[idx]

            if count > test_threshold:
                train_eval_names.append(eval_name)
                train_count += count
                continue

            if train_count + count <= target_count * 1.1:
                train_eval_names.append(eval_name)
                train_count += count

        train_df = df[df["eval_name"].isin(train_eval_names)].reset_index(drop=True)
        test_df = df[~df["eval_name"].isin(train_eval_names)].reset_index(drop=True)
        return train_df, test_df

    def _extract_all_common_models_by_group(self, directory_path):
        from pathlib import Path

        pkl_files = list(Path(directory_path).glob("*.pkl"))
        logging.info("[Routereval] Analyzing %d files...", len(pkl_files))

        file_models_with_cost = {}

        for file_path in pkl_files:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.warning("[Routereval] Failed to load %s: %s", file_path.name, e)
                continue

            models_with_cost = set()

            for difficulty in ["easy", "hard"]:
                if difficulty not in data:
                    continue

                diff_data = data[difficulty]
                for model_count in diff_data.keys():
                    pool_data = diff_data[model_count]
                    for _, group_data in pool_data.items():
                        if "model" not in group_data:
                            continue

                        model_data = group_data["model"]
                        if isinstance(model_data, np.ndarray):
                            model_data = list(model_data)

                        for model in model_data:
                            model_str = str(model).strip()
                            if not model_str:
                                continue

                            cost = self._extract_cost_from_name(model_str)
                            if cost is not None:
                                models_with_cost.add(model_str)

            file_models_with_cost[file_path.name] = models_with_cost
            logging.info("[Routereval] %s: %d models", file_path.name, len(models_with_cost))

        group_a_files = [
            "arc_router_dataset.pkl",
            "gsm8k_router_dataset.pkl",
            "harness_truthfulqa_mc_0_router_dataset.pkl",
            "hellaswag_router_dataset.pkl",
            "mmlu_router_dataset.pkl",
            "winogrande_router_dataset.pkl",
        ]

        group_b_files = [
            "bbh_router_dataset.pkl",
            "gpqa_router_dataset.pkl",
            "ifeval_router_dataset.pkl",
            "math_router_dataset.pkl",
            "mmlu_pro_router_dataset.pkl",
            "musr_router_dataset.pkl",
        ]

        group_a_models = [file_models_with_cost[f] for f in group_a_files if f in file_models_with_cost]
        group_a_common = set.intersection(*group_a_models) if group_a_models else set()

        group_b_models = [file_models_with_cost[f] for f in group_b_files if f in file_models_with_cost]
        group_b_common = set.intersection(*group_b_models) if group_b_models else set()

        return {
            "group_a": {
                "files": group_a_files,
                "common_models": list(group_a_common),
                "total_common": len(group_a_common),
            },
            "group_b": {
                "files": group_b_files,
                "common_models": list(group_b_common),
                "total_common": len(group_b_common),
            },
            "file_models": file_models_with_cost,
        }

    def _extract_performance_cost_data(self, df):
        """extract performance and cost per model per prompt. (FAST VERSION)"""
        group_info = df

        dataset_files = group_info.get("files", [])
        common_models = sorted(list(set(group_info.get("common_models", []))))
        n_common = len(common_models)

        model_to_common_idx = {m: i for i, m in enumerate(common_models)}

        all_data = []

        for dataset_file in dataset_files:
            match = re.search(r"(.+)\.pkl$", dataset_file)
            dataset_name = match.group(1) if match else "unknown_dataset"
            file_path = os.path.join(self.target_path, dataset_file)

            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.error("[Routereval] Cannot load file %s: %s", file_path, e)
                continue

            all_samples = []
            sample_id_to_row = {}

            def _add_split_samples(split_type: str, idx_key: str, prompt_key: str):
                if "split_index" not in data or "prompt" not in data:
                    return
                if idx_key not in data["split_index"] or prompt_key not in data["prompt"]:
                    return

                indices = data["split_index"][idx_key]
                prompts = data["prompt"][prompt_key]
                if isinstance(indices, np.ndarray):
                    indices = indices.tolist()
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()

                for idx, prompt in zip(indices, prompts):
                    clean_prompt = str(prompt).replace("\n", " ").replace("\r", " ")
                    clean_prompt = " ".join(clean_prompt.split())

                    sid = f"{dataset_name}{split_type}{idx}"
                    row_idx = len(all_samples)
                    sample_id_to_row[sid] = row_idx

                    all_samples.append(
                        {
                            "sample_id": sid,
                            "prompt": clean_prompt,
                            "eval_name": dataset_name,
                        }
                    )

            _add_split_samples("train", "train_indices", "train_prompt")
            _add_split_samples("val", "val_indices", "val_prompt")
            _add_split_samples("test", "test_indices", "test_prompt")

            if not all_samples:
                continue

            n_samples = len(all_samples)

            perf_mat = np.full((n_samples, n_common), np.nan, dtype=np.float32)
            cost_mat = np.full((n_samples, n_common), np.nan, dtype=np.float32)

            common_costs = np.array([self._extract_cost_from_name(m) for m in common_models], dtype=np.float32)

            for difficulty in ["easy", "hard"]:
                if difficulty not in data:
                    continue

                diff_data = data[difficulty]
                for pool_size, pool_data in diff_data.items():
                    for _, group_data in pool_data.items():
                        if "model" not in group_data or "data" not in group_data:
                            continue

                        models = group_data["model"]
                        if isinstance(models, np.ndarray):
                            models = models.tolist()
                        models = [str(m) for m in models]

                        data_dict = group_data["data"]

                        local_to_common = []
                        for j, m in enumerate(models):
                            ci = model_to_common_idx.get(m, None)
                            if ci is not None:
                                local_to_common.append((j, ci))

                        if not local_to_common:
                            continue

                        local_cols = np.array([x[0] for x in local_to_common], dtype=np.int64)
                        common_cols = np.array([x[1] for x in local_to_common], dtype=np.int64)

                        for split_type in ["train", "val", "test"]:
                            score_key = f"{split_type}_score"
                            indices_key = f"{split_type}_indices"

                            if score_key not in data_dict:
                                continue
                            scores = data_dict[score_key]
                            if not isinstance(scores, np.ndarray) or scores.ndim != 2:
                                continue

                            if "split_index" not in data or indices_key not in data["split_index"]:
                                continue
                            sample_indices = data["split_index"][indices_key]
                            if isinstance(sample_indices, np.ndarray):
                                sample_indices = sample_indices.tolist()

                            row_idxs = []
                            valid_rows = []
                            for k, sid in enumerate(sample_indices):
                                full_sid = f"{dataset_name}{split_type}{sid}"
                                ri = sample_id_to_row.get(full_sid, None)
                                if ri is not None:
                                    row_idxs.append(ri)
                                    valid_rows.append(k)

                            if not row_idxs:
                                continue

                            row_idxs = np.array(row_idxs, dtype=np.int64)
                            valid_rows = np.array(valid_rows, dtype=np.int64)

                            scores_sub = scores[valid_rows[:, None], local_cols[None, :]].astype(np.float32, copy=False)
                            perf_mat[row_idxs[:, None], common_cols[None, :]] = scores_sub

                            cost_mat[row_idxs[:, None], common_cols[None, :]] = common_costs[common_cols][None, :]

            perf_cols = [f"model_{i}_performance" for i in range(n_common)]
            cost_cols = [f"model_{i}_cost" for i in range(n_common)]

            base_df = pd.DataFrame(all_samples)
            perf_df = pd.DataFrame(perf_mat, columns=perf_cols)
            cost_df = pd.DataFrame(cost_mat, columns=cost_cols)

            data_df = pd.concat([base_df, perf_df, cost_df], axis=1, copy=False).copy()

            all_data.append(data_df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def _extract_cost_from_name(self, model_name):
        "Extract model size, supporting formats like -3b and 3x5b"

        model_name = model_name.lower()

        mult_match = re.search(r"-(\d*\.?\d+(?:[x×]\d*\.?\d+)+)b", model_name)
        if mult_match:
            expr = mult_match.group(1)
            nums = [float(n) if "." in n else int(n) for n in re.split(r"\s*[x×]\s*", expr)]
            result = 1
            for n in nums:
                result *= n
            return int(result)

        num_match = re.search(r"-(\d+(?:\.\d+)?)\s*b\b", model_name)
        if num_match:
            num = num_match.group(1)
            return float(num) if "." in num else int(num)

        return None


# Registry to allow users to add new dataset loaders
class LLMRouterBenchLoader(BaseDatasetLoader):
    """Load the Performance-Cost subset of LLMRouterBench into ORBIT format."""

    name = "LLMRouterBench"
    hf_id = "NPULH/LLMRouterBench"
    revision = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
    default_target_path = "./data/llmrouterbench"
    archive_name = "bench-release.tar.gz"
    cache_name = "llmrouterbench_performance_cost_v3.pkl"
    min_price_fit_samples = 20

    # GPT-5 Chat is intentionally excluded because it has no tau2 tool-use results.
    model_list = [
        "claude-sonnet-4",
        "deepseek-v3-0324",
        "deepseek-v3.1-terminus",
        "deepseek-r1-0528",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gpt-5",
        "qwen3-235b-a22b-2507",
        "qwen3-235b-a22b-thinking-2507",
        "glm-4.6",
        "kimi-k2-0905",
        "intern-s1",
    ]

    dataset_splits = [
        ("aime", "hybrid"),
        ("arenahard", "test"),
        ("gpqa", "test"),
        ("hle", "test"),
        ("livecodebench", "test"),
        ("livemathbench", "test"),
        ("mmlupro", "test_3000"),
        ("simpleqa", "test"),
        ("swe-bench", "verified"),
        ("tau2", "test"),
    ]

    @property
    def cache_path(self) -> str:
        return os.path.join(self.target_path, self.cache_name)

    def download(self) -> str:
        os.makedirs(self.target_path, exist_ok=True)
        if os.path.isfile(self.cache_path):
            logging.info("[LLMRouterBench] Processed cache found: %s", self.cache_path)
            return self.target_path

        bench_root = self._find_bench_root()
        if bench_root is not None:
            logging.info("[LLMRouterBench] Extracted benchmark found: %s", bench_root)
            return self.target_path

        archive_path = hf_hub_download(
            repo_id=self.hf_id,
            revision=self.revision,
            repo_type="dataset",
            filename=self.archive_name,
            local_dir=self.target_path,
        )
        extract_dir = os.path.join(self.target_path, "raw")
        os.makedirs(extract_dir, exist_ok=True)
        self._safe_extract(archive_path, extract_dir)

        if self._find_bench_root() is None:
            raise FileNotFoundError(
                f"Could not find results/bench after extracting {archive_path}."
            )
        return self.target_path

    @staticmethod
    def _safe_extract(archive_path: str, destination: str) -> None:
        destination_real = os.path.realpath(destination)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive contains an unsupported link: {member.name}")
                target = os.path.realpath(os.path.join(destination, member.name))
                if os.path.commonpath([destination_real, target]) != destination_real:
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            archive.extractall(destination, members=members)

    def _find_bench_root(self) -> Optional[str]:
        expected_datasets = {dataset for dataset, _ in self.dataset_splits}
        for root, dirs, _ in os.walk(self.target_path):
            if expected_datasets.issubset(set(dirs)):
                return root
        return None

    @staticmethod
    def _latest_result_file(model_dir: str) -> str:
        files = [
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if name.endswith(".json")
        ]
        if not files:
            raise FileNotFoundError(f"No JSON result file found in {model_dir}")

        def timestamp_key(path: str):
            match = re.search(r"(\d{8})_(\d{6})\.json$", os.path.basename(path))
            if match:
                return (1, match.group(1) + match.group(2))
            return (0, os.path.basename(path))

        return max(files, key=timestamp_key)

    @staticmethod
    def _resolve_model_dir(
        bench_root: str, dataset_name: str, split: str, model_name: str
    ) -> str:
        model_dir_names = [model_name]
        if model_name == "qwen3-235b-a22b-thinking-2507":
            model_dir_names.append("qwen3-235b-a22b-thinking")

        candidates = [
            os.path.join(bench_root, dataset_name, split, directory_name)
            for directory_name in model_dir_names
        ]
        # The release archive stores ArenaHard directly as dataset/model/file.
        if dataset_name == "arenahard":
            candidates.extend(
                os.path.join(bench_root, dataset_name, directory_name)
                for directory_name in model_dir_names
            )

        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            "Missing LLMRouterBench result directory; checked: "
            + ", ".join(candidates)
        )

    @staticmethod
    def _as_finite_float(value: Any, field: str, location: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field} at {location}: {value!r}") from exc
        if not np.isfinite(result):
            raise ValueError(f"Non-finite {field} at {location}: {value!r}")
        return result

    @staticmethod
    def _normalize_record_text(value: Any) -> str:
        return (
            str(value or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\u2028", "\n")
            .replace("\u2029", "\n")
        )

    @staticmethod
    def _text_alignment_key(value: str) -> str:
        return re.sub(r"\s+", "", value, flags=re.UNICODE)

    @staticmethod
    def _has_meaningful_response(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text or text in {"None", "null", "''", "[]", "{}"}:
            return False
        lowered = text.lower()
        failure_markers = ("generation failed", "failed after 10 attempts")
        return not any(marker in lowered for marker in failure_markers)

    @classmethod
    def _fit_token_prices(
        cls, prompt_tokens: np.ndarray, completion_tokens: np.ndarray, costs: np.ndarray
    ) -> Optional[np.ndarray]:
        valid = (
            (costs > 0)
            & (prompt_tokens >= 0)
            & (completion_tokens >= 0)
            & ((prompt_tokens + completion_tokens) > 0)
        )
        if int(valid.sum()) < cls.min_price_fit_samples:
            return None

        token_matrix = np.column_stack(
            (prompt_tokens[valid], completion_tokens[valid])
        ).astype(np.float64) / 1_000_000.0
        rates = np.linalg.lstsq(token_matrix, costs[valid], rcond=None)[0]
        if not np.isfinite(rates).all() or np.any(rates < 0) or rates.sum() <= 0:
            return None
        return rates

    @classmethod
    def _repair_costs(cls, df: pd.DataFrame, n_models: int) -> pd.DataFrame:
        all_original_cols = [
            f"model_{i}_cost_usd_original" for i in range(n_models)
        ]
        all_positive_costs = df[all_original_cols].to_numpy(dtype=np.float64)
        all_positive_costs = all_positive_costs[all_positive_costs > 0]
        if len(all_positive_costs) == 0:
            raise ValueError("LLMRouterBench has no positive costs for cost repair.")
        global_median = float(np.median(all_positive_costs))

        repair_counts = {
            "token_reprice": 0,
            "task_median": 0,
            "original_zero": 0,
        }

        for model_idx in range(n_models):
            original_col = f"model_{model_idx}_cost_usd_original"
            prompt_col = f"model_{model_idx}_prompt_tokens"
            completion_col = f"model_{model_idx}_completion_tokens"
            response_col = f"model_{model_idx}_response_present"

            original = df[original_col].to_numpy(dtype=np.float64)
            prompt_tokens = df[prompt_col].to_numpy(dtype=np.float64)
            completion_tokens = df[completion_col].to_numpy(dtype=np.float64)
            response_present = df[response_col].to_numpy(dtype=bool)
            effective = original.copy()
            repaired = np.zeros(len(df), dtype=bool)
            methods = np.full(len(df), "original", dtype=object)

            model_rates = cls._fit_token_prices(
                prompt_tokens, completion_tokens, original
            )
            model_positive = original[original > 0]
            model_median = (
                float(np.median(model_positive))
                if len(model_positive) > 0
                else global_median
            )

            valid_tokens = (prompt_tokens >= 0) & (completion_tokens >= 0)
            token_positive = (prompt_tokens + completion_tokens) > 0
            token_repair_mask = (original == 0) & valid_tokens & token_positive

            for task_name in df.loc[token_repair_mask, "eval_name"].unique():
                task_mask = df["eval_name"].eq(task_name).to_numpy()
                target_mask = token_repair_mask & task_mask
                task_rates = cls._fit_token_prices(
                    prompt_tokens[task_mask],
                    completion_tokens[task_mask],
                    original[task_mask],
                )
                rates = task_rates if task_rates is not None else model_rates
                if rates is None:
                    continue
                estimates = (
                    prompt_tokens[target_mask] * rates[0]
                    + completion_tokens[target_mask] * rates[1]
                ) / 1_000_000.0
                valid_estimates = np.isfinite(estimates) & (estimates > 0)
                target_indices = np.flatnonzero(target_mask)
                valid_indices = target_indices[valid_estimates]
                effective[valid_indices] = estimates[valid_estimates]
                repaired[valid_indices] = True
                methods[valid_indices] = "token_reprice"

            unresolved_token_repair = token_repair_mask & ~repaired
            missing_usage_mask = (
                (original == 0)
                & valid_tokens
                & ~token_positive
                & response_present
            )
            invalid_token_mask = (original == 0) & ~valid_tokens
            median_repair_mask = (
                unresolved_token_repair | missing_usage_mask | invalid_token_mask
            )

            for task_name in df.loc[median_repair_mask, "eval_name"].unique():
                task_mask = df["eval_name"].eq(task_name).to_numpy()
                target_mask = median_repair_mask & task_mask
                task_positive = original[task_mask & (original > 0)]
                replacement = (
                    float(np.median(task_positive))
                    if len(task_positive) > 0
                    else model_median
                )
                effective[target_mask] = replacement
                repaired[target_mask] = True
                methods[target_mask] = "task_median"

            df[f"model_{model_idx}_cost_usd_effective"] = effective
            df[f"model_{model_idx}_cost_repaired"] = repaired
            df[f"model_{model_idx}_cost_repair_method"] = methods

            repair_counts["token_reprice"] += int(
                (methods == "token_reprice").sum()
            )
            repair_counts["task_median"] += int((methods == "task_median").sum())
            repair_counts["original_zero"] += int(
                ((methods == "original") & (original == 0)).sum()
            )

        logging.info("[LLMRouterBench] Cost repair summary: %s", repair_counts)
        return df

    @staticmethod
    def _normalize_costs_globally(df: pd.DataFrame, n_models: int) -> pd.DataFrame:
        raw_cols = [f"model_{i}_cost_usd_effective" for i in range(n_models)]
        cost_cols = [f"model_{i}_cost" for i in range(n_models)]
        raw_costs = df[raw_cols].to_numpy(dtype=np.float64)
        if np.any(raw_costs < 0) or not np.isfinite(raw_costs).all():
            raise ValueError("LLMRouterBench contains invalid raw cost values.")

        global_max = float(raw_costs.max())
        if global_max == 0:
            normalized = np.zeros_like(raw_costs)
        else:
            normalized = raw_costs / global_max
        df[cost_cols] = normalized.astype(np.float32)
        return df

    def _build_dataframe(self, bench_root: str) -> pd.DataFrame:
        rows = []
        none_score_count = 0

        for dataset_name, split in self.dataset_splits:
            dataset_rows = {}
            expected_indices = None

            for model_idx, model_name in enumerate(self.model_list):
                model_dir = self._resolve_model_dir(
                    bench_root, dataset_name, split, model_name
                )
                result_path = self._latest_result_file(model_dir)
                with open(result_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)

                records = result.get("records")
                if not isinstance(records, list):
                    raise ValueError(f"Missing records list in {result_path}")

                current_indices = set()
                for record in records:
                    record_index = int(record["index"])
                    current_indices.add(record_index)
                    location = f"{dataset_name}/{split}/{record_index}/{model_name}"
                    prompt = self._normalize_record_text(record.get("prompt", ""))
                    if not prompt:
                        raise ValueError(f"Empty prompt at {location}")

                    row = dataset_rows.setdefault(
                        record_index,
                        {
                            "sample_id": f"{dataset_name}/{split}/{record_index}",
                            "record_index": record_index,
                            "prompt": prompt,
                            "origin_query": self._normalize_record_text(
                                record.get("origin_query", "")
                            ),
                            "eval_name": dataset_name,
                        },
                    )
                    if self._text_alignment_key(
                        row["prompt"]
                    ) != self._text_alignment_key(prompt):
                        raise ValueError(f"Prompt mismatch across models at {location}")

                    raw_score = record.get("score")
                    if raw_score is None:
                        none_score_count += 1
                        score = 0.0
                    else:
                        score = self._as_finite_float(raw_score, "score", location)
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(f"Score outside [0, 1] at {location}: {score}")

                    cost = self._as_finite_float(record.get("cost", 0.0), "cost", location)
                    if cost < 0:
                        raise ValueError(f"Negative cost at {location}: {cost}")

                    prompt_tokens = int(
                        self._as_finite_float(
                            record.get("prompt_tokens", 0), "prompt_tokens", location
                        )
                    )
                    completion_tokens = int(
                        self._as_finite_float(
                            record.get("completion_tokens", 0),
                            "completion_tokens",
                            location,
                        )
                    )
                    response_present = self._has_meaningful_response(
                        record.get("raw_output")
                    ) or self._has_meaningful_response(record.get("prediction"))

                    row[f"model_{model_idx}_performance"] = score
                    row[f"model_{model_idx}_cost_usd_original"] = cost
                    row[f"model_{model_idx}_prompt_tokens"] = prompt_tokens
                    row[f"model_{model_idx}_completion_tokens"] = completion_tokens
                    row[f"model_{model_idx}_response_present"] = response_present

                if expected_indices is None:
                    expected_indices = current_indices
                elif current_indices != expected_indices:
                    missing = sorted(expected_indices - current_indices)[:10]
                    extra = sorted(current_indices - expected_indices)[:10]
                    raise ValueError(
                        f"Record coverage mismatch for {dataset_name}/{split}/{model_name}; "
                        f"missing={missing}, extra={extra}"
                    )

            rows.extend(dataset_rows[index] for index in sorted(dataset_rows))

        df = pd.DataFrame(rows)
        n_models = len(self.model_list)
        required_cols = [
            *(f"model_{i}_performance" for i in range(n_models)),
            *(f"model_{i}_cost_usd_original" for i in range(n_models)),
            *(f"model_{i}_prompt_tokens" for i in range(n_models)),
            *(f"model_{i}_completion_tokens" for i in range(n_models)),
            *(f"model_{i}_response_present" for i in range(n_models)),
        ]
        if df[required_cols].isna().any().any():
            missing = df[required_cols].isna().sum()
            raise ValueError(f"Incomplete LLMRouterBench matrix:\n{missing[missing > 0]}")

        df = self._repair_costs(df, n_models)
        df = self._normalize_costs_globally(df, n_models)
        logging.info(
            "[LLMRouterBench] Built %d queries across %d datasets and %d models; "
            "mapped %d None scores to 0.0",
            len(df),
            df["eval_name"].nunique(),
            n_models,
            none_score_count,
        )
        return df

    def process(self, dataset: Any) -> Any:
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if cached.get("model_list") != self.model_list:
                raise ValueError("LLMRouterBench cache model list does not match the loader.")
            return list(self.model_list), cached["dataframe"]

        bench_root = self._find_bench_root()
        if bench_root is None:
            raise FileNotFoundError("Extracted LLMRouterBench results/bench directory not found.")
        dataframe = self._build_dataframe(bench_root)
        with open(self.cache_path, "wb") as handle:
            pickle.dump(
                {"model_list": list(self.model_list), "dataframe": dataframe},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logging.info("[LLMRouterBench] Saved processed cache: %s", self.cache_path)
        return list(self.model_list), dataframe


# Registry to allow users to add new dataset loaders
_LOADER_REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    RouterbenchLoader.name: RouterbenchLoader,
    MixinstructLoader.name: MixinstructLoader,
    MMRBenchLoader.name: MMRBenchLoader,
    RouterEvalLoader.name: RouterEvalLoader,
    LLMRouterBenchLoader.name: LLMRouterBenchLoader,
}


def register_loader(name: str, loader_cls: Type[BaseDatasetLoader]) -> None:
    """
    Register a new loader class under `name`. Overwrites existing entry if present.
    """
    _LOADER_REGISTRY[name] = loader_cls


def get_loader(name: str) -> Type[BaseDatasetLoader]:
    if name not in _LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: {name!r}. Valid options: {list(_LOADER_REGISTRY.keys())}")
    return _LOADER_REGISTRY[name]


def download_dataset(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Select loader based on args["dataset"]["name"] and run its pipeline.
    Uses args["dataset"]["dataset_dir"] as target_path.
    Returns: (train_df, test_df, model_list_or_None)
    """
    dataset_name = args["dataset"]["name"]
    LoaderCls = get_loader(dataset_name)
    target_path = args["dataset"]["dataset_dir"]

    loader = LoaderCls(target_path=target_path, args=args)
    return loader.run()
