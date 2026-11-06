import os
import zipfile
from typing import Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class WristXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, img_ext=".jpg"):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_ext = img_ext

        self.classes = self.df.columns[1:].tolist()

        if "image" not in self.df.columns:
            raise ValueError(
                f"CSV must have an 'image' column. Found: {self.df.columns.tolist()}")

        self.label_columns = [c for c in self.df.columns if c != "image"]

        if self.img_ext is None:
            self._path_cache = {
                img_id: self._resolve_path(img_id)
                for img_id in self.df["image"]
            }

    def _resolve_path(self, image_id):
        for ext in (".jpg", ".png", ".jpeg"):
            path = self.image_dir / (image_id + ext)
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for ID: {image_id}")

    def _get_image_path(self, image_id):
        if self.img_ext is not None:
            return self.image_dir / (image_id + self.img_ext)
        return self._path_cache[image_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image"]

        with Image.open(self._get_image_path(image_id)) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        label = torch.from_numpy(
            row[self.label_columns].to_numpy(dtype="float32")
        )

        return image, label

class Spliter():
    LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


    def extract_archive_if_needed(self):
        base_dir = Path("src/dataset")
        archive_dir = base_dir / "archive"
        zip_file = base_dir / "archive.zip"

        if archive_dir.is_dir():
            if any(archive_dir.iterdir()):
                if zip_file.is_file():
                    zip_file.unlink()
                    print(
                        "The 'archive' subdirectory already exists. 'archive.zip' deleted.")
                else:
                    print("The 'archive' subdirectory already exists.")
                return
            else:
                print(
                    "The 'archive' directory exists but is empty. Proceeding with extraction...")

        if zip_file.is_file():
            try:
                print("Extracting 'archive.zip'...")

                archive_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    for member in zip_ref.infolist():
                        member_path = Path(member.filename)
                        if member_path.parts and member_path.parts[0] == "archive":
                            relative_path = Path(*member_path.parts[1:])
                        else:
                            relative_path = member_path

                        target_path = archive_dir / relative_path

                        if member.is_dir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        else:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                target.write(source.read())

                print("Extraction complete.")
                zip_file.unlink()
                print("'archive.zip' has been deleted.")

            except (zipfile.BadZipFile, PermissionError) as e:
                print(f"Extraction failed: {e}")
                print("'archive.zip' was not deleted.")
        else:
            print("Neither the 'archive' folder nor 'archive.zip' was found.")


    def drop_zero_variance_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops only numeric columns with a mathematical variance of exactly 0."""

        numeric_df = df.select_dtypes(include='number')

        zero_var_cols = numeric_df.columns[numeric_df.var() == 0]

        return df.drop(columns=zero_var_cols)


    def load_ground_truth(self, gt_path: Path) -> pd.DataFrame:
        df = pd.read_csv(gt_path)
        df["label"] = df[self.LABEL_COLS].idxmax(axis=1)
        df = self.drop_zero_variance_numeric(df)
        return df


    def stratified_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6

        rare_mask = df["label"].isin(
            df["label"].value_counts()[lambda x: x < 4].index
        )
        rare_df = df[rare_mask]
        main_df = df[~rare_mask]

        train_df, temp_df = train_test_split(
            main_df, test_size=(1 - train_size),
            stratify=main_df["label"], random_state=seed,
        )
        val_ratio_adjusted = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_ratio_adjusted),
            stratify=temp_df["label"], random_state=seed,
        )

        for _, group in rare_df.groupby("label"):
            rows = group.sample(frac=1, random_state=seed).reset_index(drop=True)
            splits = [train_df, val_df, test_df]
            for i, (_, row) in enumerate(rows.iterrows()):
                splits[i % 3] = pd.concat([splits[i % 3], row.to_frame().T])
            train_df, val_df, test_df = splits

        def check_dist(df, name):
            print(f"\n{name}")
            print(df["label"].value_counts(normalize=True))

        check_dist(train_df, "Train")
        check_dist(val_df, "Val")
        check_dist(test_df, "Test")

        return train_df.drop(columns=['label']), val_df.drop(columns=['label']), test_df.drop(columns=['label'])


    def save_splits(
            self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out_dir / "train.csv", index=False)
        val_df.to_csv(out_dir / "val.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)