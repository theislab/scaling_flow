"""Data splitter for creating train/validation/test splits from TrainingData objects."""

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split

from scaleflow.data._data import MappedCellData, TrainingData

logger = logging.getLogger(__name__)

SplitType = Literal["holdout_groups", "holdout_combinations", "random", "stratified"]


class DataSplitter:
    """
    A lightweight class for creating train/validation/test splits from TrainingData objects.

    This class extracts metadata from TrainingData objects and returns split indices,
    making it memory-efficient for large datasets.

    Supports various splitting strategies:
    - holdout_groups: Hold out specific groups (perturbations, cell lines, etc.) for validation/test
    - holdout_combinations: Keep single treatments in training, hold out combinations for validation/test
    - random: Random split of cells
    - stratified: Stratified split maintaining proportions

    Parameters
    ----------
    training_datasets : list[TrainingData | MappedCellData]
        List of TrainingData or MappedCellData objects to process
    dataset_names : list[str]
        List of names for each dataset (for saving/loading)
    split_ratios : list[list[float]]
        List of triples, each indicating [train, validation, test] ratios for each dataset.
        Each triple must sum to 1.0. Length must match training_datasets.
    split_type : SplitType
        Type of split to perform
    split_key : str | list[str] | None
        Column name(s) in adata.obs to use for splitting (required for holdout_groups and holdout_combinations).
        Can be a single column or list of columns for combination treatments.
    force_training_values : list[str] | None
        Values that should be forced to appear only in training (e.g., ['control', 'dmso']).
        These values will never appear in validation or test sets.
    control_value : str | list[str] | None
        Value(s) that represent control/untreated condition (e.g., 'control' or ['control', 'dmso']).
        Required for holdout_combinations split type.
    hard_test_split : bool
        If True, validation and test get completely different groups (no overlap).
        If False, validation and test can share groups, split at cell level.
        Applies to all split types for consistent val/test separation control.
    random_state : int
        Random seed for reproducible splits

    Examples
    --------
    >>> # Split by holdout groups with forced training values
    >>> splitter = DataSplitter(
    ...     training_datasets=[train_data1, train_data2],
    ...     dataset_names=["dataset1", "dataset2"],
    ...     split_ratios=[[0.8, 0.2, 0.0], [0.9, 0.1, 0.0]],
    ...     split_type="holdout_groups",
    ...     split_key=["drug1", "drug2"],
    ...     force_training_values=["control", "dmso"],
    ... )
    >>> # Split by holding out combinations (singletons in training)
    >>> splitter = DataSplitter(
    ...     training_datasets=[train_data],
    ...     dataset_names=["dataset"],
    ...     split_ratios=[[0.8, 0.2, 0.0]],
    ...     split_type="holdout_combinations",
    ...     split_key=["drug1", "drug2"],
    ...     control_value=["control", "dmso"],
    ... )
    >>> results = splitter.split_all_datasets()
    >>> splitter.save_splits("./splits")

    >>> # Load split information later
    >>> split_info = DataSplitter.load_split_info("./splits", "dataset1")
    >>> train_indices = split_info["indices"]["train"]
    """

    def __init__(
        self,
        training_datasets: list[TrainingData | MappedCellData],
        dataset_names: list[str],
        split_ratios: list[list[float]],
        split_type: SplitType = "random",
        split_key: str | list[str] | None = None,
        force_training_values: list[str] | None = None,
        control_value: str | list[str] | None = None,
        hard_test_split: bool = True,
        random_state: int = 42,
    ):
        self.training_datasets = training_datasets
        self.dataset_names = dataset_names
        self.split_ratios = split_ratios
        self.split_type = split_type
        self.split_key = split_key
        self.force_training_values = force_training_values or []
        self.control_value = [control_value] if isinstance(control_value, str) else control_value
        self.hard_test_split = hard_test_split
        self.random_state = random_state

        self._validate_inputs()

        self.split_results: dict[str, dict] = {}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.training_datasets) != len(self.dataset_names):
            raise ValueError(
                f"training_datasets length ({len(self.training_datasets)}) must match "
                f"dataset_names length ({len(self.dataset_names)})"
            )

        if not isinstance(self.split_ratios, list):
            raise ValueError("split_ratios must be a list of lists")

        if len(self.split_ratios) != len(self.training_datasets):
            raise ValueError(
                f"split_ratios length ({len(self.split_ratios)}) must match "
                f"training_datasets length ({len(self.training_datasets)})"
            )

        # Check each split ratio
        for i, ratios in enumerate(self.split_ratios):
            if not isinstance(ratios, list) or len(ratios) != 3:
                raise ValueError(f"split_ratios[{i}] must be a list of 3 values [train, val, test]")

            if not np.isclose(sum(ratios), 1.0):
                raise ValueError(f"split_ratios[{i}] must sum to 1.0, got {sum(ratios)}")

            if any(ratio < 0 for ratio in ratios):
                raise ValueError(f"All values in split_ratios[{i}] must be non-negative")

        # Check split key requirement
        if self.split_type in ["holdout_groups", "holdout_combinations"] and self.split_key is None:
            raise ValueError(f"split_key must be provided for split_type '{self.split_type}'")

        # Check control_value requirement for holdout_combinations
        if self.split_type == "holdout_combinations" and self.control_value is None:
            raise ValueError("control_value must be provided for split_type 'holdout_combinations'")

        for i, td in enumerate(self.training_datasets):
            if not isinstance(td, (TrainingData, MappedCellData)):
                raise ValueError(f"training_datasets[{i}] must be a TrainingData or MappedCellData object")

    def extract_perturbation_info(self, training_data: TrainingData | MappedCellData) -> dict:
        """
        Extract perturbation information from TrainingData or MappedCellData.

        Parameters
        ----------
        training_data : TrainingData | MappedCellData
            Training data object

        Returns
        -------
        dict
            Dictionary containing:
            - perturbation_covariates_mask: array mapping cells to perturbation indices
            - perturbation_idx_to_covariates: dict mapping perturbation indices to covariate tuples
            - n_cells: total number of cells
        """
        perturbation_covariates_mask = np.asarray(training_data.perturbation_covariates_mask)
        perturbation_idx_to_covariates = training_data.perturbation_idx_to_covariates

        n_cells = len(perturbation_covariates_mask)

        logger.info(f"Extracted perturbation info for {n_cells} cells")
        logger.info(f"Number of unique perturbations: {len(perturbation_idx_to_covariates)}")

        return {
            "perturbation_covariates_mask": perturbation_covariates_mask,
            "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
            "n_cells": n_cells,
        }

    def _get_unique_perturbation_values(
        self, perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    ) -> list[str]:
        """Get all unique covariate values from perturbation dictionary."""
        all_unique_vals = set()
        for covariates in perturbation_idx_to_covariates.values():
            all_unique_vals.update(covariates)
        return list(all_unique_vals)

    def _split_random(self, n_cells: int, split_ratios: list[float]) -> dict[str, np.ndarray]:
        """Perform random split of cells."""
        train_ratio, val_ratio, test_ratio = split_ratios

        # Generate random indices
        indices = np.arange(n_cells)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        if self.hard_test_split:
            # HARD: Val and test are completely separate
            train_end = int(train_ratio * n_cells)
            val_end = train_end + int(val_ratio * n_cells)

            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end] if val_ratio > 0 else np.array([])
            test_idx = indices[val_end:] if test_ratio > 0 else np.array([])

            logger.info("HARD RANDOM SPLIT: Completely separate val/test")
        else:
            # SOFT: Val and test can overlap (split val+test at cell level)
            train_end = int(train_ratio * n_cells)
            train_idx = indices[:train_end]
            val_test_idx = indices[train_end:]

            # Split val+test according to val/test ratios
            if len(val_test_idx) > 0 and val_ratio + test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_idx, test_idx = train_test_split(
                    val_test_idx, train_size=val_size, random_state=self.random_state + 1
                )
            else:
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("SOFT RANDOM SPLIT: Val/test can overlap")

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def _split_by_values(
        self,
        perturbation_covariates_mask: np.ndarray,
        perturbation_idx_to_covariates: dict[int, tuple[str, ...]],
        split_ratios: list[float],
    ) -> dict[str, np.ndarray]:
        """Split by holding out specific perturbations."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for holdout_groups splitting")

        # Get all unique covariate values
        unique_values = self._get_unique_perturbation_values(perturbation_idx_to_covariates)

        # Remove forced training values from consideration for val/test splits
        available_values = [v for v in unique_values if v not in self.force_training_values]
        forced_train_values = [v for v in unique_values if v in self.force_training_values]

        logger.info(f"Total unique values: {len(unique_values)}")
        logger.info(f"Forced training values: {forced_train_values}")
        logger.info(f"Available for val/test: {len(available_values)}")

        n_values = len(available_values)

        if n_values < 3:
            warnings.warn(
                f"Only {n_values} unique values found across columns {self.split_key}. "
                "Consider using random split instead.",
                stacklevel=2,
            )

        # Split values according to ratios
        train_ratio, val_ratio, test_ratio = split_ratios

        # Calculate number of values for each split
        n_train = max(1, int(train_ratio * n_values))
        n_val = int(val_ratio * n_values)
        n_test = n_values - n_train - n_val

        # Ensure we don't exceed total values
        if n_train + n_val + n_test != n_values:
            n_test = n_values - n_train - n_val

        # Shuffle available values for random assignment (excluding forced training values)
        np.random.seed(self.random_state)
        shuffled_values = np.random.permutation(available_values)

        # Assign values to splits
        train_values_random = shuffled_values[:n_train]
        val_values = shuffled_values[n_train : n_train + n_val] if n_val > 0 else []
        test_values = shuffled_values[n_train + n_val :] if n_test > 0 else []

        # Combine forced training values with randomly assigned training values
        train_values = list(train_values_random) + forced_train_values

        logger.info(f"Split values - Train: {len(train_values)}, Val: {len(val_values)}, Test: {len(test_values)}")
        logger.info(f"Train values: {train_values}")
        logger.info(f"Val values: {val_values}")
        logger.info(f"Test values: {test_values}")

        # Create masks by checking which perturbation indices contain which values
        def _get_cells_with_values(values_set):
            """Get cell indices for perturbations containing any of the specified values."""
            if len(values_set) == 0:
                return np.array([], dtype=int)

            # Find perturbation indices that contain any of these values
            matching_pert_indices = []
            for pert_idx, covariates in perturbation_idx_to_covariates.items():
                if any(val in covariates for val in values_set):
                    matching_pert_indices.append(pert_idx)

            # Get cells with these perturbation indices
            if len(matching_pert_indices) == 0:
                return np.array([], dtype=int)

            cell_mask = np.isin(perturbation_covariates_mask, matching_pert_indices)
            return np.where(cell_mask)[0]

        if self.hard_test_split:
            # HARD: Val and test get different values (existing logic)
            train_idx = _get_cells_with_values(train_values)
            val_idx = _get_cells_with_values(val_values)
            test_idx = _get_cells_with_values(test_values)

            logger.info("HARD HOLDOUT GROUPS: Val and test get different values")
        else:
            # SOFT: Val and test can share values, split at cell level
            train_values_all = list(train_values_random) + forced_train_values
            val_test_values = list(val_values) + list(test_values)

            train_idx = _get_cells_with_values(train_values_all)
            val_test_idx = _get_cells_with_values(val_test_values)

            # Split val+test cells according to val/test ratios
            if len(val_test_idx) > 0 and val_ratio + test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_idx, test_idx = train_test_split(
                    val_test_idx, train_size=val_size, random_state=self.random_state + 1
                )
            else:
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("SOFT HOLDOUT GROUPS: Val/test can share values")

        # Log overlap information (important for combination treatments)
        total_assigned = len(set(train_idx) | set(val_idx) | set(test_idx))
        logger.info(f"Total cells assigned to splits: {total_assigned} out of {len(perturbation_covariates_mask)}")

        overlaps = []
        if len(set(train_idx) & set(val_idx)) > 0:
            overlaps.append("train-val")
        if len(set(train_idx) & set(test_idx)) > 0:
            overlaps.append("train-test")
        if len(set(val_idx) & set(test_idx)) > 0:
            overlaps.append("val-test")

        if overlaps:
            logger.warning(
                f"Found overlapping cells between splits: {overlaps}. This is expected with combination treatments."
            )

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def _split_holdout_combinations(
        self,
        perturbation_covariates_mask: np.ndarray,
        perturbation_idx_to_covariates: dict[int, tuple[str, ...]],
        split_ratios: list[float],
    ) -> dict[str, np.ndarray]:
        """Split by keeping singletons in training and holding out combinations for val/test."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for holdout_combinations splitting")
        if self.control_value is None:
            raise ValueError("control_value must be provided for holdout_combinations splitting")

        logger.info("Identifying combinations vs singletons from perturbation covariates")
        logger.info(f"Control value(s): {self.control_value}")

        # Classify each perturbation index as control, singleton, or combination
        control_pert_indices = []
        singleton_pert_indices = []
        combination_pert_indices = []

        for pert_idx, covariates in perturbation_idx_to_covariates.items():
            non_control_values = [c for c in covariates if c not in self.control_value]
            n_non_control = len(non_control_values)

            if n_non_control == 0:
                control_pert_indices.append(pert_idx)
            elif n_non_control == 1:
                singleton_pert_indices.append(pert_idx)
            else:
                combination_pert_indices.append(pert_idx)

        # Get cell indices for each type
        if len(control_pert_indices) > 0:
            control_mask = np.isin(perturbation_covariates_mask, control_pert_indices)
        else:
            control_mask = np.zeros(len(perturbation_covariates_mask), dtype=bool)

        if len(singleton_pert_indices) > 0:
            singleton_mask = np.isin(perturbation_covariates_mask, singleton_pert_indices)
        else:
            singleton_mask = np.zeros(len(perturbation_covariates_mask), dtype=bool)

        if len(combination_pert_indices) > 0:
            combination_mask = np.isin(perturbation_covariates_mask, combination_pert_indices)
        else:
            combination_mask = np.zeros(len(perturbation_covariates_mask), dtype=bool)

        # Count each type
        n_combinations = combination_mask.sum()
        n_singletons = singleton_mask.sum()
        n_controls = control_mask.sum()

        logger.info(f"Found {n_combinations} combination treatments")
        logger.info(f"Found {n_singletons} singleton treatments")
        logger.info(f"Found {n_controls} control treatments")

        if n_combinations == 0:
            warnings.warn("No combination treatments found. Consider using 'holdout_groups' instead.", stacklevel=2)

        # Get indices for each type
        combination_indices = np.where(combination_mask)[0]
        singleton_indices = np.where(singleton_mask)[0]
        control_indices = np.where(control_mask)[0]

        # All singletons and controls go to training
        train_idx = np.concatenate([singleton_indices, control_indices])

        # Split combinations according to the provided ratios
        train_ratio, val_ratio, test_ratio = split_ratios

        if n_combinations > 0:
            # Get perturbation identifiers for combination cells
            # Map each cell to its perturbation tuple (non-control values only)
            perturbation_ids = []
            for cell_idx in combination_indices:
                pert_idx = perturbation_covariates_mask[cell_idx]
                covariates = perturbation_idx_to_covariates[pert_idx]
                # Extract non-control values
                non_control_vals = [c for c in covariates if c not in self.control_value]
                perturbation_id = tuple(sorted(non_control_vals))
                perturbation_ids.append(perturbation_id)

            # Get unique perturbation combinations
            unique_perturbations = list(set(perturbation_ids))
            n_unique_perturbations = len(unique_perturbations)

            logger.info(f"Found {n_unique_perturbations} unique perturbation combinations")

            if self.hard_test_split:
                # HARD TEST SPLIT: Val and test get completely different perturbations
                # Calculate number of perturbation combinations for each split
                n_train_perturbations = int(train_ratio * n_unique_perturbations)
                n_val_perturbations = int(val_ratio * n_unique_perturbations)
                n_test_perturbations = n_unique_perturbations - n_train_perturbations - n_val_perturbations

                # Ensure we don't exceed total perturbations
                if n_train_perturbations + n_val_perturbations + n_test_perturbations != n_unique_perturbations:
                    n_test_perturbations = n_unique_perturbations - n_train_perturbations - n_val_perturbations

                # Shuffle perturbations for random assignment
                np.random.seed(self.random_state)
                shuffled_perturbations = np.random.permutation(unique_perturbations)

                # Assign perturbations to splits
                train_perturbations = (
                    shuffled_perturbations[:n_train_perturbations] if n_train_perturbations > 0 else []
                )
                val_perturbations = (
                    shuffled_perturbations[n_train_perturbations : n_train_perturbations + n_val_perturbations]
                    if n_val_perturbations > 0
                    else []
                )
                test_perturbations = (
                    shuffled_perturbations[n_train_perturbations + n_val_perturbations :]
                    if n_test_perturbations > 0
                    else []
                )

                # Assign all cells with same perturbation to same split
                train_combo_idx = []
                val_combo_idx = []
                test_combo_idx = []

                for i, perturbation_id in enumerate(perturbation_ids):
                    cell_idx = combination_indices[i]
                    if perturbation_id in train_perturbations:
                        train_combo_idx.append(cell_idx)
                    elif perturbation_id in val_perturbations:
                        val_combo_idx.append(cell_idx)
                    elif perturbation_id in test_perturbations:
                        test_combo_idx.append(cell_idx)

                logger.info(
                    f"HARD TEST SPLIT - Perturbation split: Train={len(train_perturbations)}, Val={len(val_perturbations)}, Test={len(test_perturbations)}"
                )

            else:
                # SOFT TEST SPLIT: Val and test can share perturbations, split at cell level
                # First assign perturbations to train vs (val+test)
                n_train_perturbations = int(train_ratio * n_unique_perturbations)
                n_val_test_perturbations = n_unique_perturbations - n_train_perturbations

                # Shuffle perturbations
                np.random.seed(self.random_state)
                shuffled_perturbations = np.random.permutation(unique_perturbations)

                train_perturbations = (
                    shuffled_perturbations[:n_train_perturbations] if n_train_perturbations > 0 else []
                )
                val_test_perturbations = (
                    shuffled_perturbations[n_train_perturbations:] if n_val_test_perturbations > 0 else []
                )

                # Get cells for train perturbations (all go to train)
                train_combo_idx = []
                val_test_combo_idx = []

                for i, perturbation_id in enumerate(perturbation_ids):
                    cell_idx = combination_indices[i]
                    if perturbation_id in train_perturbations:
                        train_combo_idx.append(cell_idx)
                    else:
                        val_test_combo_idx.append(cell_idx)

                # Now split val_test cells according to val/test ratios
                if len(val_test_combo_idx) > 0 and val_ratio + test_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio)
                    np.random.seed(self.random_state + 1)  # Different seed for cell-level split

                    val_combo_idx, test_combo_idx = train_test_split(
                        val_test_combo_idx, train_size=val_size, random_state=self.random_state + 1
                    )
                else:
                    val_combo_idx = np.array([])
                    test_combo_idx = np.array([])

                logger.info(
                    f"SOFT TEST SPLIT - Perturbation split: Train={len(train_perturbations)}, Val+Test={len(val_test_perturbations)}"
                )
                logger.info(f"Cell split within Val+Test: Val={len(val_combo_idx)}, Test={len(test_combo_idx)}")

            # Convert to numpy arrays
            train_combo_idx = np.array(train_combo_idx)
            val_combo_idx = np.array(val_combo_idx)
            test_combo_idx = np.array(test_combo_idx)

            # Combine singletons/controls with assigned combinations
            train_idx = np.concatenate([train_idx, train_combo_idx])
            val_idx = val_combo_idx
            test_idx = test_combo_idx

            logger.info(
                f"Final cell split: Train={len(train_combo_idx)}, Val={len(val_combo_idx)}, Test={len(test_combo_idx)}"
            )
        else:
            val_idx = np.array([])
            test_idx = np.array([])

        logger.info(
            f"Final split - Train: {len(train_idx)} (singletons + controls + {len(train_combo_idx) if n_combinations > 0 else 0} combination cells)"
        )
        logger.info(f"Final split - Val: {len(val_idx)} (combination cells only)")
        logger.info(f"Final split - Test: {len(test_idx)} (combination cells only)")

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def _split_stratified(
        self,
        perturbation_covariates_mask: np.ndarray,
        split_ratios: list[float],
    ) -> dict[str, np.ndarray]:
        """Perform stratified split maintaining proportions of perturbations."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for stratified splitting")

        train_ratio, val_ratio, test_ratio = split_ratios
        # Use perturbation indices as stratification labels
        labels = perturbation_covariates_mask
        indices = np.arange(len(perturbation_covariates_mask))

        if self.hard_test_split:
            # HARD: Val and test get different stratification groups (existing logic)
            if val_ratio + test_ratio > 0:
                train_idx, temp_idx = train_test_split(
                    indices, train_size=train_ratio, stratify=labels, random_state=self.random_state
                )

                if val_ratio > 0 and test_ratio > 0:
                    temp_labels = labels[temp_idx]
                    val_size = val_ratio / (val_ratio + test_ratio)
                    val_idx, test_idx = train_test_split(
                        temp_idx, train_size=val_size, stratify=temp_labels, random_state=self.random_state
                    )
                elif val_ratio > 0:
                    val_idx = temp_idx
                    test_idx = np.array([])
                else:
                    val_idx = np.array([])
                    test_idx = temp_idx
            else:
                train_idx = indices
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("HARD STRATIFIED SPLIT: Val and test get different strata")
        else:
            # SOFT: Val and test can share stratification groups, split at cell level
            if val_ratio + test_ratio > 0:
                train_idx, val_test_idx = train_test_split(
                    indices, train_size=train_ratio, stratify=labels, random_state=self.random_state
                )

                # Split val+test cells (not stratified)
                if len(val_test_idx) > 0 and val_ratio + test_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio)
                    val_idx, test_idx = train_test_split(
                        val_test_idx, train_size=val_size, random_state=self.random_state + 1
                    )
                else:
                    val_idx = np.array([])
                    test_idx = np.array([])
            else:
                train_idx = indices
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("SOFT STRATIFIED SPLIT: Val/test can share strata")

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def split_single_dataset(self, training_data: TrainingData | MappedCellData, dataset_index: int) -> dict:
        """
        Split a single TrainingData or MappedCellData object according to the specified strategy.

        Parameters
        ----------
        training_data : TrainingData | MappedCellData
            Training data object to split
        dataset_index : int
            Index of the dataset to get the correct split ratios

        Returns
        -------
        dict
            Dictionary containing split indices and metadata
        """
        # Extract perturbation information
        pert_info = self.extract_perturbation_info(training_data)
        perturbation_covariates_mask = pert_info["perturbation_covariates_mask"]
        perturbation_idx_to_covariates = pert_info["perturbation_idx_to_covariates"]
        n_cells = pert_info["n_cells"]

        # Get split ratios for this specific dataset
        current_split_ratios = self.split_ratios[dataset_index]

        # Perform split based on strategy
        if self.split_type == "random":
            split_indices = self._split_random(n_cells, current_split_ratios)
        elif self.split_type == "holdout_groups":
            split_indices = self._split_by_values(
                perturbation_covariates_mask, perturbation_idx_to_covariates, current_split_ratios
            )
        elif self.split_type == "holdout_combinations":
            split_indices = self._split_holdout_combinations(
                perturbation_covariates_mask, perturbation_idx_to_covariates, current_split_ratios
            )
        elif self.split_type == "stratified":
            split_indices = self._split_stratified(perturbation_covariates_mask, current_split_ratios)
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

        # Create result dictionary with indices and metadata
        result = {
            "indices": split_indices,
            "metadata": {
                "total_cells": n_cells,
                "split_type": self.split_type,
                "split_key": self.split_key,
                "split_ratios": current_split_ratios,
                "random_state": self.random_state,
            },
        }

        # Add force_training_values and control_value to metadata
        if self.force_training_values:
            result["metadata"]["force_training_values"] = self.force_training_values
        if self.control_value:
            result["metadata"]["control_value"] = self.control_value

        # Add split values information if applicable
        if self.split_type in ["holdout_groups", "holdout_combinations"] and self.split_key:
            unique_values = self._get_unique_perturbation_values(perturbation_idx_to_covariates)

            def _get_split_values(indices):
                """Get all unique covariate values for cells in this split."""
                if len(indices) == 0:
                    return []
                split_vals = set()
                for idx in indices:
                    pert_idx = perturbation_covariates_mask[idx]
                    covariates = perturbation_idx_to_covariates[pert_idx]
                    split_vals.update(covariates)
                return list(split_vals)

            train_values = _get_split_values(split_indices["train"])
            val_values = _get_split_values(split_indices["val"])
            test_values = _get_split_values(split_indices["test"])

            result["split_values"] = {
                "train": train_values,
                "val": val_values,
                "test": test_values,
                "all_unique": unique_values,
            }

        # Log split statistics
        logger.info(f"Split results for {self.dataset_names[dataset_index]}:")
        for split_name, indices in split_indices.items():
            if len(indices) > 0:
                logger.info(f"  {split_name}: {len(indices)} cells")

        return result

    def split_all_datasets(self) -> dict[str, dict]:
        """
        Split all TrainingData objects according to the specified strategy.

        Returns
        -------
        dict[str, dict]
            Nested dictionary with dataset names as keys and split information as values
        """
        logger.info(f"Starting data splitting with strategy: {self.split_type}")
        logger.info(f"Number of datasets: {len(self.training_datasets)}")
        for i, ratios in enumerate(self.split_ratios):
            logger.info(f"Dataset {i} ratios: train={ratios[0]}, val={ratios[1]}, test={ratios[2]}")

        for i, (training_data, dataset_name) in enumerate(zip(self.training_datasets, self.dataset_names, strict=True)):
            logger.info(f"\nProcessing dataset {i}: {dataset_name}")
            logger.info(f"Using split ratios: {self.split_ratios[i]}")

            split_result = self.split_single_dataset(training_data, i)
            self.split_results[dataset_name] = split_result

        logger.info(f"\nCompleted splitting {len(self.training_datasets)} datasets")
        return self.split_results

    def save_splits(self, output_dir: str | Path) -> None:
        """
        Save all split information to the specified directory.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save the split information
        """
        import json
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving splits to: {output_dir}")

        for dataset_name, split_info in self.split_results.items():
            # Save indices as numpy arrays (more efficient for large datasets)
            indices_dir = output_dir / dataset_name / "indices"
            indices_dir.mkdir(parents=True, exist_ok=True)

            for split_name, indices in split_info["indices"].items():
                if len(indices) > 0:
                    indices_file = indices_dir / f"{split_name}_indices.npy"
                    np.save(indices_file, indices)
                    logger.info(f"Saved {split_name} indices: {len(indices)} cells -> {indices_file}")

            # Save metadata as JSON
            metadata_file = output_dir / dataset_name / "metadata.json"
            with open(metadata_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                metadata = split_info["metadata"].copy()
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata -> {metadata_file}")

            # Save split values if available
            if "split_values" in split_info:
                split_values_file = output_dir / dataset_name / "split_values.json"
                with open(split_values_file, "w") as f:
                    json.dump(split_info["split_values"], f, indent=2)
                logger.info(f"Saved split values -> {split_values_file}")

            # Save complete split info as pickle for easy loading
            complete_file = output_dir / dataset_name / "split_info.pkl"
            with open(complete_file, "wb") as f:
                pickle.dump(split_info, f)
            logger.info(f"Saved complete split info -> {complete_file}")

        logger.info("All splits saved successfully")

    @staticmethod
    def load_split_info(split_dir: str | Path, dataset_name: str) -> dict:
        """
        Load split information from disk.

        Parameters
        ----------
        split_dir : str | Path
            Directory containing saved splits
        dataset_name : str
            Name of the dataset

        Returns
        -------
        dict
            Dictionary containing split indices and metadata
        """
        import pickle

        split_dir = Path(split_dir)
        dataset_dir = split_dir / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {dataset_dir}")

        # Load complete split info from pickle
        complete_file = dataset_dir / "split_info.pkl"
        if complete_file.exists():
            with open(complete_file, "rb") as f:
                return pickle.load(f)

        # Fallback: reconstruct from individual files
        logger.warning("Complete split info not found, reconstructing from individual files")

        # Load indices
        indices_dir = dataset_dir / "indices"
        indices = {}
        for split_name in ["train", "val", "test"]:
            indices_file = indices_dir / f"{split_name}_indices.npy"
            if indices_file.exists():
                indices[split_name] = np.load(indices_file)
            else:
                indices[split_name] = np.array([])

        # Load metadata
        import json

        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load split values if available
        split_values = None
        split_values_file = dataset_dir / "split_values.json"
        if split_values_file.exists():
            with open(split_values_file) as f:
                split_values = json.load(f)

        result = {"indices": indices, "metadata": metadata}
        if split_values:
            result["split_values"] = split_values

        return result
