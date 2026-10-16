def _checkpointed_adjusted_confirmation_streaming(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    candidates: pd.DataFrame,
    feature_index: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    policy: PhenomeAdjustmentPolicy,
    checkpoint_root: Path,
    checkpoint_input_hash: str,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if candidates.empty:
        return confirm_replicated_phenome_hits(
            candidates,
            feature_index,
            current,
            history_body,
            policy=policy,
        )
    validation_index = feature_index.loc[
        feature_index["split"].astype(str).eq("validation")
    ].reset_index(drop=True)
    ordered_features = tuple(
        dict.fromkeys(candidates["feature_id"].astype(str).tolist())
    )
    paths: list[Path] = []
    for block_index, feature_block in enumerate(
        _feature_blocks(ordered_features)
    ):
        selected = candidates.loc[
            candidates["feature_id"].astype(str).isin(feature_block)
        ].copy()
        raw_block = _load_phenome_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=feature_block,
            splits=("validation",),
        )
        features = _combine_phenome_feature_block(
            validation_index,
            raw_block,
            feature_block,
        )
        keys = tuple(
            (str(row.phenotype_id), str(row.feature_id), str(row.panel_id))
            for row in selected.itertuples(index=False)
        )
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace="stream_blocks_v1",
            chunk_index=block_index,
            input_hash=checkpoint_input_hash,
            item_keys=keys,
            compute=lambda block_hits=selected, block_features=features, index=block_index: (
                confirm_replicated_phenome_hits(
                    block_hits,
                    block_features,
                    current,
                    history_body,
                    policy=policy,
                    checkpoint_root=(
                        checkpoint_root / f"block-{index:04d}"
                    ),
                    checkpoint_input_hash=checkpoint_input_hash,
                    progress=None,
                    finalize=False,
                )
            ),
        )
        paths.append(path)
        if progress:
            progress(
                f"PheWAS Adjustierungsblock {block_index + 1:,}/"
                f"{len(_feature_blocks(ordered_features)):,}: "
                f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
            )
        del features, raw_block
    raw_output = materialize_frame_chunks(paths)
    order = {
        (
            str(row.phenotype_id),
            str(row.feature_id),
            str(row.panel_id),
        ): index
        for index, row in enumerate(candidates.itertuples(index=False))
    }
    if not raw_output.empty:
        raw_output["_candidate_order"] = [
            order[(str(row.phenotype_id), str(row.feature_id), str(row.panel_id))]
            for row in raw_output.itertuples(index=False)
        ]
        raw_output = raw_output.sort_values(
            "_candidate_order", kind="stable"
        ).drop(columns="_candidate_order").reset_index(drop=True)
    return finalize_phenome_adjustment_output(raw_output, policy)

def _overlay_replicated_bootstraps(
    validation_results: pd.DataFrame,
    bootstrapped: pd.DataFrame,
) -> pd.DataFrame:
    output = validation_results.copy()
    output["bootstrap_scope"] = np.where(
        output["replicated"].fillna(False),
        "subject_cluster_replicated_hit",
        "not_requested_not_replicated",
    )
    if bootstrapped.empty:
        return output
    keys = ["phenotype_id", "feature_id", "panel_id"]
    bootstrap_columns = [
        column
        for column in bootstrapped.columns
        if column.endswith("_ci_low")
        or column.endswith("_ci_high")
        or column in {
            "bootstrap_iterations",
            "bootstrap_successful_iterations",
            "bootstrap_unit",
        }
    ]
    updates = bootstrapped[
        keys + bootstrap_columns
    ].drop_duplicates(keys)
    if updates.duplicated(keys).any():
        raise RuntimeError("Duplicate PheWAS bootstrap result keys")
    merged = output.merge(
        updates.assign(_bootstrap_matched=True),
        on=keys,
        how="left",
        suffixes=("", "__bootstrap"),
        validate="one_to_one",
    )
    matched = merged["_bootstrap_matched"].eq(True)
    for column in bootstrap_columns:
        update_column = f"{column}__bootstrap"
        if update_column in merged:
            merged.loc[matched, column] = merged.loc[
                matched,
                update_column,
            ].to_numpy()
            merged = merged.drop(columns=update_column)
    merged = merged.drop(columns="_bootstrap_matched")
    return merged
