from datasets import load_dataset



def adding_rewriting(ds, rewrites_paths, idx="idx"):
    """
    Returns: {datasets.Dataset}
        The original dataset with additional columns `rewritedCode_0`, `rewritedCode_1`, etc.,
        one for each rewrite file provided.
    """


    # Sanity checks for required columns
    if idx not in ds.column_names:
        print(f"{idx} is not inside ds")


    # Mapping function to add a single rewrite column to each row
    def add_column(row, ds_map, i, idx):
        rewrited = ds_map.get(row[idx])  # use .get() to avoid KeyError
        return {f"rewritedCode_{i}": rewrited}

    # Process each rewrite file
    for i, r in enumerate(rewrites_paths):
        
        # Load rewrite JSONL as Dataset
        ds_jsonl = load_dataset("json", data_files=r)["train"]

        # Flatten nested structures (e.g., metadata.index → top-level)
        ds_jsonl = ds_jsonl.flatten()

        # Select only the needed columns
        ds_jsonl = ds_jsonl.select_columns([f"metadata.{idx}", "message.content"])

        # Rename columns to match our convention
        ds_jsonl = ds_jsonl.rename_column(f"metadata.{idx}", idx)
        ds_jsonl = ds_jsonl.rename_column("message.content", "content")

        # Convert rewrite dataset to dict: index → content
        ds_map = {row[idx]: row["content"] for row in ds_jsonl}

        # Add the rewrite column to the main dataset
        ds = ds.map(add_column, fn_kwargs={"ds_map": ds_map, "i": i, "idx": idx})

    return ds
