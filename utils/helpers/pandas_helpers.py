import functools


def index_slice(df, axis=1, **kwargs):
    mask_list = []
    for level, values in kwargs.items():
        values = [values] if not isinstance(values, list) else values
        multiindex = df.columns if axis == 1 else df.index
        mask = multiindex.get_level_values(level).isin(values)
        mask_list += [mask]

    joined_mask = functools.reduce(lambda x, y: x & y, mask_list)
    slice_df = df.loc[:, joined_mask] if axis == 1 else df.loc[joined_mask, :]
    return slice_df


def keep_levels(df, levels_to_keep: list, axis=1):
    multiindex_names = df.columns.names if axis == 1 else df.index.names
    levels_to_drop = [level for level in multiindex_names if level not in levels_to_keep]
    df = df.droplevel(levels_to_drop, axis=axis)
    return df
