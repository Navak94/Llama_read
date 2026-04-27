import os
import pandas as pd
import plotly.graph_objects as go

MODE_FILES = {
    1: "migration_mode_1.csv",
    2: "migration_mode_2.csv",
    3: "migration_mode_3.csv",
    4: "migration_mode_4.csv",
}

def normalize_stage_column(df, col, fill_value="not_used"):
    if col not in df.columns:
        df[col] = fill_value
    df[col] = df[col].fillna(fill_value).astype(str).str.strip()
    df[col] = df[col].replace("", fill_value)
    return df

def make_stage_flow(df, source_col, target_col, source_prefix, target_prefix):
    flow = (
        df.groupby([source_col, target_col])
          .size()
          .reset_index(name="value")
    )
    flow["source_node"] = source_prefix + ": " + flow[source_col]
    flow["target_node"] = target_prefix + ": " + flow[target_col]
    return flow[["source_node", "target_node", "value"]]

def make_sankey_for_mode(mode, csv_file):
    df = pd.read_csv(csv_file)

    if mode == 1:
        stage_cols = ["pressure_url_1", "pressure_article", "pressure_final"]
        stage_labels = ["URL1_7B", "ARTICLE_7B", "FINAL"]

    elif mode == 2:
        stage_cols = ["pressure_url_1", "pressure_url_2", "pressure_final"]
        stage_labels = ["URL1_7B", "URL2_72B", "FINAL"]

    elif mode == 3:
        stage_cols = ["pressure_url_1", "pressure_url_2", "pressure_article", "pressure_final"]
        stage_labels = ["URL1_7B", "URL2_72B", "ARTICLE_7B", "FINAL"]

    elif mode == 4:
        stage_cols = ["pressure_url_1", "pressure_article", "pressure_final"]
        stage_labels = ["URL1_MISTRAL", "ARTICLE_MISTRAL", "FINAL"]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for col in stage_cols:
        normalize_stage_column(df, col)

    all_flows = []
    for i in range(len(stage_cols) - 1):
        flow = make_stage_flow(
            df,
            stage_cols[i],
            stage_cols[i + 1],
            stage_labels[i],
            stage_labels[i + 1]
        )
        all_flows.append(flow)

    all_flows = pd.concat(all_flows, ignore_index=True)

    labels = pd.unique(
        pd.concat([all_flows["source_node"], all_flows["target_node"]], ignore_index=True)
    ).tolist()

    label_to_index = {label: i for i, label in enumerate(labels)}

    sources = all_flows["source_node"].map(label_to_index).tolist()
    targets = all_flows["target_node"].map(label_to_index).tolist()
    values = all_flows["value"].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Migration Label Flow - Mode {mode}",
        font_size=11
    )

    output_html = f"migration_sankey_mode_{mode}.html"
    fig.write_html(output_html)
    print(f"Saved: {output_html}")

if __name__ == "__main__":
    for mode, csv_file in MODE_FILES.items():
        if os.path.exists(csv_file):
            make_sankey_for_mode(mode, csv_file)
        else:
            print(f"Skipping mode {mode}, file not found: {csv_file}")