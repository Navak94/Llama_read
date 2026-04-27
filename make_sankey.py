import pandas as pd
import plotly.graph_objects as go

CSV_FILE = "migration_mode_3.csv"   # change if needed
OUTPUT_HTML = "migration_sankey.html"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_FILE)

# Fill blanks so Sankey doesn't break
for col in ["pressure_url_1", "final_source", "pressure_final"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
    df[col] = df[col].fillna("missing").astype(str).str.strip()
    df[col] = df[col].replace("", "missing")

# Optional: keep only rows with URLs if you want
# df = df[df["sourceurl"].notna()]

# -----------------------------
# Build flows
# -----------------------------
# Stage 1: pressure_url_1 -> final_source
flow1 = (
    df.groupby(["pressure_url_1", "final_source"])
      .size()
      .reset_index(name="value")
)

# Stage 2: final_source -> pressure_final
flow2 = (
    df.groupby(["final_source", "pressure_final"])
      .size()
      .reset_index(name="value")
)

# Prefix node labels so identical text in different stages stays separate
flow1["source_node"] = "URL1: " + flow1["pressure_url_1"]
flow1["target_node"] = "SRC: " + flow1["final_source"]

flow2["source_node"] = "SRC: " + flow2["final_source"]
flow2["target_node"] = "FINAL: " + flow2["pressure_final"]

all_flows = pd.concat([
    flow1[["source_node", "target_node", "value"]],
    flow2[["source_node", "target_node", "value"]]
], ignore_index=True)

# -----------------------------
# Create node index mapping
# -----------------------------
labels = pd.unique(
    pd.concat([all_flows["source_node"], all_flows["target_node"]], ignore_index=True)
).tolist()

label_to_index = {label: i for i, label in enumerate(labels)}

sources = all_flows["source_node"].map(label_to_index).tolist()
targets = all_flows["target_node"].map(label_to_index).tolist()
values = all_flows["value"].tolist()

# -----------------------------
# Plot Sankey
# -----------------------------
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
    title_text=f"Sankey Flow for {CSV_FILE}",
    font_size=12
)

fig.write_html(OUTPUT_HTML)
print(f"Saved Sankey plot to: {OUTPUT_HTML}")