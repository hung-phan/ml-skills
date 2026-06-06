---
name: plotly
description: Interactive charting library — Plotly Express, graph_objects, Dash apps, 3D plots, and animations. Use when building web-embeddable visualizations with hover/zoom/pan or interactive dashboards with Dash.
---

# Plotly Skill Reference

## Why This Exists

**Problem**: Static plots (matplotlib/seaborn) cannot be interacted with in notebooks or web apps — you can't zoom, filter, hover for exact data values, or animate over time without exporting separate images for each view.

**Key insight**: Plotly produces interactive charts backed by JavaScript that work natively in Jupyter, Streamlit, Dash, and any web browser, with hover tooltips, zoom/pan, and animation built in by default.

**Reach for this when**: You need interactivity (hover, zoom, pan, animation), are building a dashboard or web app (Dash), need 3D scatter or surface plots, or want to embed shareable HTML charts. Use seaborn/matplotlib instead for static publication figures where interactivity adds no value.

## plotly.express vs graph_objects

```python
import plotly.express as px          # High-level, concise, DataFrame-oriented
import plotly.graph_objects as go    # Low-level, full control, trace-by-trace

# Express: one-liner from DataFrame
fig = px.scatter(df, x="x", y="y", color="category", size="value")

# graph_objects: manual trace construction
fig = go.Figure(data=go.Scatter(x=df["x"], y=df["y"], mode="markers"))
```

**When to use which:**
- `px` — quick EDA, standard chart types, DataFrame input, automatic legends/axes
- `go` — custom traces, mixed chart types, fine-grained control, non-DataFrame data

Both return `go.Figure`. You can start with `px` and customize with `fig.update_*()`.

---

## Interactive Charts

### Scatter
```python
fig = px.scatter(df, x="col_a", y="col_b", color="label",
                 size="magnitude", hover_data=["extra_col"],
                 marginal_x="histogram", marginal_y="box")
```

### Line
```python
fig = px.line(df, x="epoch", y="loss", color="model",
              line_dash="variant", markers=True)
```

### Bar
```python
fig = px.bar(df, x="category", y="value", color="group",
             barmode="group")  # "stack", "relative", "overlay"
```

### Histogram
```python
fig = px.histogram(df, x="score", nbins=50, color="split",
                   marginal="rug", histnorm="probability density")
```

### Heatmap
```python
fig = px.imshow(matrix, text_auto=True, color_continuous_scale="RdBu_r",
                labels=dict(x="Predicted", y="Actual", color="Count"))
```

---

## Subplots (make_subplots)

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Loss", "Accuracy", "LR Schedule", "Grad Norm"),
    specs=[[{}, {}], [{"secondary_y": True}, {}]],  # per-cell config
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

fig.add_trace(go.Scatter(x=epochs, y=loss, name="Loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=epochs, y=acc, name="Acc"), row=1, col=2)
fig.add_trace(go.Bar(x=epochs, y=lr, name="LR"), row=2, col=1)
fig.add_trace(go.Scatter(x=epochs, y=grad, name="Grad"), row=2, col=2)

fig.update_layout(height=800, showlegend=True)
```

**Mixed subplot types:**
```python
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy"}, {"type": "scene"}]]  # 2D + 3D side by side
)
```

---

## 3D Plots

### Scatter3D
```python
fig = px.scatter_3d(df, x="x", y="y", z="z", color="cluster",
                    size="importance", symbol="category",
                    opacity=0.7)

# With graph_objects for more control
fig = go.Figure(data=go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.8)
))
```

### Surface
```python
fig = go.Figure(data=go.Surface(
    z=z_matrix, x=x_range, y=y_range,
    colorscale="Plasma", showscale=True
))
fig.update_layout(scene=dict(
    xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
))
```

### Embedding Visualization (t-SNE/UMAP)
```python
fig = px.scatter_3d(df, x="dim0", y="dim1", z="dim2",
                    color="label", hover_name="text",
                    title="Embedding Space")
fig.update_traces(marker_size=3)
```

---

## Animations (Frames)

```python
# Express animation (simplest)
fig = px.scatter(df, x="x", y="y", color="category",
                 animation_frame="epoch",       # slider
                 animation_group="sample_id",   # track identity across frames
                 range_x=[0, 10], range_y=[0, 10])  # fix axes

# Manual frames with graph_objects
frames = [go.Frame(data=[go.Scatter(x=data[k]["x"], y=data[k]["y"])],
                   name=str(k)) for k in sorted(data.keys())]

fig = go.Figure(data=[go.Scatter(x=data[0]["x"], y=data[0]["y"])],
                frames=frames,
                layout=go.Layout(
                    updatemenus=[dict(type="buttons", buttons=[
                        dict(label="Play", method="animate",
                             args=[None, {"frame": {"duration": 100}}]),
                        dict(label="Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0},
                                            "mode": "immediate"}])
                    ])]
                ))
```

---

## Dash Integration Basics

```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id="metric", options=["loss", "accuracy"], value="loss"),
    dcc.Graph(id="chart")
])

@app.callback(Output("chart", "figure"), Input("metric", "value"))
def update(metric):
    return px.line(df, x="epoch", y=metric, color="model")

app.run(debug=True, port=8050)  # bind 127.0.0.1 for local
```

**Key Dash components:**
- `dcc.Graph(figure=fig)` — embed any Plotly figure
- `dcc.Slider`, `dcc.RangeSlider` — numeric inputs
- `dcc.Interval(interval=5000)` — auto-refresh for live dashboards
- Callbacks chain `Input` → function → `Output` reactively

---

## Figure Updating

```python
# update_layout — global figure properties
fig.update_layout(
    title=dict(text="Model Performance", x=0.5),
    template="plotly_dark",       # "plotly", "plotly_white", "ggplot2", "seaborn"
    font_size=12,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis_title="Epoch",
    yaxis_title="Loss",
    margin=dict(l=40, r=40, t=60, b=40),
    width=900, height=500
)

# update_traces — modify trace properties
fig.update_traces(
    marker=dict(size=8, line=dict(width=1, color="black")),
    selector=dict(mode="markers")  # only affect marker traces
)

# update_xaxes / update_yaxes — axis-specific
fig.update_yaxes(type="log", row=1, col=1)
fig.update_xaxes(tickangle=45, dtick=5)

# add_hline / add_vline / add_annotation
fig.add_hline(y=0.95, line_dash="dash", line_color="green",
              annotation_text="Target")
fig.add_vrect(x0="2024-01-01", x1="2024-02-01", fillcolor="red",
              opacity=0.1, line_width=0)
```

---

## Color Scales

```python
# Built-in sequential: Viridis, Plasma, Inferno, Magma, Cividis, Turbo
# Built-in diverging: RdBu, RdYlGn, Picnic, Portland
# Built-in qualitative: Plotly, D3, Set1, Pastel, Bold

# Apply to continuous color
fig = px.scatter(df, x="x", y="y", color="score",
                 color_continuous_scale="Viridis",
                 range_color=[0, 1])

# Apply to discrete color
fig = px.scatter(df, x="x", y="y", color="label",
                 color_discrete_sequence=px.colors.qualitative.Set2)

# Custom discrete mapping
fig = px.bar(df, x="model", y="f1", color="model",
             color_discrete_map={"bert": "#1f77b4", "gpt": "#ff7f0e"})

# Midpoint for diverging
fig = px.imshow(corr_matrix, color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0)
```

---

## Hover Templates

```python
# Simple customization
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<br>N=%{customdata[0]}<extra></extra>"
)

# With customdata for extra fields
fig = px.scatter(df, x="x", y="y", custom_data=["name", "category"])
fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>"
                  "Category: %{customdata[1]}<br>"
                  "(%{x:.2f}, %{y:.2f})<extra></extra>"
)

# Suppress secondary box
# <extra></extra> removes the trace name box
```

**Format specifiers:** `%{y:.2f}` (2 decimals), `%{x:,}` (thousands separator), `%{z:.1%}` (percentage)

---

## Exporting

### HTML (interactive, self-contained)
```python
fig.write_html("report.html", include_plotlyjs="cdn")  # smaller file
fig.write_html("report.html", include_plotlyjs=True)   # fully offline
fig.to_html(full_html=False)  # returns HTML string (div only, for embedding)
```

### Static Images
```python
# Requires kaleido: pip install -U kaleido
fig.write_image("plot.png", width=1200, height=600, scale=2)  # 2x for retina
fig.write_image("plot.svg")   # vector
fig.write_image("plot.pdf")   # vector

# In-memory bytes
img_bytes = fig.to_image(format="png", width=800, height=400)
```

### JSON (for serialization/caching)
```python
fig.write_json("fig.json")
fig2 = go.Figure(json.loads(open("fig.json").read()))
```

---

## ML-Specific Plots

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)

fig = px.imshow(cm, x=class_names, y=class_names, text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"))
fig.update_layout(title="Confusion Matrix")
```

### ROC Curve
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fig = px.area(x=fpr, y=tpr,
              labels=dict(x="FPR", y="TPR"),
              title=f"ROC Curve (AUC={roc_auc:.3f})")
fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
              line=dict(dash="dash", color="gray"))

# Multi-class ROC
fig = go.Figure()
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc(fpr,tpr):.2f})",
                             mode="lines"))
```

### Feature Importance
```python
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True).tail(20)

fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
             title="Top 20 Feature Importances")
```

### Training Curves
```python
fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss"), row=1, col=1)
fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                         line=dict(dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Acc"), row=1, col=2)
fig.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Acc",
                         line=dict(dash="dash")), row=1, col=2)

fig.add_hline(y=best_val_loss, line_dash="dot", line_color="red",
              annotation_text=f"Best: {best_val_loss:.4f}", row=1, col=1)
fig.update_layout(height=400)
```

### Hyperparameter Search (Parallel Coordinates)
```python
fig = px.parallel_coordinates(
    trials_df,
    dimensions=["lr", "batch_size", "hidden_dim", "dropout", "val_loss"],
    color="val_loss", color_continuous_scale="Viridis_r",
    labels={"lr": "Learning Rate", "val_loss": "Val Loss"}
)
```

### Learning Rate Finder
```python
fig = px.line(x=lrs, y=losses, log_x=True,
              labels=dict(x="Learning Rate", y="Loss"),
              title="LR Finder")
fig.add_vline(x=suggested_lr, line_dash="dash",
              annotation_text=f"Suggested: {suggested_lr:.1e}")
```

### Distribution Comparison (Train vs Test)
```python
fig = go.Figure()
fig.add_trace(go.Histogram(x=train_preds, name="Train", opacity=0.7, nbinsx=50))
fig.add_trace(go.Histogram(x=test_preds, name="Test", opacity=0.7, nbinsx=50))
fig.update_layout(barmode="overlay", title="Prediction Distribution")
```

---

## Large Dataset Handling (WebGL)

```python
# Scattergl — WebGL-accelerated scatter (handles 100K+ points)
fig = go.Figure(data=go.Scattergl(
    x=large_x, y=large_y, mode="markers",
    marker=dict(size=2, color=large_color, colorscale="Viridis",
                line=dict(width=0))
))

# Express with render_mode
fig = px.scatter(df, x="x", y="y", color="label",
                 render_mode="webgl")  # auto-uses Scattergl

# Scatter3d already uses WebGL by default

# Datashader for extreme scale (1M+ points) — pre-rasterize
import datashader as ds
cvs = ds.Canvas(plot_width=800, plot_height=600)
agg = cvs.points(df, "x", "y")
# Then overlay rasterized image on plotly figure
```

**Performance tips:**
- Use `Scattergl` / `render_mode="webgl"` above 10K points
- Reduce `marker.size` and set `marker.line.width=0` for dense plots
- Downsample or aggregate before plotting if > 500K points
- Use `fig.update_layout(uirevision="constant")` to preserve zoom/pan on updates
- Disable hover on huge datasets: `hovermode=False` or `hoverinfo="skip"`

---

## Quick Reference: Common Patterns

```python
# Save reusable template
import plotly.io as pio
pio.templates["custom"] = go.layout.Template(
    layout=dict(font_size=14, title_x=0.5, colorway=px.colors.qualitative.Set2)
)
pio.templates.default = "plotly_white+custom"

# Facets (small multiples)
fig = px.scatter(df, x="x", y="y", color="model",
                 facet_col="dataset", facet_row="metric",
                 facet_col_wrap=3)

# Secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=x, y=loss, name="Loss"), secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=lr, name="LR"), secondary_y=True)

# Figure factory (legacy but useful)
import plotly.figure_factory as ff
fig = ff.create_annotated_heatmap(z=matrix, x=labels, y=labels)
fig = ff.create_dendrogram(X, labels=names)
```

## When to Use

| ✅ Use Plotly | ❌ Don't Use |
|---|---|
| Interactive charts (hover, zoom, pan) | Static PDFs/papers (use matplotlib/seaborn) |
| Dashboards (Dash framework) | Minimal dependencies needed |
| Web-embedded visualizations | Offline batch plotting |
| 3D scatter/surface plots | When interactivity adds no value |
| ML experiment tracking (parallel coords) | Large-scale rendering (>100K points, use datashader) |

**Decision rule**: Need interactivity or web embedding → Plotly. Need publication static → seaborn/matplotlib.

---

## References

- [Plotly Python Documentation](https://plotly.com/python/)