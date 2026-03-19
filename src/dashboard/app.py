from __future__ import annotations

from pathlib import Path

import dash
from dash import Dash, Input, Output, dcc, html, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PALETTE = {
    "bg": "#0B1120",
    "card": "#131C31",
    "card_border": "#1E2D4A",
    "accent": "#3B82F6",
    "accent_light": "#60A5FA",
    "accent_glow": "rgba(59,130,246,0.15)",
    "gold": "#F59E0B",
    "silver": "#94A3B8",
    "bronze": "#D97706",
    "text": "#F1F5F9",
    "text_muted": "#94A3B8",
    "positive": "#22C55E",
    "table_header": "#1E293B",
    "table_row": "#0F172A",
    "table_row_alt": "#131C31",
    "table_hover": "#1E2D4A",
    "prediction_accent": "#8B5CF6",
    "prediction_glow": "rgba(139,92,246,0.15)",
}

RANK_COLORS = {1: PALETTE["gold"], 2: PALETTE["silver"], 3: PALETTE["bronze"]}

STAT_COLS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "STL": "Steals", "BLK": "Blocks", "FG_PCT": "FG%",
    "FG3_PCT": "3P%", "FT_PCT": "FT%", "PLUS_MINUS": "+/-",
    "MIN": "Minutes", "GP": "Games", "TEAM_WIN_PCT": "Team Win%",
}


def _load_processed() -> pd.DataFrame:
    candidates = sorted(PROCESSED_DIR.glob("training_dataset_*.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[0])


def _load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions_current.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


df = _load_processed()
pred_df = _load_predictions()

available_seasons = sorted(df["season"].unique()) if not df.empty else []
available_awards = sorted(df["AWARD_TYPE"].unique()) if not df.empty else []
pred_awards = sorted(pred_df["AWARD_TYPE"].unique()) if not pred_df.empty else []

year_range = f"{available_seasons[0]} - {available_seasons[-1]}" if available_seasons else "N/A"

app: Dash = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    title="NBA Awards Predictor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


def _card(children, **kwargs):
    return html.Div(
        children,
        style={
            "background": PALETTE["card"],
            "border": f"1px solid {PALETTE['card_border']}",
            "borderRadius": "12px",
            "padding": "24px",
            "marginBottom": "20px",
            **kwargs.pop("style", {}),
        },
        **kwargs,
    )


def _stat_pill(label: str, value: str, color: str = PALETTE["accent_light"]):
    return html.Div(
        [
            html.Span(value, style={"fontSize": "22px", "fontWeight": "700", "color": color}),
            html.Span(label, style={"fontSize": "11px", "color": PALETTE["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        ],
        style={"display": "flex", "flexDirection": "column", "alignItems": "center",
               "padding": "12px 18px", "background": PALETTE["card"],
               "border": f"1px solid {PALETTE['card_border']}", "borderRadius": "10px",
               "minWidth": "90px"},
    )


TAB_STYLE = {
    "backgroundColor": PALETTE["bg"],
    "color": PALETTE["text_muted"],
    "border": "none",
    "borderBottom": f"2px solid transparent",
    "padding": "12px 24px",
    "fontWeight": "500",
    "fontSize": "14px",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": PALETTE["accent_light"],
    "borderBottom": f"2px solid {PALETTE['accent']}",
    "fontWeight": "700",
}


def _historical_tab():
    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"},
            children=[
                html.Div([
                    html.Label("Season", style={"fontSize": "11px", "color": PALETTE["text_muted"],
                                                 "textTransform": "uppercase", "letterSpacing": "1px",
                                                 "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="season-dropdown",
                        options=[{"label": s, "value": s} for s in available_seasons],
                        value=available_seasons[-1] if available_seasons else None,
                        clearable=False,
                        style={"width": "160px", "color": "#000"},
                    ),
                ]),
                html.Div([
                    html.Label("Award", style={"fontSize": "11px", "color": PALETTE["text_muted"],
                                                 "textTransform": "uppercase", "letterSpacing": "1px",
                                                 "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="award-dropdown",
                        options=[{"label": a, "value": a} for a in available_awards],
                        value=available_awards[0] if available_awards else None,
                        clearable=False,
                        style={"width": "160px", "color": "#000"},
                    ),
                ]),
            ],
        ),

        html.Div(id="kpi-row", style={"display": "flex", "gap": "14px",
                                       "flexWrap": "wrap", "marginBottom": "24px"}),

        dbc.Row([
            dbc.Col(_card([html.Div(id="winner-spotlight")], style={"height": "100%"}), md=4),
            dbc.Col(_card([dcc.Graph(id="vote-share-bar", config={"displayModeBar": False})],
                          style={"height": "100%"}), md=8),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(_card([dcc.Graph(id="radar-chart", config={"displayModeBar": False})]), md=6),
            dbc.Col(_card([
                html.Label("Compare stat across seasons", style={
                    "fontSize": "11px", "color": PALETTE["text_muted"],
                    "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="stat-dropdown",
                    options=[{"label": v, "value": k} for k, v in STAT_COLS.items()],
                    value="PTS", clearable=False,
                    style={"width": "200px", "marginBottom": "12px", "color": "#000"},
                ),
                dcc.Graph(id="stat-trend", config={"displayModeBar": False}),
            ]), md=6),
        ], className="g-3"),

        _card([
            html.H3("Full Voting Results", style={"fontSize": "16px", "fontWeight": "600",
                                                   "marginBottom": "16px", "color": PALETTE["text"]}),
            dash_table.DataTable(
                id="awards-table",
                columns=[
                    {"name": "#", "id": "Rank"},
                    {"name": "Player", "id": "player_name"},
                    {"name": "Team", "id": "TEAM_ABBREVIATION"},
                    {"name": "Vote Pts", "id": "vote_points", "type": "numeric", "format": {"specifier": ".0f"}},
                    {"name": "Vote Share", "id": "vote_share", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "1st Votes", "id": "first_place_votes", "type": "numeric"},
                    {"name": "PTS", "id": "PTS", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "REB", "id": "REB", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "AST", "id": "AST", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "GP", "id": "GP", "type": "numeric"},
                    {"name": "Team W%", "id": "TEAM_WIN_PCT", "type": "numeric", "format": {"specifier": ".3f"}},
                ],
                page_size=15, sort_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": PALETTE["table_header"], "color": PALETTE["text_muted"],
                    "fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase",
                    "letterSpacing": "0.5px", "border": "none",
                    "borderBottom": f"2px solid {PALETTE['card_border']}", "padding": "12px 14px",
                },
                style_cell={
                    "backgroundColor": "transparent", "color": PALETTE["text"],
                    "border": "none", "borderBottom": f"1px solid {PALETTE['card_border']}",
                    "padding": "10px 14px", "fontSize": "13px", "fontFamily": "'Inter', sans-serif",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": PALETTE["table_row_alt"]},
                    {"if": {"row_index": "even"}, "backgroundColor": PALETTE["table_row"]},
                ],
            ),
        ]),
    ])


def _predictions_tab():
    if pred_df.empty:
        return _card([
            html.Div(
                style={"textAlign": "center", "padding": "60px 20px"},
                children=[
                    html.H3("No Predictions Yet", style={"color": PALETTE["text_muted"], "fontWeight": "600"}),
                    html.P("Run the prediction pipeline first:",
                           style={"color": PALETTE["text_muted"], "marginTop": "12px"}),
                    html.Code("python -m src.models.predict",
                              style={"color": PALETTE["prediction_accent"], "fontSize": "14px"}),
                ],
            )
        ])

    pred_season = pred_df["season"].iloc[0] if "season" in pred_df.columns else "2025-26"

    return html.Div([
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px",
                   "marginBottom": "24px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Award", style={"fontSize": "11px", "color": PALETTE["text_muted"],
                                                 "textTransform": "uppercase", "letterSpacing": "1px",
                                                 "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="pred-award-dropdown",
                        options=[{"label": a, "value": a} for a in pred_awards],
                        value=pred_awards[0] if pred_awards else None,
                        clearable=False,
                        style={"width": "160px", "color": "#000"},
                    ),
                ]),
                html.Div(
                    [
                        html.Span("LIVE", style={
                            "background": PALETTE["positive"], "color": "#fff",
                            "padding": "3px 10px", "borderRadius": "12px",
                            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "1px",
                        }),
                        html.Span(f"  Season {pred_season} (in progress)",
                                  style={"color": PALETTE["text_muted"], "fontSize": "13px", "marginLeft": "8px"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginTop": "18px"},
                ),
            ],
        ),

        html.Div(id="pred-kpi-row", style={"display": "flex", "gap": "14px",
                                             "flexWrap": "wrap", "marginBottom": "24px"}),

        dbc.Row([
            dbc.Col(_card([html.Div(id="pred-spotlight")], style={"height": "100%"}), md=4),
            dbc.Col(_card([dcc.Graph(id="pred-bar", config={"displayModeBar": False})],
                          style={"height": "100%"}), md=8),
        ], className="g-3"),

        _card([
            html.H3("Full Prediction Rankings", style={"fontSize": "16px", "fontWeight": "600",
                                                        "marginBottom": "16px", "color": PALETTE["text"]}),
            dash_table.DataTable(
                id="pred-table",
                columns=[
                    {"name": "#", "id": "predicted_rank"},
                    {"name": "Player", "id": "player_name"},
                    {"name": "Team", "id": "TEAM_ABBREVIATION"},
                    {"name": "Age", "id": "AGE", "type": "numeric"},
                    {"name": "Exp", "id": "experience_years", "type": "numeric"},
                    {"name": "Pred. Share", "id": "predicted_vote_share", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "PTS", "id": "PTS", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "REB", "id": "REB", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "AST", "id": "AST", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "GP", "id": "GP", "type": "numeric"},
                    {"name": "MIN", "id": "MIN", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "W", "id": "W", "type": "numeric"},
                    {"name": "L", "id": "L", "type": "numeric"},
                    {"name": "Team W%", "id": "TEAM_WIN_PCT", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "W% Impr.", "id": "WIN_PCT_IMPROVEMENT", "type": "numeric", "format": {"specifier": "+.3f"}},
                    {"name": "Media Hype", "id": "media_hype", "type": "numeric", "format": {"specifier": ".0f"}},
                ],
                page_size=20, sort_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": PALETTE["table_header"], "color": PALETTE["text_muted"],
                    "fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase",
                    "letterSpacing": "0.5px", "border": "none",
                    "borderBottom": f"2px solid {PALETTE['card_border']}", "padding": "12px 14px",
                },
                style_cell={
                    "backgroundColor": "transparent", "color": PALETTE["text"],
                    "border": "none", "borderBottom": f"1px solid {PALETTE['card_border']}",
                    "padding": "10px 14px", "fontSize": "13px", "fontFamily": "'Inter', sans-serif",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": PALETTE["table_row_alt"]},
                    {"if": {"row_index": "even"}, "backgroundColor": PALETTE["table_row"]},
                ],
            ),
        ]),
    ])


def layout():
    return html.Div(
        style={"background": PALETTE["bg"], "minHeight": "100vh", "color": PALETTE["text"],
               "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"},
        children=[
            html.Div(
                style={"background": f"linear-gradient(135deg, {PALETTE['card']} 0%, {PALETTE['bg']} 100%)",
                       "borderBottom": f"1px solid {PALETTE['card_border']}",
                       "padding": "28px 40px 0"},
                children=[
                    html.Div(
                        style={"maxWidth": "1400px", "margin": "0 auto"},
                        children=[
                            html.H1("NBA Awards Predictor",
                                    style={"fontSize": "28px", "fontWeight": "800", "margin": "0 0 4px",
                                           "background": f"linear-gradient(135deg, {PALETTE['accent_light']}, {PALETTE['gold']})",
                                           "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
                            html.P(f"Historical analysis ({year_range})  +  Live predictions",
                                   style={"color": PALETTE["text_muted"], "margin": "0 0 16px", "fontSize": "13px"}),
                            dcc.Tabs(
                                id="main-tabs",
                                value="historical",
                                children=[
                                    dcc.Tab(label="Historical Analysis", value="historical",
                                            style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                    dcc.Tab(label="2025-26 Predictions", value="predictions",
                                            style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                ],
                                style={"borderBottom": "none"},
                            ),
                        ],
                    ),
                ],
            ),

            html.Div(
                style={"maxWidth": "1400px", "margin": "0 auto", "padding": "28px 40px"},
                children=[html.Div(id="tab-content")],
            ),
        ],
    )


app.layout = layout


# ── Tab routing ──

@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "predictions":
        return _predictions_tab()
    return _historical_tab()


# ── Historical callbacks ──

def _filtered(season: str, award: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[(df["season"] == season) & (df["AWARD_TYPE"] == award)].sort_values("vote_share", ascending=False)


@callback(Output("kpi-row", "children"),
          [Input("season-dropdown", "value"), Input("award-dropdown", "value")])
def update_kpi(season, award):
    if not season or not award:
        return []
    sub = _filtered(season, award)
    if sub.empty:
        return [html.Span("No data for this selection.", style={"color": PALETTE["text_muted"]})]
    w = sub.iloc[0]
    return [
        _stat_pill("Candidates", str(len(sub))),
        _stat_pill("Winner", str(w["player_name"]), PALETTE["gold"]),
        _stat_pill("Winner Vote Pts", f"{w.get('vote_points', 0):.0f}"),
        _stat_pill("Winner Vote Share", f"{w['vote_share']:.3f}", PALETTE["positive"]),
        _stat_pill("1st Place Votes", f"{int(w.get('first_place_votes', 0))}"),
        _stat_pill("Winner PPG", f"{w.get('PTS', 0):.1f}", PALETTE["accent_light"]),
    ]


@callback(Output("winner-spotlight", "children"),
          [Input("season-dropdown", "value"), Input("award-dropdown", "value")])
def update_spotlight(season, award):
    if not season or not award:
        return []
    sub = _filtered(season, award)
    if sub.empty:
        return html.P("No data.", style={"color": PALETTE["text_muted"]})
    top3 = sub.head(3)
    children = [html.H3("Top 3", style={"fontSize": "16px", "fontWeight": "600",
                                         "marginBottom": "20px", "color": PALETTE["text"]})]
    medals = ["1st", "2nd", "3rd"]
    for i, (_, row) in enumerate(top3.iterrows()):
        color = RANK_COLORS.get(i + 1, PALETTE["text_muted"])
        children.append(
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "14px",
                       "padding": "14px", "borderRadius": "10px", "marginBottom": "10px",
                       "background": f"linear-gradient(135deg, {PALETTE['card']} 0%, {PALETTE['bg']} 100%)",
                       "border": f"1px solid {PALETTE['card_border']}"},
                children=[
                    html.Div(medals[i],
                             style={"background": color, "color": PALETTE["bg"], "fontWeight": "800",
                                    "fontSize": "12px", "borderRadius": "6px",
                                    "padding": "4px 10px", "minWidth": "36px", "textAlign": "center"}),
                    html.Div([
                        html.Div(row["player_name"], style={"fontWeight": "600", "fontSize": "15px"}),
                        html.Div(
                            f"{row.get('TEAM_ABBREVIATION', '?')}  |  "
                            f"{row['vote_share']:.3f} share  |  "
                            f"{row.get('PTS', 0):.1f} PPG",
                            style={"fontSize": "12px", "color": PALETTE["text_muted"], "marginTop": "2px"},
                        ),
                    ]),
                ],
            )
        )
    return children


@callback(Output("vote-share-bar", "figure"),
          [Input("season-dropdown", "value"), Input("award-dropdown", "value")])
def update_vote_share_chart(season, award):
    sub = _filtered(season, award).head(10) if season and award else pd.DataFrame()
    if sub.empty:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    colors = [RANK_COLORS.get(i + 1, PALETTE["accent"]) for i in range(len(sub))]
    fig = go.Figure(go.Bar(
        x=sub["vote_share"], y=sub["player_name"], orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=sub["vote_share"].apply(lambda v: f"{v:.3f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"], size=12),
    ))
    fig.update_layout(
        title=dict(text=f"{award} Vote Share  |  {season}", font=dict(size=14, color=PALETTE["text"])),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=PALETTE["text"])),
        xaxis=dict(title="Vote Share", range=[0, min(1.05, sub["vote_share"].max() * 1.25)],
                   tickfont=dict(color=PALETTE["text_muted"])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=50, b=40), height=350,
    )
    return fig


@callback(Output("radar-chart", "figure"),
          [Input("season-dropdown", "value"), Input("award-dropdown", "value")])
def update_radar(season, award):
    sub = _filtered(season, award).head(5) if season and award else pd.DataFrame()
    if sub.empty:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    radar_stats = [s for s in ["PTS", "REB", "AST", "STL", "BLK"] if s in sub.columns]
    if not radar_stats:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    max_vals = df[radar_stats].max().replace(0, 1)
    palette_list = [PALETTE["gold"], PALETTE["accent_light"], PALETTE["bronze"],
                    PALETTE["positive"], PALETTE["silver"]]
    fig = go.Figure()
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [(row[s] / max_vals[s]) * 100 for s in radar_stats] + [(row[radar_stats[0]] / max_vals[radar_stats[0]]) * 100]
        cats = [STAT_COLS.get(s, s) for s in radar_stats] + [STAT_COLS.get(radar_stats[0], radar_stats[0])]
        c = palette_list[i % len(palette_list)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=row["player_name"],
            line=dict(color=c, width=2),
            fillcolor=c.replace(")", ",0.08)").replace("rgb", "rgba") if c.startswith("rgb") else c + "14",
        ))
    fig.update_layout(
        title=dict(text=f"Top 5 Stat Profile  |  {season}", font=dict(size=14, color=PALETTE["text"])),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color=PALETTE["text_muted"]),
                                   gridcolor=PALETTE["card_border"]),
                   angularaxis=dict(tickfont=dict(size=11, color=PALETTE["text"]), gridcolor=PALETTE["card_border"])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=11, color=PALETTE["text"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=60, b=40), height=400,
    )
    return fig


@callback(Output("stat-trend", "figure"),
          [Input("award-dropdown", "value"), Input("stat-dropdown", "value")])
def update_stat_trend(award, stat_col):
    if not award or not stat_col or df.empty:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    winners = (
        df[df["AWARD_TYPE"] == award]
        .sort_values("vote_share", ascending=False)
        .groupby("season", sort=False).first().reset_index().sort_values("season")
    )
    if winners.empty or stat_col not in winners.columns:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    fig = go.Figure(go.Scatter(
        x=winners["season"], y=winners[stat_col], mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=7, color=PALETTE["accent_light"], line=dict(color=PALETTE["accent"], width=1.5)),
        text=winners["player_name"], hovertemplate="%{text}<br>%{y:.1f}<extra></extra>",
        fill="tozeroy", fillcolor=PALETTE["accent_glow"],
    ))
    stat_label = STAT_COLS.get(stat_col, stat_col)
    fig.update_layout(
        title=dict(text=f"{award} Winner {stat_label} Over Time", font=dict(size=14, color=PALETTE["text"])),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color=PALETTE["text_muted"]), gridcolor=PALETTE["card_border"]),
        yaxis=dict(title=stat_label, tickfont=dict(size=10, color=PALETTE["text_muted"]), gridcolor=PALETTE["card_border"]),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=60), height=340,
    )
    return fig


@callback(Output("awards-table", "data"),
          [Input("season-dropdown", "value"), Input("award-dropdown", "value")])
def update_table(season, award):
    if not season or not award:
        return []
    return _filtered(season, award).to_dict("records")


# ── Prediction callbacks ──

def _pred_filtered(award: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    return pred_df[pred_df["AWARD_TYPE"] == award].sort_values("predicted_vote_share", ascending=False)


@callback(Output("pred-kpi-row", "children"), Input("pred-award-dropdown", "value"))
def update_pred_kpi(award):
    if not award:
        return []
    sub = _pred_filtered(award)
    if sub.empty:
        return [html.Span("No predictions available.", style={"color": PALETTE["text_muted"]})]
    top = sub.iloc[0]
    return [
        _stat_pill("Candidates", str(len(sub))),
        _stat_pill("Predicted Winner", str(top["player_name"]), PALETTE["prediction_accent"]),
        _stat_pill("Pred. Share", f"{top['predicted_vote_share']:.4f}", PALETTE["positive"]),
        _stat_pill("PPG", f"{top.get('PTS', 0):.1f}", PALETTE["accent_light"]),
        _stat_pill("Team", str(top.get("TEAM_ABBREVIATION", "?")), PALETTE["gold"]),
    ]


@callback(Output("pred-spotlight", "children"), Input("pred-award-dropdown", "value"))
def update_pred_spotlight(award):
    if not award:
        return []
    sub = _pred_filtered(award)
    if sub.empty:
        return html.P("No predictions.", style={"color": PALETTE["text_muted"]})
    top5 = sub.head(5)
    children = [html.H3("Top 5 Predicted", style={"fontSize": "16px", "fontWeight": "600",
                                                    "marginBottom": "20px", "color": PALETTE["text"]})]
    medal_labels = ["1st", "2nd", "3rd", "4th", "5th"]
    for i, (_, row) in enumerate(top5.iterrows()):
        color = RANK_COLORS.get(i + 1, PALETTE["prediction_accent"])
        children.append(
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "14px",
                       "padding": "14px", "borderRadius": "10px", "marginBottom": "10px",
                       "background": f"linear-gradient(135deg, {PALETTE['card']} 0%, {PALETTE['bg']} 100%)",
                       "border": f"1px solid {PALETTE['card_border']}"},
                children=[
                    html.Div(medal_labels[i],
                             style={"background": color, "color": PALETTE["bg"], "fontWeight": "800",
                                    "fontSize": "12px", "borderRadius": "6px",
                                    "padding": "4px 10px", "minWidth": "36px", "textAlign": "center"}),
                    html.Div([
                        html.Div(row["player_name"], style={"fontWeight": "600", "fontSize": "15px"}),
                        html.Div(
                            f"{row.get('TEAM_ABBREVIATION', '?')}  |  "
                            f"pred: {row['predicted_vote_share']:.4f}  |  "
                            f"{row.get('PTS', 0):.1f} PPG  |  "
                            f"age {int(row.get('AGE', 0))}  |  "
                            f"{row.get('MIN', 0):.0f} min"
                            + (f"  |  hype: {int(row['media_hype'])}" if row.get('media_hype', 0) > 0 else ""),
                            style={"fontSize": "12px", "color": PALETTE["text_muted"], "marginTop": "2px"},
                        ),
                    ]),
                ],
            )
        )
    return children


@callback(Output("pred-bar", "figure"), Input("pred-award-dropdown", "value"))
def update_pred_bar(award):
    sub = _pred_filtered(award).head(10) if award else pd.DataFrame()
    if sub.empty:
        return go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    colors = [RANK_COLORS.get(i + 1, PALETTE["prediction_accent"]) for i in range(len(sub))]
    fig = go.Figure(go.Bar(
        x=sub["predicted_vote_share"], y=sub["player_name"], orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=sub["predicted_vote_share"].apply(lambda v: f"{v:.4f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"], size=12),
    ))
    fig.update_layout(
        title=dict(text=f"{award} Predicted Vote Share  |  2025-26", font=dict(size=14, color=PALETTE["text"])),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=PALETTE["text"])),
        xaxis=dict(title="Predicted Vote Share",
                   range=[0, max(0.1, sub["predicted_vote_share"].max() * 1.25)],
                   tickfont=dict(color=PALETTE["text_muted"])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=50, t=50, b=40), height=350,
    )
    return fig


@callback(Output("pred-table", "data"), Input("pred-award-dropdown", "value"))
def update_pred_table(award):
    if not award:
        return []
    return _pred_filtered(award).head(50).to_dict("records")


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
