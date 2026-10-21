"""Dark fintech look: colours, CSS, and small reusable UI pieces (hero, KPI cards, insights)."""
import streamlit as st

# ---- colour palette (one place, so every page matches) ----
BG, CARD, BORDER = "#0B1020", "#101830", "#1E2A47"
TEXT, MUTED = "#E8EEF9", "#8FA0C2"
TEAL, VIOLET, AMBER, ROSE, GREEN, BLUE = "#2DD4BF", "#8B5CF6", "#F59E0B", "#FB7185", "#34D399", "#38BDF8"
STATUS_COLORS = {"No default": TEAL, "Default": ROSE}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp {{ font-family: 'Inter', 'Segoe UI', sans-serif; color: {TEXT};
  background: radial-gradient(1100px 560px at 8% -8%, rgba(45,212,191,0.10), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,0.11), transparent 55%), {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: #0D1428; border-right: 1px solid {BORDER}; }}

.brand {{ display:flex; align-items:center; gap:12px; padding: 6px 4px 18px 4px; }}
.brand .logo {{ width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#04110F; background: linear-gradient(135deg, {TEAL}, {BLUE}); box-shadow: 0 6px 20px rgba(45,212,191,.35); }}
.brand .name {{ font-weight:800; font-size:1.02rem; line-height:1.1; }}
.brand .tag {{ color:{MUTED}; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; }}
.side-note {{ margin-top:18px; padding:12px 14px; border-radius:12px; border:1px solid rgba(245,158,11,.35);
  background: rgba(245,158,11,.07); color:#F6C875; font-size:.78rem; line-height:1.45; }}

.hero {{ padding: 28px 32px; border-radius: 20px; border: 1px solid {BORDER}; margin-bottom: 22px;
  background: linear-gradient(135deg, rgba(45,212,191,0.15), rgba(139,92,246,0.15)), {CARD}; }}
.hero h1 {{ font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; padding: 0;
  background: linear-gradient(90deg, #FFFFFF, #7DF3E1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.hero p {{ color: #A7B6D6; margin: 0; font-size: 1rem; max-width: 820px; }}
.badge {{ display:inline-block; padding:4px 11px; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.07em;
  text-transform:uppercase; border:1px solid rgba(45,212,191,.45); color:#5EEAD4; background:rgba(45,212,191,.09); margin-bottom:12px; }}

.kpi {{ background: linear-gradient(180deg, #141D38, #0F1730); border:1px solid {BORDER}; border-radius:16px;
  padding:18px 20px 16px 20px; position:relative; overflow:hidden; transition: transform .15s ease, border-color .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
.kpi::before {{ content:''; position:absolute; left:0; top:0; height:3px; width:100%; background: var(--accent); }}
.kpi .label {{ font-size:.72rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }}
.kpi .value {{ font-size:1.85rem; font-weight:800; color:#F4F7FF; margin-top:6px; letter-spacing:-0.02em; line-height:1.15; }}
.kpi .sub {{ font-size:.8rem; color:#7C8DB0; margin-top:3px; }}
.kpi.mini {{ padding:12px 14px; }} .kpi.mini .value {{ font-size:1.25rem; margin-top:3px; }}

.section-title {{ font-size:1.15rem; font-weight:700; margin: 30px 0 2px 0; color:{TEXT}; }}
.section-caption {{ color:{MUTED}; font-size:.88rem; margin-bottom:12px; }}
.insight {{ border:1px solid {BORDER}; border-left:4px solid var(--accent); background:#111A33; border-radius:12px;
  padding:13px 16px; margin-bottom:10px; color:#D5DEF0; font-size:.93rem; line-height:1.5; }}
.insight b {{ color:#FFFFFF; }}
.disclaimer {{ margin-top:26px; padding:14px 18px; border-radius:12px; border:1px solid rgba(245,158,11,.35);
  background: rgba(245,158,11,.07); color:#F6C875; font-size:.85rem; line-height:1.5; }}

[data-testid="stPlotlyChart"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:6px; }}
[data-baseweb="tab-list"] {{ gap: 6px; }}
[data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding: 8px 16px; }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{ border-radius:10px; padding:8px 12px; margin-bottom:2px; }}
.pill {{ display:inline-block; padding:7px 16px; border-radius:999px; font-weight:700; font-size:.85rem; letter-spacing:.02em;
  color:var(--accent); border:1px solid var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }}
.flow {{ display:flex; flex-wrap:wrap; gap:10px; margin: 6px 0 4px 0; }}
.flow .step {{ flex:1 1 150px; border:1px solid {BORDER}; border-radius:14px; background:#111A33; padding:14px 14px 12px 14px;
  border-top:3px solid var(--accent); }}
.flow .n {{ color:var(--accent); font-weight:800; font-size:.72rem; letter-spacing:.12em; }}
.flow .t {{ font-weight:700; margin:3px 0 4px 0; color:#F4F7FF; }}
.flow .d {{ color:{MUTED}; font-size:.8rem; line-height:1.4; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:14px; background:{CARD}; }}
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo">LR</div><div><div class="name">Loan Risk<br>Intelligence</div>'
            '<div class="tag">ML prototype</div></div></div>', unsafe_allow_html=True)


def sidebar_note():
    st.sidebar.markdown(
        '<div class="side-note"><b>Prototype only.</b> Scores are model estimates, not lending decisions.</div>',
        unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = "Machine Learning Prototype"):
    st.markdown(f'<div class="hero"><span class="badge">{badge}</span><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def kpi_html(label, value, sub="", accent=TEAL, mini=False) -> str:
    cls = "kpi mini" if mini else "kpi"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    size = "font-size:1.3rem;" if len(str(value)) > 11 and not mini else ""   # long text values stay on one line
    return (f'<div class="{cls}" style="--accent:{accent}"><div class="label">{label}</div>'
            f'<div class="value" style="{size}">{value}</div>{sub_html}</div>')


def kpi_row(items, mini=False, weights=None):
    """items: list of (label, value, sub, accent) tuples shown side by side.
    weights: optional relative widths, e.g. [1.7, 1, 1] makes the first card wider."""
    cols = st.columns(weights or len(items))
    for col, (label, value, sub, accent) in zip(cols, items):
        col.markdown(kpi_html(label, value, sub, accent, mini), unsafe_allow_html=True)


def insight(text_html: str, accent=TEAL):
    st.markdown(f'<div class="insight" style="--accent:{accent}">{text_html}</div>', unsafe_allow_html=True)


def disclaimer(text=None):
    text = text or ("<b>Responsible use:</b> this is a learning prototype trained on a public dataset that looks "
                    "synthetic. Predicted probabilities are estimates, not guaranteed outcomes, and must not be "
                    "used to approve or reject real loan applications.")
    st.markdown(f'<div class="disclaimer">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=360, legend=True):
    """Apply the dark look to any Plotly figure."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#C9D4EA", size=12),
        margin=dict(l=16, r=16, t=48, b=16), showlegend=legend,
        title=dict(font=dict(size=15, color=TEXT), x=0.02),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="#16213F", font_color="#FFFFFF", bordercolor=BORDER),
        colorway=[TEAL, VIOLET, AMBER, ROSE, BLUE, GREEN],
    )
    fig.update_xaxes(gridcolor="#1A2540", zerolinecolor="#1A2540", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#1A2540", zerolinecolor="#1A2540", linecolor=BORDER)
    return fig


def pill(text: str, accent=TEAL) -> str:
    return f'<span class="pill" style="--accent:{accent}">{text}</span>'


def flow(steps):
    """steps: list of (title, description, accent). Shows a numbered left-to-right process."""
    html = "".join(f'<div class="step" style="--accent:{a}"><div class="n">STEP {i}</div><div class="t">{t}</div>'
                   f'<div class="d">{d}</div></div>' for i, (t, d, a) in enumerate(steps, 1))
    st.markdown(f'<div class="flow">{html}</div>', unsafe_allow_html=True)
