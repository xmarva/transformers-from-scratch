"""Figures, charts and the cover of the post "How accurate is Polymarket?" on xmarva.github.io.

Every chart is drawn from data/: analysis.json comes from analysis.py (resolved markets from Polymarket's Gamma API
and their price histories from its CLOB API, downloaded by download.py and chosen by markets.py), cases.json from
cases.py and snapshot.json from snapshot.py, taken on 31 October 2026.

Colours keep one meaning throughout: amber is Polymarket's prices, teal what actually happened, rose costs and
losses, violet a correction fitted to the prices, grey baselines and the diagonal of perfect calibration.

    uv run python tools/render.py --article polymarket --out <site>/assets/posts/polymarket-accuracy
"""

import json
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"
HORIZONS = ["1 day", "1 week", "1 month", "3 months", "6 months"]
TOPIC_ORDER = ["politics", "sports", "crypto", "crypto-updown", "economy", "culture", "science", "other"]
TOPIC_NAMES = {"politics": "Politics and world", "sports": "Sports", "crypto": "Crypto prices and events",
               "crypto-updown": "Crypto up or down", "economy": "Economy and markets", "culture": "Culture and media",
               "science": "Science, tech, weather", "other": "Other"}


@cache
def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, y=0.975, gap=0.08):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, y - gap, note, ha="left", va="top", fontsize=10, color=c["muted"])


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def cents(x):
    v = 100 * x
    return f"{v:.1f}¢" if abs(v - round(v)) > 1e-9 else f"{v:.0f}¢"


def band_label(low, high):
    def one(v):
        v = 100 * v
        return f"{v:g}"

    return f"{one(low)}–{one(high)}¢"


def horizon_entry(h):
    return load("analysis")["horizons"][h]


def date_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc)


# ---------------------------------------------------------------- what a price is


def anatomy() -> Figure:
    s = load("snapshot")
    senate = s["events"]["senate"]["markets"][0]
    price, bid, ask = senate["price"], senate["bestBid"], senate["bestAsk"]
    taken = datetime.fromisoformat(s["taken"])
    fig = Figure(
        760,
        282,
        alt=f"A Polymarket share on the US Senate on {taken:%-d %B %Y}: Yes, Democrats control the Senate, at "
        f"{cents(price)}, and No at {cents(1 - price)}; together they make a dollar. The best bid is {cents(bid)} and "
        f"the best ask {cents(ask)}. A Yes share bought at {cents(ask)} pays $1 if Democrats win, a gain of "
        f"{cents(1 - ask)}, and nothing if they lose, a loss of {cents(ask)}. It pays off on average only if the "
        f"chance of a Democratic Senate is above {pct(ask)}",
    )
    fig.text(14, 28, "What a price on Polymarket is", kind="title")
    fig.text(14, 50, f"the Senate market on {taken:%-d %B %Y}, three days before the US midterm elections",
             kind="note")
    x0, x1, y = 40, 720, 86
    split = x0 + price * (x1 - x0)
    fig.rect(x0, y, split - x0 - 2, 46, role="enc", tint=True, r=6)
    fig.rect(split + 2, y, x1 - split - 2, 46, role="dec", tint=True, r=6)
    fig.text(x0 + 14, y + 20, "Yes: Democrats control the Senate", kind="body")
    fig.text(x0 + 14, y + 38, cents(price), kind="mono")
    fig.text(x1 - 14, y + 20, "No", kind="body", anchor="end")
    fig.text(x1 - 14, y + 38, cents(1 - price), kind="mono", anchor="end")
    fig.text(x0, y + 66, "$0", kind="small")
    fig.text(x1, y + 66, "$1", kind="small", anchor="end")
    fig.text((x0 + x1) / 2, y + 66, "one share of each side together always costs about a dollar", kind="small",
             anchor="middle")
    yb = y + 102
    fig.text(x0, yb, "The order book", kind="subtitle")
    fig.text(x0, yb + 24, f"best bid {cents(bid)}: what a buyer offers", kind="body")
    fig.text(x0, yb + 44, f"best ask {cents(ask)}: what a seller asks", kind="body")
    fig.text(x0, yb + 64, f"the price shown, {cents(price)}, is the middle", kind="note")
    xc = 400
    fig.text(xc, yb, f"A Yes share bought at {cents(ask)}", kind="subtitle")
    fig.text(xc, yb + 24, f"Democrats win: pays $1, a gain of {cents(1 - ask)}", kind="body")
    fig.text(xc, yb + 44, f"Democrats lose: pays nothing, a loss of {cents(ask)}", kind="body")
    fig.text(xc, yb + 64, f"worth buying only if their chance is above {pct(ask)}", kind="note")
    return fig


# ---------------------------------------------------------------- how to check a forecaster


def _reliability_axes(ax, c, label=True):
    ax.plot([0, 1], [0, 1], color=c["muted"], lw=1, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.grid(axis="both", color=c["grid"])
    if label:
        ax.set_xlabel("forecast, %")


def _points(ax, bins, color, c, size=5, errors=True, label=None):
    x = np.array([b["mean_price"] for b in bins])
    y = np.array([b["frequency"] for b in bins])
    if errors:
        low = np.array([b["low"] for b in bins])
        high = np.array([b["high"] for b in bins])
        ax.errorbar(x, y, yerr=[np.maximum(y - low, 0), np.maximum(high - y, 0)], fmt="none", ecolor=color,
                    elinewidth=1.2, capsize=0, zorder=2)
    ax.plot(x, y, color=color, lw=1.4, zorder=3)
    ax.plot(x, y, "o", ms=size, color=color, mec=c["card"], mew=1, zorder=4, label=label)


def chart_forecasters(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    f = load("analysis")["forecasters"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.55))
    fig.subplots_adjust(left=0.08, right=0.965, top=0.74, bottom=0.17, wspace=0.3)
    panels = [("market", "Polymarket's prices", c["data"]), ("bold", "a bolder copy", c["dec"]),
              ("timid", "a timid copy", c["att"])]
    for ax, (key, title, color) in zip(axes, panels, strict=True):
        _reliability_axes(ax, c)
        _points(ax, f[key]["calibration"], color, c, size=4, errors=False)
        ax.plot([f["base_rate"]], [f["base_rate"]], "s", ms=7, color=c["muted"], mec=c["card"], mew=1, zorder=5)
        ax.set_title(f"{title}\nBrier {f[key]['brier']:.3f}", fontsize=10, loc="left", color=c["ink"])
    axes[0].set_ylabel("how often it happened, %")
    axes[0].annotate(f"always {pct(f['base_rate'])}:\nBrier {f['brier_base_rate']:.3f}",
                     xy=(f["base_rate"], f["base_rate"]), xytext=(0.36, 0.08), fontsize=8.5, color=c["muted"],
                     arrowprops={"arrowstyle": "-", "color": c["muted"], "lw": 0.8})
    _caption(fig, "Three forecasters on the same markets, a week before each closed",
             "on the diagonal a forecaster is calibrated; the grey square is one who always says the base rate", c,
             gap=0.075)
    return fig


def chart_election(theme):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    e = load("cases")["election"]
    trump = e["markets"][0]["id"]
    series = e["whole"][trump]
    t = [date_of(a) for a, _ in series]
    p = [100 * b for _, b in series]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    ax.plot(t, p, color=c["data"], lw=1.8)
    ax.axhline(50, color=c["muted"], lw=1, ls="--")
    marks = [("2024-06-28", "the first debate ", 97, "right"), ("2024-07-21", " Biden withdraws", 97, "left"),
             ("2024-11-05", "election day ", 90, "right")]
    for day, label, height, side in marks:
        when = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        ax.axvline(when, color=c["grid"], lw=1, zorder=0)
        ax.text(when, height, label, fontsize=8.5, color=c["muted"], va="top", ha=side)
    eve = [b for a, b in series if a <= datetime(2024, 11, 5, 12, tzinfo=timezone.utc).timestamp()][-1]
    ax.annotate(f"{pct(eve)} on the eve", xy=(datetime(2024, 11, 5, tzinfo=timezone.utc), 100 * eve),
                xytext=(datetime(2024, 9, 1, tzinfo=timezone.utc), 30), fontsize=9, color=c["ink"],
                arrowprops={"arrowstyle": "-", "color": c["muted"], "lw": 0.8})
    ax.set_ylim(0, 100)
    ax.set_ylabel("price of Yes, ¢")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _caption(fig, "Will Donald Trump win the 2024 US presidential election?",
             f"Polymarket's price over 2024, every 12 hours; {e['markets'][0]['volume'] / 1e9:.2f} billion dollars "
             "traded in this one market", c)
    return fig


# ---------------------------------------------------------------- the data


TOPIC_COLORS = {"politics": "att", "sports": "enc", "crypto": "data", "crypto-updown": "data", "economy": "dec",
                "culture": "muted", "science": "line", "other": "grid"}


def chart_population(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    pop = load("analysis")["population"]
    years = [y for y in sorted(pop["counts"]) if y >= "2024"]  # the API records few markets before 2024
    total = sum(sum(pop["counts"][y].values()) for y in years)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.1, right=0.66, top=0.8, bottom=0.12)
    bottom = np.zeros(len(years))
    for topic in TOPIC_ORDER:
        values = np.array([pop["counts"][y].get(topic, 0) for y in years], dtype=float) / 1000
        if values.sum() == 0:
            continue
        alpha = 0.45 if topic == "crypto-updown" else 1
        ax.bar(years, values, bottom=bottom, color=c[TOPIC_COLORS[topic]], alpha=alpha, width=0.66,
               label=TOPIC_NAMES[topic], edgecolor=c["card"], linewidth=0.6)
        bottom += values
    for i in range(len(years)):
        ax.text(i, bottom[i] * 1.01 + 1, f"{bottom[i]:,.0f}K" if bottom[i] >= 10 else f"{bottom[i]:.1f}K",
                ha="center", va="bottom", fontsize=8.5, color=c["muted"])
    ax.set_ylabel("resolved markets, thousands")
    ax.set_ylim(0, bottom.max() * 1.14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    _caption(fig, "Resolved markets on Polymarket by the year they closed",
             f"binary markets in events with at least $1,000 traded, {total:,} from 2024 to October 2026; "
             f"{pct(pop['share_yes'])} resolved Yes", c)
    return fig


# ---------------------------------------------------------------- calibration


H_COLORS = {"1 day": "enc", "1 week": "data", "1 month": "att"}


def chart_calibration(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    a = load("analysis")
    w = a["horizons"]["1 week"]
    fig = plt.figure(figsize=(7.6, 4.3))
    left = fig.add_axes([0.075, 0.14, 0.42, 0.62])
    right = fig.add_axes([0.6, 0.14, 0.38, 0.62])
    _reliability_axes(left, c)
    _points(left, w["calibration"], c["data"], c, size=5)
    left.set_ylabel("how often it happened, %")
    shares = np.array(w["sharpness"])
    edges = np.linspace(0, 1, len(shares) + 1)
    right.bar(edges[:-1] * 100, 100 * shares, width=4.4, align="edge", color=c["data"])
    right.set_xlim(0, 100)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_xlabel("price a week before the close, ¢")
    right.set_ylabel("share of markets, %")
    right.set_title("where the forecasts are", fontsize=10, loc="left", color=c["ink"])
    left.set_title("calibration", fontsize=10, loc="left", color=c["ink"])
    _caption(fig, "Polymarket's prices a week before the close, against what happened",
             f"{w['markets']:,} markets with at least ${a['trap']['live_dollars']:,.0f} traded that week; bars are 95% "
             "intervals from a bootstrap over events", c, gap=0.065)
    return fig


def chart_accuracy(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    a = load("analysis")["common"]
    hs = ["1 month", "1 week", "1 day"]
    x = np.arange(len(hs))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.14, wspace=0.3)
    acc = [100 * a[h]["accuracy"] for h in hs]
    naive = [100 * a[h]["always_no_accuracy"] for h in hs]
    left.bar(x - 0.19, acc, width=0.36, color=c["data"], label="the favourite won")
    left.bar(x + 0.19, naive, width=0.36, color=c["muted"], alpha=0.55, label='"No" every time')
    for i in range(len(hs)):
        left.text(x[i] - 0.19, acc[i] + 1.5, f"{acc[i]:.0f}", ha="center", fontsize=8.5, color=c["ink"])
        left.text(x[i] + 0.19, naive[i] + 1.5, f"{naive[i]:.0f}", ha="center", fontsize=8.5, color=c["muted"])
    left.set_ylim(0, 128)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_xticks(x, [h + " before" for h in hs], fontsize=8.5)
    left.set_ylabel("share right, %")
    left.set_title("accuracy", fontsize=10, loc="left", color=c["ink"])
    left.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2, handlelength=1.2, columnspacing=1)
    skill = [100 * a[h]["skill"] for h in hs]
    doubt = [100 * a[h]["in_doubt"]["skill"] for h in hs]
    right.bar(x - 0.19, skill, width=0.36, color=c["data"], label="all markets")
    right.bar(x + 0.19, doubt, width=0.36, color=c["enc"], label="priced 2–98¢ then")
    for i in range(len(hs)):
        right.text(x[i] - 0.19, skill[i] + 1.5, f"{skill[i]:.0f}", ha="center", fontsize=8.5, color=c["ink"])
        right.text(x[i] + 0.19, doubt[i] + 1.5, f"{doubt[i]:.0f}", ha="center", fontsize=8.5, color=c["ink"])
    right.set_ylim(0, 128)
    right.set_yticks([0, 25, 50, 75, 100])
    right.set_xticks(x, [h + " before" for h in hs], fontsize=8.5)
    right.set_ylabel("Brier skill over the base rate, %")
    right.set_title("skill", fontsize=10, loc="left", color=c["ink"])
    right.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2, handlelength=1.2, columnspacing=1)
    n = a["1 day"]["markets"]
    _caption(fig, "Accuracy flatters; skill measures",
             f"the same {n:,} markets at each horizon; skill: the share of the base-rate forecaster's Brier score "
             "removed", c)
    return fig


def chart_convergence(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    cv_ = load("analysis")["convergence"]
    days = np.array(cv_["days"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 4.3), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.12, hspace=0.12)
    top.plot(days, cv_["brier"], color=c["data"], lw=2)
    top.axhline(cv_["uncertainty"], color=c["muted"], lw=1, ls="--")
    top.text(59, cv_["uncertainty"], "always the base rate", fontsize=8.5, color=c["muted"], ha="left", va="bottom")
    top.set_ylim(0, cv_["uncertainty"] * 1.25)
    top.set_ylabel("Brier score")
    bottom.plot(days, 100 * np.array(cv_["decided"]), color=c["enc"], lw=2)
    bottom.set_ylim(0, 100)
    bottom.set_ylabel("priced below 1¢\nor above 99¢, %", fontsize=9)
    bottom.set_xlim(60, 0)
    bottom.set_xlabel("days before the market closed")
    _caption(fig, "How the forecast sharpens as the close approaches",
             f"the same {cv_['markets']:,} markets, each open for 60 days or more and traded a month before the "
             "close; prices every 12 hours", c, gap=0.06)
    return fig


# ---------------------------------------------------------------- longshots and favourites


def chart_returns(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    entry = horizon_entry("1 week")
    rows = entry["returns"]
    net = {(r["low_edge"], r["high_edge"]): r for r in (entry.get("net_returns") or [])}
    cheap = [r for r in rows if r["high_edge"] <= 0.2 + 1e-9]
    dear = [r for r in rows if r["low_edge"] >= 0.8 - 1e-9]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.2, wspace=0.28)
    for ax, group, title in ((axes[0], cheap, "longshots: contracts under 20¢"),
                             (axes[1], dear, "favourites: contracts over 80¢")):
        x = np.arange(len(group))
        mean = np.array([100 * r["return"] for r in group])
        low = np.array([100 * r["low"] for r in group])
        high = np.array([100 * r["high"] for r in group])
        colors = [c["enc"] if m >= 0 else c["dec"] for m in mean]
        ax.bar(x, mean, width=0.6, color=colors, alpha=0.85)
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="none", ecolor=c["ink"], elinewidth=1, capsize=0)
        nets = [net.get((r["low_edge"], r["high_edge"])) for r in group]
        if any(nets):
            xs = [i for i, n in enumerate(nets) if n]
            ys = [100 * nets[i]["net"] for i in xs]
            ax.plot(xs, ys, "D", ms=5, color=c["ink"], mfc=c["card"])
            lo = min(min(ys), low.min())
            hi = max(max(ys), high.max())
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color=c["muted"], lw=1)
        ax.set_xticks(x, [band_label(r["low_edge"], r["high_edge"]) for r in group], fontsize=8.5, rotation=0)
        ax.set_title(title, fontsize=10, loc="left", color=c["ink"])
    axes[0].set_ylabel("average return on the stake, %")
    _caption(fig, "What buying a contract a week before the close returned, by its price",
             "mid prices, bars with 95% intervals over events; diamonds: after paying half the spread", c)
    return fig


def chart_spread(theme):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    c = _style(theme)
    b = load("analysis")["book"]
    rows = b["by_price"]
    x = np.arange(len(rows))
    med = np.array([100 * r["median_cost"] for r in rows])
    q25 = np.array([100 * r["q25"] for r in rows])
    q75 = np.array([100 * r["q75"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.17)
    ax.bar(x, med, width=0.6, color=c["dec"], alpha=0.85)
    ax.errorbar(x, med, yerr=[med - q25, q75 - med], fmt="none", ecolor=c["ink"], elinewidth=1, capsize=0)
    for i, m in enumerate(med):
        ax.text(i, q75[i] * 1.12, f"{m:.1f}%" if m < 10 else f"{m:.0f}%", ha="center", fontsize=8, color=c["ink"])
    ax.set_yscale("log")
    ax.set_ylim(max(0.05, q25.min() * 0.6), q75.max() * 2.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x, [band_label(r["low_edge"], r["high_edge"]) for r in rows], fontsize=8, rotation=0)
    ax.set_xlabel("price of the contract")
    ax.set_ylabel("cost of buying at the ask, %")
    taken = datetime.fromisoformat(b["taken"])
    _caption(fig, "What crossing the spread costs, as a share of the price paid",
             f"half the bid-ask spread of {b['with_two_sided_book']:,} open markets on {taken:%-d %B %Y}; "
             "medians, with quartiles; log scale", c)
    return fig


# ---------------------------------------------------------------- big markets, small markets, topics


VOLUME_BANDS = ["under $1K", "$1K–10K", "$10K–100K", "$100K–1M", "$1M–10M", "$10M+"]
TRADED_BANDS = ["under $100", "$100–1K", "$1K–10K", "$10K–100K", "$100K+"]


def _ratio_bars(ax, groups, order, color, c):
    names = [k for k in order if k in groups]
    x = np.arange(len(names))
    r = np.array([groups[k]["ratio"] for k in names])
    low = np.array([groups[k]["low"] for k in names])
    high = np.array([groups[k]["high"] for k in names])
    ax.bar(x, r, width=0.62, color=color, alpha=0.85)
    ax.errorbar(x, r, yerr=[r - low, high - r], fmt="none", ecolor=c["ink"], elinewidth=1, capsize=0)
    ax.axhline(1, color=c["muted"], lw=1, ls="--")
    ax.set_xticks(x, names, fontsize=8)
    return names


def chart_trap(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    tr = load("analysis")["trap"]["main"]["longshots"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True,
                                      gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.2, wspace=0.08)
    _ratio_bars(left, tr["by_volume"], VOLUME_BANDS, c["dec"], c)
    _ratio_bars(right, tr["by_traded"], TRADED_BANDS, c["enc"], c)
    left.set_title("by the market's total volume,\nknown only after the outcome", fontsize=10, loc="left",
                   color=c["ink"])
    right.set_title("by dollars traded in the week before,\nknown at the time", fontsize=10, loc="left",
                    color=c["ink"])
    left.set_ylabel("how often they won ÷ their price")
    right.set_xlabel("dollars traded that week")
    left.set_xlabel("dollars traded over the market's life")
    top = max(max(g["high"] for g in tr["by_volume"].values()), max(g["high"] for g in tr["by_traded"].values()))
    left.set_ylim(0, top * 1.08)
    _caption(fig, "The same longshots, grouped two ways",
             "contracts priced 2–20¢ a week before the close; at 1 they win exactly as often as priced", c,
             gap=0.07)
    return fig


def chart_topics(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    t = load("analysis")["topics"]
    names = [k for k in TOPIC_ORDER if k in t]
    names.sort(key=lambda k: t[k]["skill"])
    y = np.arange(len(names))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.78, bottom=0.14, wspace=0.12)
    skill = [100 * t[k]["skill"] for k in names]
    ece = [100 * t[k]["ece"] for k in names]
    left.barh(y, skill, height=0.6, color=[c[TOPIC_COLORS[k]] for k in names],
              alpha=0.9)
    for i in range(len(names)):
        left.text(skill[i] + 1, i, f"{skill[i]:.0f}", va="center", fontsize=8.5, color=c["ink"])
    left.set_yticks(y, [f"{TOPIC_NAMES[k]}\n{t[k]['markets']:,} markets" for k in names], fontsize=8.5)
    left.set_xlim(0, 100)
    left.set_xlabel("Brier skill, %")
    left.grid(axis="x", color=c["grid"])
    left.grid(axis="y", visible=False)
    right.barh(y, ece, height=0.6, color=c["muted"], alpha=0.7)
    for i in range(len(names)):
        right.text(ece[i] + 0.1, i, f"{ece[i]:.1f}", va="center", fontsize=8.5, color=c["ink"])
    right.set_xlim(0, max(ece) * 1.3)
    right.set_xlabel("average calibration error, points")
    right.grid(axis="x", color=c["grid"])
    right.grid(axis="y", visible=False)
    _caption(fig, "Topics, a week before the close",
             "skill: higher is better; calibration error: lower is better", c)
    return fig


def chart_crypto(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    cm = load("analysis")["crypto_months"]
    months = [m for m in sorted(cm["months"]) if cm["months"][m]["markets"] >= 10 and m in cm["btc"]]
    x = np.arange(len(months))
    won = np.array([100 * cm["months"][m]["bull_won"] / cm["months"][m]["markets"] for m in months])
    price = np.array([100 * cm["months"][m]["bull_price"] for m in months])
    btc = np.array([100 * cm["btc"][m] for m in months])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.14, hspace=0.12)
    top.bar(x, won, width=0.7, color=c["enc"], label="the bet on rising prices won")
    top.plot(x, price, "_", ms=12, mew=2, color=c["data"], label="its price a week before")
    top.set_ylim(0, 122)
    top.set_yticks([0, 25, 50, 75, 100])
    top.set_ylabel("%")
    top.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    bottom.bar(x, btc, width=0.7, color=[c["enc"] if v >= 0 else c["dec"] for v in btc])
    bottom.axhline(0, color=c["muted"], lw=1)
    bottom.set_ylabel("Bitcoin, %")
    labels = [datetime.strptime(m, "%Y-%m").strftime("%b\n%Y") if m.endswith("-01") or i == 0 else
              datetime.strptime(m, "%Y-%m").strftime("%b") for i, m in enumerate(months)]
    bottom.set_xticks(x, labels, fontsize=8)
    n = sum(cm["months"][m]["markets"] for m in months)
    _caption(fig, "Crypto close calls resolve together",
             f"{n:,} crypto price markets at {100 * cm['low']:.0f}–{100 * cm['high']:.0f}¢ a week out, in months "
             "with ten or more; Bitcoin's return in the same month", c, gap=0.06)
    return fig


# ---------------------------------------------------------------- long-dated markets


def chart_long(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    ld = load("analysis")["long_dated"]
    groups = [("1 month", "1 month"), ("3 months", "3 months"), ("6 months", "6 months")]
    bands = [(0.9, 0.95), (0.95, 0.98), (0.98, 0.995)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.2, wspace=0.3)
    width = 0.26
    colors = [c["data"], c["enc"], c["att"]]
    for j, (low, high) in enumerate(bands):
        xs, price, win, ann = [], [], [], []
        for i, (h, _) in enumerate(groups):
            key = f"{h} {low:.3f}-{high:.3f}"
            if key not in ld:
                continue
            xs.append(i + (j - 1) * width)
            price.append(100 * ld[key]["mean_price"])
            win.append(100 * ld[key]["win_rate"])
            ann.append(100 * ld[key]["annualized"])
        left.scatter(xs, price, marker="_", s=160, color=c["muted"], zorder=3)
        left.scatter(xs, win, marker="o", s=26, color=colors[j], zorder=4, label=band_label(low, high))
        for x, p, w in zip(xs, price, win, strict=True):
            left.plot([x, x], [p, w], color=colors[j], lw=1.2)
        right.bar(xs, ann, width=width * 0.9, color=colors[j], label=band_label(low, high))
    left.set_xticks(range(len(groups)), [g[0] + "\nbefore" for g in groups], fontsize=8.5)
    left.set_ylabel("price (dash) and win rate (dot), %")

    left.set_title("favourites win about as often as priced", fontsize=10, loc="left", color=c["ink"])
    right.axhline(0, color=c["muted"], lw=1)
    right.set_xticks(range(len(groups)), [g[0] + "\nbefore" for g in groups], fontsize=8.5)
    right.set_ylabel("return per year, %")
    right.set_title("what holding them to the end paid", fontsize=10, loc="left", color=c["ink"])
    right.legend(loc="upper right", frameon=False, fontsize=8, title="price band", title_fontsize=8)
    _caption(fig, "Buying the favourite months before the close",
             "contracts priced above 90¢ at the time, both sides of every market, mid prices", c)
    return fig


# ---------------------------------------------------------------- a correction to the prices


def chart_recalibration(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    r = load("analysis")["recalibration"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.09, right=0.6, top=0.8, bottom=0.14)
    ax.plot([0, 1], [0, 1], color=c["muted"], lw=1, ls="--")
    for h, color in (("1 week", c["att"]), ("1 month", c["data"])):
        ax.plot(r[h]["curve"]["price"], r[h]["curve"]["corrected"], color=color, lw=2, label=f"{h} before")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.grid(axis="both", color=c["grid"])
    ax.set_xlabel("Polymarket's price, %")
    ax.set_ylabel("corrected probability, %")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    lines = []
    for h in ("1 week", "1 month"):
        x = r[h]
        gain = x["brier_market"] - x["brier_corrected"]
        lines.append(f"{h} before: a = {x['a']:.2f}, b = {x['b']:+.2f}")
        lines.append(f"Brier {x['brier_market']:.4f} → {x['brier_corrected']:.4f}")
        lines.append(f"gain {gain:+.4f}, 95%: {x['gain_low']:+.4f} to {x['gain_high']:+.4f}")
        lines.append("")
    fig.text(0.64, 0.74, "\n".join(lines), fontsize=8.6, color=c["ink"], va="top", family="monospace",
             linespacing=1.5)
    split = r["1 week"]["split"]
    _caption(fig, "A two-number correction, fitted on the past and tested on the future",
             f"fitted on markets that closed before {datetime.fromisoformat(split):%B %Y}, scored on those after", c)
    return fig


# ---------------------------------------------------------------- the Nobel Peace Prize


NOBEL_NAMES = {"Sudan's Emergency Response Rooms": "Emergency Response Rooms"}


def chart_nobel(theme):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    n = load("cases")["nobel"]
    announced = datetime.fromisoformat(n["announced"].replace("Z", "+00:00"))
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.09, right=0.7, top=0.78, bottom=0.15)
    start = datetime(2025, 10, 8, 12, tzinfo=timezone.utc)
    last = {}
    for m in n["markets"]:
        series = [(date_of(t), 100 * p) for t, p in n["prices"][m["id"]] if date_of(t) >= start]
        if not series:
            continue
        won = m["final"][0] in ("1", 1)
        t, p = zip(*series, strict=True)
        ax.plot(t, p, color=c["data"] if won else c["muted"], lw=2.2 if won else 1.1, alpha=1 if won else 0.75)
        before = [v for when, v in series if when < announced]
        last[m["outcome"].strip()] = (before[-1], won)
    shown = sorted(last.items(), key=lambda kv: -kv[1][0])[:3]
    floor = None
    for name, (value, won) in shown:  # labels from the top down, at least 7 points apart
        y = value if floor is None else min(value, floor - 7)
        floor = y
        ax.annotate(f"{NOBEL_NAMES.get(name, name)}, {value:.0f}¢", xy=(announced, value),
                    xytext=(announced + timedelta(minutes=40), y),
                    textcoords="data", fontsize=8.5, va="center", ha="left", color=c["ink"] if won else c["muted"])
    ax.axvline(announced, color=c["dec"], lw=1.2, ls="--")
    ax.text(announced, 96, "announcement, 09:00 ", fontsize=8.5, color=c["dec"], ha="right", va="top")
    ax.set_ylim(0, 100)
    ax.set_ylabel("price, ¢")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d Oct\n%H:%M"))
    ax.set_xlim(start, announced)
    _caption(fig, "Nobel Peace Prize 2025: the night before the announcement",
             "Polymarket's prices every 10 minutes up to the announcement, UTC; the winner in amber", c)
    return fig


CHARTS = {
    "forecasters": chart_forecasters,
    "election": chart_election,
    "population": chart_population,
    "calibration": chart_calibration,
    "accuracy": chart_accuracy,
    "convergence": chart_convergence,
    "returns": chart_returns,
    "spread": chart_spread,
    "trap": chart_trap,
    "topics": chart_topics,
    "crypto": chart_crypto,
    "long": chart_long,
    "recalibration": chart_recalibration,
    "nobel": chart_nobel,
}


def chart_alts():
    a = load("analysis")
    h = a["horizons"]
    alts = {}
    f = a["forecasters"]
    alts["forecasters"] = (
        "Three reliability diagrams for the same markets a week before they closed. Polymarket's own prices lie close "
        f"to the diagonal, Brier score {f['market']['brier']:.3f}. A bolder copy, with the log-odds doubled, bends "
        f"away from it, Brier {f['bold']['brier']:.3f}; a timid copy, with them halved, bends the other way, Brier "
        f"{f['timid']['brier']:.3f}. A forecaster who always says the base rate, {pct(f['base_rate'])}, is a single "
        f"point on the diagonal, Brier {f['brier_base_rate']:.3f}"
    )
    e = load("cases")["election"]
    alts["election"] = (
        "Polymarket's price for Donald Trump winning the 2024 US presidential election over 2024: near 50 cents in "
        "spring, up after the first debate, down to about even after Biden withdrew and Harris entered, rising again "
        f"in October; {e['markets'][0]['volume'] / 1e9:.2f} billion dollars traded"
    )
    pop = a["population"]
    years = [y for y in sorted(pop["counts"]) if y >= "2024"]
    alts["population"] = (
        "Stacked bars of resolved Polymarket markets by the year they closed and topic: "
        + ", ".join(f"{y}: {sum(pop['counts'][y].values()):,}" for y in years)
        + f". {pct(pop['share_yes'])} of all of them resolved Yes"
    )
    w = h["1 week"]
    alts["calibration"] = (
        "Reliability diagram: markets grouped by their price a week before closing, against the share that resolved "
        "Yes: " + "; ".join(f"priced {pct(b['mean_price'])}, happened {pct(b['frequency'])}" for b in w["calibration"])
        + ". Next to it, a histogram of those prices"
    )
    cm = a["common"]
    order = ["1 month", "1 week", "1 day"]
    alts["accuracy"] = (
        f"The same {cm['1 day']['markets']:,} markets a month, a week and a day before they closed. Left: share where "
        "the side priced above 50 cents won, against the share a forecaster who always says No gets right: "
        + "; ".join(f"{k} before {pct(cm[k]['accuracy'])} against {pct(cm[k]['always_no_accuracy'])}" for k in order)
        + ". Right: Brier skill over the base rate: " + "; ".join(
            f"{k} {pct(cm[k]['skill'])} for all, {pct(cm[k]['in_doubt']['skill'])} for those priced 2 to 98 cents"
            for k in order)
    )
    cvg = a["convergence"]
    alts["convergence"] = (
        f"Brier score of {cvg['markets']:,} markets, each open for at least 60 days, by days before the close: "
        f"{cvg['brier'][-1]:.3f} at 60 days, {cvg['brier'][59]:.3f} at 30, {cvg['brier'][13]:.3f} at 7 and "
        f"{cvg['brier'][0]:.3f} at half a day, against {cvg['uncertainty']:.3f} for always saying the base rate. "
        f"The share priced below 1 cent or above 99 rises from {pct(cvg['decided'][-1])} to {pct(cvg['decided'][0])}"
    )
    alts["returns"] = (
        "Average return of buying a contract a week before the close, by price band: " + "; ".join(
            f"{band_label(r['low_edge'], r['high_edge'])} {100 * r['return']:+.1f}%" for r in w["returns"])
    )
    b = a["book"]
    alts["spread"] = (
        "Median cost of buying at the ask instead of the middle of the book, as a share of the price, by contract "
        "price: " + "; ".join(f"{band_label(r['low_edge'], r['high_edge'])} {100 * r['median_cost']:.1f}%"
                              for r in b["by_price"])
    )
    tr = a["trap"]["main"]["longshots"]
    alts["trap"] = (
        "Contracts priced 2 to 20 cents a week before the close: how often they won divided by their price, grouped by "
        "the market's total volume and by the dollars traded in it in the week before. By total volume: " + "; ".join(
            f"{k} {tr['by_volume'][k]['ratio']:.2f}" for k in VOLUME_BANDS if k in tr["by_volume"])
        + ". By the week before: " + "; ".join(
            f"{k} {tr['by_traded'][k]['ratio']:.2f}" for k in TRADED_BANDS if k in tr["by_traded"])
    )
    t = a["topics"]
    alts["topics"] = (
        "Brier skill and calibration error a week before the close, by topic: " + "; ".join(
            f"{TOPIC_NAMES[k]} skill {pct(t[k]['skill'])}, error {100 * t[k]['ece']:.1f} points"
            for k in TOPIC_ORDER if k in t)
    )
    cm = a["crypto_months"]
    shown = [m for m in sorted(cm["months"]) if cm["months"][m]["markets"] >= 10 and m in cm["btc"]]
    alts["crypto"] = (
        "Two panels by month. Top: among crypto price markets priced 30 to 70 cents a week before the close, the "
        "share where the side betting on rising prices won, against that side's average price. Bottom: Bitcoin's "
        "return that month. The two move together: " + "; ".join(
            f"{m}: {100 * cm['months'][m]['bull_won'] / cm['months'][m]['markets']:.0f}% of "
            f"{cm['months'][m]['markets']}, Bitcoin {100 * cm['btc'][m]:+.0f}%" for m in shown)
    )
    ld = a["long_dated"]
    alts["long"] = (
        "Contracts priced above 90 cents one, three and six months before the close: their price, how often they won "
        "and what holding them to the end returned per year: " + "; ".join(
            f"{x['horizon']} {band_label(x['low_edge'], x['high_edge'])}: price {pct(x['mean_price'], 1)}, won "
            f"{pct(x['win_rate'], 1)}, {100 * x['annualized']:+.1f}% a year" for x in ld.values())
    )
    r = a["recalibration"]
    alts["recalibration"] = (
        "The fitted correction from Polymarket's price to a probability, against the diagonal, for one week and one "
        "month before the close: " + "; ".join(
            f"{k}: slope {r[k]['a']:.2f}, Brier on later markets {r[k]['brier_market']:.4f} before and "
            f"{r[k]['brier_corrected']:.4f} after" for k in r)
    )
    alts["nobel"] = (
        "Prices of the leading candidates for the 2025 Nobel Peace Prize on Polymarket from midday on 8 October to the "
        "announcement on 10 October. María Corina Machado's price is near the bottom until late on 9 October, then "
        "rises within hours to about two thirds of a dollar before the announcement at 09:00 UTC"
    )
    return alts


# ---------------------------------------------------------------- cover


def cover():
    a = load("analysis")["horizons"]["1 week"]["calibration"]
    x0, x1, top, bottom = cv.MOTIF_X0 + 40, cv.MOTIF_X1 - 10, 150, 500
    size = min(x1 - x0, bottom - top)
    x1 = x0 + size

    def X(v):
        return x0 + v * size

    def Y(v):
        return bottom - v * size

    out = [cv.label(cv.MOTIF_X0, 118, "PRICE AGAINST WHAT HAPPENED")]
    out.append(f'<rect x="{x0}" y="{bottom - size}" width="{size}" height="{size}" fill="none" stroke="{cv.LINE}" '
               'stroke-width="1"/>')
    for k in (0.25, 0.5, 0.75):
        out.append(f'<line x1="{X(k):.1f}" y1="{Y(0):.1f}" x2="{X(k):.1f}" y2="{Y(1):.1f}" stroke="{cv.LINE}" '
                   'stroke-width="0.6" stroke-dasharray="3 5"/>')
        out.append(f'<line x1="{X(0):.1f}" y1="{Y(k):.1f}" x2="{X(1):.1f}" y2="{Y(k):.1f}" stroke="{cv.LINE}" '
                   'stroke-width="0.6" stroke-dasharray="3 5"/>')
    out.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" stroke="{cv.CHIP_INK}" '
               'stroke-width="1.4" stroke-dasharray="6 6"/>')
    pts = " ".join(f"{X(b['mean_price']):.1f},{Y(b['frequency']):.1f}" for b in a)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{cv.RED}" stroke-width="2.6" stroke-linejoin="round"/>')
    for b in a:
        out.append(f'<line x1="{X(b["mean_price"]):.1f}" y1="{Y(b["low"]):.1f}" x2="{X(b["mean_price"]):.1f}" '
                   f'y2="{Y(b["high"]):.1f}" stroke="{cv.RED}" stroke-width="1.6"/>')
        out.append(f'<circle cx="{X(b["mean_price"]):.1f}" cy="{Y(b["frequency"]):.1f}" r="5" fill="{cv.RED}"/>')
    out.append(cv.text(X(0), bottom + 30, "price", 16, cv.MUTED, cv.MONO))
    out.append(cv.text(X(1), bottom + 30, "100¢", 16, cv.MUTED, cv.MONO, anchor="end"))
    out.append(cv.text(X(0) - 12, Y(1) + 14, "100%", 16, cv.MUTED, cv.MONO, anchor="end"))
    out.append(cv.text(X(0) - 12, Y(0), "0", 16, cv.MUTED, cv.MONO, anchor="end"))
    cv.fits(x1)
    hook = ["Prices of resolved markets", "against what happened"]
    return cv.frame("blog / finance", ["How accurate", "is Polymarket?"], hook, "".join(out),
                    footer="Calibration, biases and costs")


COVER = cover

FIGURES = {
    "anatomy": anatomy,
}
