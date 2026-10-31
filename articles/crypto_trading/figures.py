"""Figures, charts and the cover of the post "AI and LLMs in crypto trading" on xmarva.github.io.

Every chart is drawn from the measurements in data/: market.json, direction.json, volatility.json,
overfitting.json and funding.json come from the scripts of the same names, run locally on Binance's public data
(prepare.py downloads it); depth.json is a set of order-book snapshots taken by depth.py on 31 October 2026;
arena.json is Alpha Arena's season 1, read by arena.py from the Internet Archive's copy of Nof1's API; memory.json
and foundation_raw.json come from memory.py and foundation.py, run on Kaggle's T4 GPUs by kaggle.py (memory.json
from two runs merged by `memory.py merge`), and foundation.json from `foundation.py summarize`.

Colours keep one meaning throughout: amber is the market itself, teal a model's forecast or a strategy, rose
costs and losses, violet language models, grey baselines and theory. Where coins are compared side by side,
Bitcoin is amber, Ether teal and Solana grey.

    uv run python tools/render.py --article crypto_trading --out <site>/assets/posts/ai-crypto-trading
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"
COIN_NAMES = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL"}
COIN_COLORS = {"BTCUSDT": "data", "ETHUSDT": "enc", "SOLUSDT": "muted"}


@cache
def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, y=0.975, gap=0.08):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, y - gap, note, ha="left", va="top", fontsize=10, color=c["muted"])


def years_between(first, last):
    from datetime import date

    a, b = date.fromisoformat(first[:10]), date.fromisoformat(last[:10])
    return (b - a).days / 365.25


# ---------------------------------------------------------------- what the market looks like to a model


def chart_tails(theme):
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    c = _style(theme)
    m = load("market")
    btc = m["coins"]["BTCUSDT"]
    edges = np.array(btc["histogram"]["edges"])
    counts = np.array(btc["histogram"]["counts"], dtype=float)
    width = edges[1] - edges[0]
    density = counts / counts.sum() / width
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.15)
    shown = np.where(density > 0, density, np.nan)
    ax.stairs(np.nan_to_num(shown, nan=1e-9), edges, fill=True, color=c["data"], alpha=0.85, label="BTC, hourly")
    z = np.linspace(-20, 20, 1601)
    ax.plot(z, norm.pdf(z), color=c["muted"], lw=1.6, ls="--", label="normal law, same mean and spread")
    ax.set_yscale("log")
    ax.set_ylim(3e-6, 1.5)
    ax.set_xlim(-20, 20)
    ax.set_xticks(range(-20, 21, 4))
    ax.set_xlabel("hourly return, in standard deviations")
    ax.set_ylabel("density, log scale")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    hours = btc["hours"]
    years = 1 / btc["tails"]["normal_beyond_6"] / (24 * 365.25)
    ax.text(-19.3, 6e-2, f"{btc['tails']['beyond_6'] * hours:.0f} hours moved more than 6 sd.\n"
            f"A normal law expects one such hour\nin about {round(years, -3):,.0f} years.", fontsize=9,
            color=c["ink"], va="center")
    _caption(fig, "Bitcoin's hourly returns and a normal law",
             f"{hours:,} hours from January 2021 to October 2026, Binance spot", c)
    return fig


def chart_acf(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    m = load("market")
    btc = m["coins"]["BTCUSDT"]
    lags = np.arange(1, len(btc["acf_returns"]) + 1)
    band = 2 / np.sqrt(btc["hours"])
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.16)
    ax.axhspan(-band, band, color=c["grid"], lw=0)
    ax.plot(lags, btc["acf_absolute"], color=c["enc"], lw=2, label="absolute returns: the size of a move")
    ax.bar(lags, btc["acf_returns"], width=0.8, color=c["data"], label="returns")
    ax.axhline(0, color=c["muted"], lw=0.8)
    for day in (24, 48, 72, 96, 120, 144, 168):
        ax.axvline(day, color=c["grid"], lw=0.8, zorder=0)
    ax.set_xlim(0, 169)
    ax.set_ylim(-0.03, 0.34)
    ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("lag, hours")
    ax.set_ylabel("autocorrelation")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _caption(fig, "Autocorrelation of Bitcoin's hourly returns and of their size",
             "a lag of 24 means this hour against the same hour a day earlier; the grey band is noise", c)
    return fig


HORIZON_ORDER = ["5 min", "15 min", "1 hour", "4 hours", "1 day"]
COST_STYLE = {"2": ("muted", "2 bp: maker at a high-volume tier"), "4": ("enc", "4 bp: maker fees on futures"),
              "10": ("dec", "10 bp: taker fees on futures"), "20": ("att", "20 bp: taker fees on spot")}


def chart_breakeven(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    m = load("market")
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.2)
    x = np.arange(len(HORIZON_ORDER))
    btc = m["coins"]["BTCUSDT"]["moves"]
    for cost, (key, label) in COST_STYLE.items():
        y = np.array([100 * btc[h]["breakeven_hit_rate"][cost] for h in HORIZON_ORDER])
        possible = y <= 100
        ax.plot(x[possible], y[possible], color=c[key], lw=2, marker="o", ms=5, label=label)
        if not possible.all():
            first = int(np.argmax(possible))
            ax.annotate("", xy=(x[first] - 0.45, 100), xytext=(x[first], y[first]),
                        arrowprops={"arrowstyle": "-", "color": c[key], "lw": 2, "ls": (0, (2, 2))})
    ax.axhline(50, color=c["muted"], lw=1, ls="--")
    ax.text(-0.2, 51.0, "a coin toss", color=c["muted"], fontsize=9, ha="left")
    ax.set_xticks(x, [f"{h}\n{btc[h]['mean_abs_bps']:.0f} bp" for h in HORIZON_ORDER], fontsize=9)
    ax.set_xlim(-0.25, len(HORIZON_ORDER) - 0.75)
    ax.set_ylim(48, 101)
    ax.set_ylabel("hit rate needed, %")
    ax.set_xlabel("holding period, and Bitcoin's average move over it")
    ax.legend(loc="upper right", frameon=False, fontsize=9, title="cost of a round trip", title_fontsize=9)
    _caption(fig, "The hit rate a strategy needs to pay for its trades",
             "a strategy that opens and closes a position on every bar; 1 bp (basis point) is 0.01%", c)
    return fig


# ---------------------------------------------------------------- predicting the direction


def walk_forward() -> Figure:
    d = load("direction")
    months = len(d["coins"]["BTCUSDT"]["monthly_hit_rate"])
    fig = Figure(
        760,
        336,
        alt=f"Walk-forward testing: for each of {months} months from January 2022 to October 2026, a new model is "
        f"trained on all hours before the month, less a gap of a day, and predicts that month only. The months it "
        f"predicts are joined into one out-of-sample record",
    )
    fig.text(14, 28, "Walk-forward test", kind="title")
    fig.text(14, 50, f"a new model every month, {months} in all, each trained only on hours before its month",
             kind="note")
    x0, x1 = 110, 730
    first, last = 2020.0, 2026 + 10 / 12

    def x(t):
        return x0 + (t - first) / (last - first) * (x1 - x0)

    rows = [("Jan 2022", 2022.0), ("Feb 2022", 2022 + 1 / 12), ("Mar 2022", 2022 + 2 / 12),
            ("Jul 2024", 2024.5), ("Oct 2026", 2026 + 9 / 12)]
    y = 78
    for i, (name, start) in enumerate(rows):
        if i == 3:
            fig.text(x0 - 12, y + 10, "...", kind="label", anchor="end")
            y += 26
        fig.text(x0 - 12, y + 17, name, kind="mono", anchor="end")
        fig.rect(x(first), y + 4, x(start) - x(first) - 4, 18, role="enc", tint=True, r=3)
        fig.rect(x(start), y + 4, x(start + 1 / 12) - x(start), 18, role="dec", r=3)
        y += 32
    for year in range(2020, 2027):
        fig.line((x(year), 76), (x(year), y), role="muted", width=0.6, dash=True)
        fig.text(x(year), y + 18, str(year), kind="small", anchor="middle")
    ly = y + 46
    fig.rect(x0, ly - 11, 26, 14, role="enc", tint=True, r=3)
    fig.text(x0 + 34, ly, "training data: all hours before the month, less a day", kind="note")
    fig.rect(x0 + 430, ly - 11, 26, 14, role="dec", r=3)
    fig.text(x0 + 464, ly, "the month it predicts", kind="note")
    return fig


def chart_reversal(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    m = load("market")
    years = list(m["coins"]["BTCUSDT"]["reversal_by_year"])
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.12)
    x = np.arange(len(years))
    for i, (symbol, key) in enumerate(COIN_COLORS.items()):
        values = [100 * m["coins"][symbol]["reversal_by_year"][y]["same_sign"] for y in years]
        # bars hang from 50%, the level of a series without memory
        ax.bar(x + (i - 1) * 0.26, [v - 50 for v in values], bottom=50, width=0.24, color=c[key],
               label=COIN_NAMES[symbol])
    ax.axhline(50, color=c["muted"], lw=1)
    labels = [y if y != "2026" else "2026\nto Oct" for y in years]
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(44.5, 50.9)
    ax.set_ylabel("same direction, % of hours")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncols=3)
    _caption(fig, "How often an hour moves the same way as the hour before",
             "50% would mean no memory; below 50%, a move tends to be partly taken back", c)
    return fig


def chart_growth(theme):
    from datetime import date

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = load("direction")
    btc = d["coins"]["BTCUSDT"]["curves"]
    days = [date.fromisoformat(x) for x in btc["dates"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.1, right=0.8, top=0.78, bottom=0.12)
    lines = [("fee_0", "enc", "-", "the model,\nno fees"), ("buy_and_hold", "data", "-", "holding\nBitcoin"),
             ("fee_2", "dec", "--", "the model,\n2 bp a side"), ("fee_5", "dec", "-", "the model,\n5 bp a side")]
    ends = {}
    for key, color, style, _label in lines:
        wealth = np.exp(btc[key])
        ax.plot(days, wealth, color=c[color], lw=1.8, ls=style)
        ends[key] = np.log10(wealth[-1])
    # labels at the right end of each line, pushed apart where lines end close together
    placed = []
    for key, color, _dash, label in sorted(lines, key=lambda line: -ends[line[0]]):
        y = ends[key]
        if placed and placed[-1] - y < 1.1:
            y = placed[-1] - 1.1
        placed.append(y)
        ax.text(days[-1], 10**y, "  " + label, color=c[color], fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.axhline(1, color=c["muted"], lw=0.8)
    ax.set_ylabel("value of 1 dollar, log scale")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _caption(fig, "Trading the model's forecasts for Bitcoin, hour by hour",
             "long when the next hour is more likely up, short otherwise; fees are paid on every change of side", c)
    return fig


def chart_fees(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = load("direction")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.15, wspace=0.08)
    styles = {"0": ("enc", "no fees"), "1": ("muted", "1 bp a side"), "2": ("att", "2 bp"), "5": ("dec", "5 bp")}
    for ax, symbol in zip(axes, ("BTCUSDT", "SOLUSDT"), strict=True):
        strategies = d["coins"][symbol]["strategies"]
        changes = [s["position_changes_per_day"] for s in strategies.values()]
        for fee, (key, label) in styles.items():
            values = [100 * s["fees"][fee]["annual_return"] for s in strategies.values()]
            ax.plot(changes, values, color=c[key], lw=2, marker="o", ms=4, label=label)
        ax.axhline(0, color=c["muted"], lw=0.8)
        ax.set_title(COIN_NAMES[symbol], fontsize=10.5, color=c["ink"])
        ax.set_xlabel("changes of side per day")
        ax.set_xlim(0, 17)
    axes[0].set_ylim(-260, 150)
    axes[0].set_ylabel("return per year, %")
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)
    _caption(fig, "What the fees leave of the edge",
             "the same forecasts traded less often: the position changes only on stronger signals", c)
    return fig


# ---------------------------------------------------------------- language models as traders

ARENA_ORDER = ["qwen3-max", "deepseek-chat-v3.1", "claude-sonnet-4-5", "grok-4", "gemini-2.5-pro", "gpt-5"]


def arena_noise_sigma(a):
    """Median hourly volatility of the six models: the spread of a zero-edge trader with the same risk."""
    values = sorted(a["models"][m]["hourly_volatility"] for m in ARENA_ORDER)
    return (values[2] + values[3]) / 2


def chart_arena(theme):
    from datetime import datetime

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    a = load("arena")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.subplots_adjust(left=0.135, right=0.78, top=0.8, bottom=0.12)
    first = datetime.fromisoformat(a["models"]["gpt-5"]["times"][0])
    sigma = arena_noise_sigma(a)
    times = [datetime.fromisoformat(t) for t in a["models"]["gpt-5"]["times"]]
    hours = np.array([(t - first).total_seconds() / 3600 for t in times])
    for k, alpha in ((2, 0.35), (1, 0.6)):
        ax.fill_between(times, 1e4 * np.exp(-k * sigma * np.sqrt(hours)), 1e4 * np.exp(k * sigma * np.sqrt(hours)),
                        color=c["grid"], alpha=alpha, lw=0)
    ends = {}
    for i, model in enumerate(ARENA_ORDER + ["buynhold_btc"]):
        m = a["models"][model]
        t = [datetime.fromisoformat(x) for x in m["times"]]
        color = c["data"] if model == "buynhold_btc" else c["att"]
        width = 2.2 if model == "buynhold_btc" else 1.4
        alpha = 1.0 if model == "buynhold_btc" else 1.0 - 0.1 * i
        ax.plot(t, m["equity"], color=color, lw=width, alpha=alpha)
        ends[model] = (t[-1], m["final_equity"], color, alpha)
    placed = []
    for model in sorted(ends, key=lambda k: -ends[k][1]):
        t, value, color, alpha = ends[model]
        y = np.log10(value)
        if placed and placed[-1] - y < 0.055:
            y = placed[-1] - 0.055
        placed.append(y)
        m = a["models"][model]
        change = "0%" if abs(m["return"]) < 0.005 else f"{100 * m['return']:+.0f}%"
        ax.text(t, 10**y, f"  {m['name'].replace(', bought and held', ' held')}  {change}",
                color=color, fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_yticks([2500, 5000, 10000, 20000, 40000], ["$2,500", "$5,000", "$10,000", "$20,000", "$40,000"])
    ax.minorticks_off()
    ax.set_ylim(2200, 50000)
    ax.set_ylabel("account value, log scale")
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(18, 22, 26, 30, 3)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    _caption(fig, "Alpha Arena, season 1: six LLMs trading crypto perpetuals with real money",
             "Nof1's hourly records; grey bands: where a trader with no skill and the same risk lands, 1 and 2 sd",
             c)
    return fig


MEMORY_NAMES = {
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    "Qwen/Qwen3-8B": "Qwen3 8B", "allenai/Olmo-3-7B-Instruct": "Olmo 3 7B", "Qwen/Qwen3.5-9B": "Qwen3.5 9B",
}


@cache
def memory_items():
    return json.loads((DATA / "memory_items.json").read_text())


def memory_models():
    """The models that answered, oldest release first."""
    m = load("memory")["models"]
    ok = [repo for repo in m if "error" not in m[repo]]
    return sorted(ok, key=lambda repo: m[repo]["released"])


def price_errors(repo):
    """(date, relative error of the model's answer) for every price question; a missing number counts as 100%."""
    answers = load("memory")["models"][repo]["prices"]
    out = []
    for item, answer in zip(memory_items()["prices"], answers, strict=True):
        value = answer["value"]
        error = 1.0 if value is None else min(abs(value / item["close"] - 1), 1.0)
        out.append((item["date"], error))
    return out


def direction_calls(repo, kind):
    """(date, went up, model says up) for the monthly or weekly questions. The model's own median probability is the
    threshold, so that a model that always leans one way is still judged on how it orders the dates."""
    answers = load("memory")["models"][repo][kind]
    p = np.array([a["p_up"] for a in answers])
    threshold = float(np.median(p))
    out = []
    for item, prob in zip(memory_items()[kind], p, strict=True):
        date = item.get("month", item.get("start"))
        out.append((date, bool(item["up"]), bool(prob > threshold), item["return"]))
    return out


def by_half_year(rows, stat=np.mean):
    """Average of the values per half-year: {"2018 H1": value, ...}."""
    groups = {}
    for date, value in rows:
        key = f"{date[:4]} H{1 if int(date[5:7]) <= 6 else 2}"
        groups.setdefault(key, []).append(value)
    return {k: float(stat(v)) for k, v in sorted(groups.items())}


def chart_memory_prices(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = _style(theme)
    models = memory_models()
    m = load("memory")["models"]
    halves = list(by_half_year([(it["date"], 0) for it in memory_items()["prices"]]))
    grid = np.array([[by_half_year(price_errors(r), np.median)[h] for h in halves] for r in models])
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(models) + 1.9))
    fig.subplots_adjust(left=0.2, right=0.9, top=1 - 0.95 / (0.42 * len(models) + 1.9), bottom=0.2)
    cmap = LinearSegmentedColormap.from_list("error", [c["card"], c["att"]])
    image = ax.imshow(100 * grid, aspect="auto", cmap=cmap, vmin=0, vmax=50)
    ax.set_yticks(range(len(models)), [f"{MEMORY_NAMES[r]}" for r in models], fontsize=9)
    ax.set_xticks(range(len(halves))[::2], [h[:4] for h in halves][::2], fontsize=9)
    ax.grid(False)
    for row, repo in enumerate(models):
        released = m[repo]["released"]
        key = f"{released[:4]} H{1 if int(released[5:7]) <= 6 else 2}"
        if key in halves:
            col = halves.index(key)
            ax.plot([col - 0.5, col - 0.5], [row - 0.45, row + 0.45], color=c["dec"], lw=2.2)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("median error, %", fontsize=9, color=c["muted"])
    bar.outline.set_visible(False)
    _caption(fig, "How far each model's answer is from Bitcoin's real closing price",
             "two questions a month, 2018 to October 2026, by half-year; red mark: the model's release", c,
             gap=0.3 / (0.42 * len(models) + 1.9) * 2.4)
    return fig


MEMORY_PANELS = ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Instruct",
                 "Qwen/Qwen3.5-9B"]


def chart_memory_answers(theme):
    from datetime import date

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    m = load("memory")["models"]
    panels = [r for r in MEMORY_PANELS if r in m and "error" not in m[r]]
    items = memory_items()["prices"]
    days = [date.fromisoformat(it["date"]) for it in items]
    truth = [it["close"] for it in items]
    rows = (len(panels) + 1) // 2
    height = 2.3 * rows + 1.2
    fig, axes = plt.subplots(rows, 2, figsize=(7.6, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=1 - 1.05 / height, bottom=0.3 / height, hspace=0.32, wspace=0.06)
    for ax, repo in zip(axes.flat, panels, strict=False):
        ax.plot(days, truth, color=c["data"], lw=1.6)
        values = [a["value"] for a in m[repo]["prices"]]
        shown = [(d, v) for d, v in zip(days, values, strict=True) if v is not None and 100 < v < 1e6]
        ax.scatter([d for d, _ in shown], [v for _, v in shown], s=9, color=c["att"], lw=0, zorder=3)
        released = date.fromisoformat(m[repo]["released"])
        ax.axvline(released, color=c["dec"], lw=1.2, ls="--")
        ax.set_title(f"{MEMORY_NAMES[repo]}, released {released:%b %Y}", fontsize=9.5, color=c["ink"])
        ax.set_yscale("log")
        ax.set_ylim(1000, 200000)
        ax.set_yticks([1000, 10000, 100000], ["$1K", "$10K", "$100K"])
        ax.minorticks_off()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    _caption(fig, "What Bitcoin closed at, according to four open models",
             "amber: the real price; violet: the model's answer for the 1st and 15th of each month; red: release",
             c, y=1 - 0.1 / height, gap=0.32 / height)
    return fig


def months_between(a, b):
    return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])


def memory_accuracy(repo, kind="months"):
    """Share of right up-or-down answers for periods that ended at least a year before the model's release, and for
    periods after it."""
    released = load("memory")["models"][repo]["released"]
    early, late = [], []
    for date, up, says_up, _ in direction_calls(repo, kind):
        if months_between(date, released) >= 12:
            early.append(up == says_up)
        elif date > released:
            late.append(up == says_up)
    return float(np.mean(early)), float(np.mean(late)), len(early), len(late)


def chart_memory_direction(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    models = memory_models()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.2)
    x = np.arange(len(models))
    early = [100 * memory_accuracy(r)[0] for r in models]
    late = [100 * memory_accuracy(r)[1] for r in models]
    ax.bar(x - 0.2, [v - 50 for v in early], bottom=50, width=0.38, color=c["att"],
           label="months that ended a year or more before the model's release")
    ax.bar(x + 0.2, [v - 50 for v in late], bottom=50, width=0.38, color=c["muted"],
           label="months after its release")
    ax.axhline(50, color=c["muted"], lw=1)
    ax.set_xticks(x, [MEMORY_NAMES[r].replace(" ", "\n", 1) for r in models], fontsize=8.5)
    ax.set_ylim(25, 78)
    ax.set_ylabel("right about up or down, %")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    _caption(fig, "Did Bitcoin go up or down that month? The models remember",
             "each model's answers split at its own median, so that a model that always says up is judged fairly", c)
    return fig


def chart_memory_strategy(theme):
    from datetime import date

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    repo = MEMORY_STRATEGY_MODEL
    calls = direction_calls(repo, "months")
    months = [date.fromisoformat(d + "-01") for d, *_ in calls]
    hold = np.cumsum([np.log1p(r) for *_, r in calls])
    traded = np.cumsum([np.log1p(r) if says_up else -np.log1p(r) for _, _, says_up, r in calls])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.12)
    ax.plot(months, np.exp(traded), color=c["att"], lw=2, label=f"trading on {MEMORY_NAMES[repo]}'s answers")
    ax.plot(months, np.exp(hold), color=c["data"], lw=1.6, label="holding Bitcoin")
    released = date.fromisoformat(load("memory")["models"][repo]["released"])
    ax.axvline(released, color=c["dec"], lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.text(released, 0.6, "  model released", color=c["dec"], fontsize=9, va="bottom")
    ax.set_ylabel("value of 1 dollar, log scale")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _caption(fig, "A backtest of an LLM trader that has read the future",
             "each month: long Bitcoin if the model says the month went up, short if it says down", c)
    return fig


MEMORY_STRATEGY_MODEL = "allenai/OLMo-2-1124-7B-Instruct"  # the longest stretch of months after its release


def lightgbm_hit_rate(symbol, year="2026"):
    """The walk-forward model's hit rate over the months of a year, each month weighted by its hours."""
    from calendar import monthrange

    d = load("direction")
    monthly = d["coins"][symbol]["monthly_hit_rate"]
    last_day = int(d["end"][8:10])
    total, hours = 0.0, 0
    for month, rate in monthly.items():
        if not month.startswith(year):
            continue
        y, m = int(month[:4]), int(month[5:])
        days = last_day if month == d["end"][:7] else monthrange(y, m)[1]
        total += rate * days * 24
        hours += days * 24
    return total / hours


def chart_foundation(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    f = load("foundation")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.12, wspace=0.3)
    names = [COIN_NAMES[s] for s in COIN_NAMES]
    x = np.arange(len(names))
    rows = [("LightGBM, 20 features", "enc", 1.0, [lightgbm_hit_rate(s) for s in COIN_NAMES]),
            ("Chronos-2", "att", 1.0, [f["models"]["chronos"]["coins"][s]["hit_rate"] for s in COIN_NAMES]),
            ("Kronos-small", "att", 0.45, [f["models"]["kronos"]["coins"][s]["hit_rate"] for s in COIN_NAMES])]
    for i, (label, key, alpha, values) in enumerate(rows):
        # bars start at a coin toss, so that they show how far each model is from it
        ax.bar(x + (i - 1) * 0.26, [100 * v - 50 for v in values], bottom=50, width=0.24, color=c[key], alpha=alpha,
               label=label)
    ax.axhline(50, color=c["muted"], lw=1)
    ax.set_xticks(x, names)
    ax.set_ylim(47, 55.5)
    ax.set_ylabel("next hour's direction right, %")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("direction of the next hour", fontsize=10, color=c["muted"], fontweight="normal")
    chronos = [f["models"]["chronos"]["coins"][s]["r2_log_rv"] for s in COIN_NAMES]
    har = [f["models"]["chronos"]["coins"][s]["r2_log_rv_har"] for s in COIN_NAMES]
    bx.bar(x - 0.2, har, width=0.38, color=c["enc"], label="HAR")
    bx.bar(x + 0.2, chronos, width=0.38, color=c["att"], label="Chronos-2")
    bx.set_xticks(x, names)
    bx.set_ylim(0, 1.05)
    bx.set_ylabel("R²")
    bx.legend(loc="upper left", frameon=False, fontsize=8.5, ncols=2)
    bx.set_title("tomorrow's volatility", fontsize=10, color=c["muted"], fontweight="normal")
    _caption(fig, "Foundation models for time series, zero-shot, on 2026",
             "every hour and every day from January to October 2026; Kronos on every fourth hour", c)
    return fig


# ---------------------------------------------------------------- agents


def agent() -> Figure:
    fig = Figure(
        760,
        472,
        alt="Parts of an LLM trading agent. Inputs: prices and order books, news and social posts, on-chain data, and "
        "the agent's memory of earlier sessions. All of them reach the language model as text, and the model calls "
        "tools and plugins. It proposes an order, which passes through risk checks written in ordinary code "
        "(position and loss limits, allowed "
        "markets) before it reaches the exchange with an API key that can trade but not withdraw. Attack points: "
        "instructions hidden in a post, planted memories, a malicious plugin or tool, and a leaked key",
    )
    fig.text(14, 28, "Parts of an LLM trading agent, and where it is attacked", kind="title")
    fig.text(14, 50, "everything the model reads is text, and any text can carry instructions", kind="note")
    inputs = [("prices, order books", "data"), ("news, social posts", "data"), ("on-chain data", "data"),
              ("memory of past sessions", "data")]
    y = 82
    for label, role in inputs:
        fig.box(20, y, 196, 40, role=role, label=label)
        fig.arrow((218, y + 20), (282, 216), role="muted", width=1.2)
        y += 58
    fig.box(286, 176, 150, 80, role="att", label="language model", strong=True)
    fig.text(361, 162, "proposes an order", kind="small", anchor="middle")
    fig.box(286, 290, 150, 40, role="att", label="tools, plugins")
    fig.arrow((350, 258), (350, 288), role="att", width=1.2)
    fig.arrow((372, 288), (372, 258), role="att", width=1.2)
    fig.arrow((438, 216), (478, 216), role="att")
    fig.box(482, 168, 132, 96, role="enc")
    fig.text(548, 192, "risk checks", kind="block", role="enc", anchor="middle")
    fig.text(548, 214, "plain code:", kind="small", anchor="middle")
    fig.text(548, 230, "size, loss limits,", kind="small", anchor="middle")
    fig.text(548, 246, "allowed markets", kind="small", anchor="middle")
    fig.arrow((616, 216), (650, 216), role="enc")
    fig.box(654, 176, 90, 80, role="neutral")
    fig.text(699, 210, "exchange", kind="body", anchor="middle")
    fig.text(699, 230, "trade-only", kind="small", anchor="middle")
    fig.text(699, 245, "API key", kind="small", anchor="middle")
    # attack points
    legend = [(20, 362, "1", "instructions hidden in a post or a web page"),
              (20, 394, "2", "planted memories that the agent trusts later"),
              (400, 362, "3", "a malicious plugin, tool or server"),
              (400, 394, "4", "a key that can withdraw, or a leaked key")]
    for x, y, number, text in legend:
        fig.chip(x, y, number, role="dec", strong=True, h=24, size=12)
        fig.text(x + 32, y + 17, text, kind="body")
    fig.chip(236, 154, "1", role="dec", strong=True, h=24, size=12)
    fig.chip(236, 270, "2", role="dec", strong=True, h=24, size=12)
    fig.chip(446, 298, "3", role="dec", strong=True, h=24, size=12)
    fig.chip(712, 140, "4", role="dec", strong=True, h=24, size=12)
    fig.line((20, 432), (740, 432), role="muted", width=0.8)
    fig.text(20, 456, "the model proposes orders; limits are enforced by code outside it, which it cannot change",
             kind="note")
    return fig


def chart_depth(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = load("depth")
    sizes = d["sizes"]
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    fig.subplots_adjust(left=0.1, right=0.64, top=0.8, bottom=0.14)
    contracts = d["contracts"]
    shades = np.linspace(1.0, 0.35, len(contracts))
    ends = []
    for contract, shade in zip(contracts, shades, strict=True):
        xs = [size for size in sizes if contract["costs_bps"][str(size)] is not None]
        ys = [contract["costs_bps"][str(size)] for size in xs]
        ax.plot(xs, ys, color=c["dec"], alpha=shade, lw=1.8, marker="o", ms=4)
        ends.append((np.log10(ys[-1]), xs[-1], ys[-1], contract, shade))
    # a column of labels right of the chart, spread apart, each joined to the end of its line
    label_x, placed = sizes[-1] * 2.6, []
    for y, x_end, y_end, contract, shade in sorted(ends, key=lambda e: -e[0]):
        if placed and placed[-1] - y < 0.24:
            y = placed[-1] - 0.24
        placed.append(y)
        ax.plot([x_end * 1.08, label_x / 1.12], [y_end, 10**y], color=c["dec"], alpha=0.5 * shade, lw=0.8)
        name = contract["symbol"].removesuffix("USDT")
        thin = "" if x_end == sizes[-1] else ", too thin for $1M"
        ax.text(label_x, 10**y, f"#{contract['rank']} {name}{thin}", color=c["dec"], alpha=max(shade, 0.65),
                fontsize=8.5, va="center")
    ax.axhline(5, color=c["muted"], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, ["$1K", "$10K", "$100K", "$1M"])
    ax.set_xlim(sizes[0] / 1.5, sizes[-1] * 2.4)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("size of a market buy")
    ax.set_ylabel("cost against the mid price, bp")
    from datetime import datetime

    day = datetime.fromisoformat(d["snapshots"][0]).strftime("%-d %B %Y")
    _caption(fig, "What a market order costs, by size and by how much a contract trades",
             f"Binance perpetuals ranked by 24-hour turnover, median of {len(d['snapshots'])} snapshots, {day}; "
             f"dashed: a 5 bp taker fee", c)
    return fig


# ---------------------------------------------------------------- overfitting


def chart_overfitting(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    o = load("overfitting")
    ins = np.array([p["in_sample"] for p in o["points"]])
    oos = np.array([p["out_of_sample"] for p in o["points"]])
    top = {(t["coin"], t["rule"]) for t in o["top"][:20]}
    is_top = np.array([(p["coin"], p["rule"]) in top for p in o["points"]])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)
    ax.scatter(ins[~is_top], oos[~is_top], s=9, color=c["enc"], alpha=0.45, lw=0)
    ax.scatter(ins[is_top], oos[is_top], s=26, color=c["dec"], lw=0, label="the 20 best in 2021-2023")
    low, high = ins.min() - 0.1, ins.max() + 0.1
    ax.plot([low, high], [low, high], color=c["muted"], lw=1, ls="--")
    ax.text(low + 0.05, low - 0.35, "on this line: as good later as before", color=c["muted"], fontsize=8.5,
            ha="left", va="top")
    ax.axhline(0, color=c["muted"], lw=0.8)
    ax.axvline(0, color=c["muted"], lw=0.8)
    ax.set_xlabel("Sharpe ratio, 2021-2023 (in sample)")
    ax.set_ylabel("Sharpe ratio, 2024-2026")
    ax.grid(True, axis="both")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _caption(fig, f"{o['strategies']:,} textbook strategies, before and after",
             f"{o['rules']} rules with different parameters on {len(o['coins'])} coins, hourly, after "
             f"{o['fee_bps']} bp per side", c)
    return fig


def chart_noise(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    n = load("overfitting")["noise"]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.15)
    ax.plot(n["counts"], n["best_theory"], color=c["muted"], lw=1.8, ls="--",
            label="expected maximum, Bailey and Lopez de Prado")
    ax.plot(n["counts"], n["best_empirical"], color=c["dec"], lw=0, marker="o", ms=6,
            label="best of N random strategies")
    ax.axhline(0, color=c["muted"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of strategies tried, N")
    ax.set_ylabel("best Sharpe ratio")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _caption(fig, "The best of many strategies that are pure noise",
             "random long and short positions on Bitcoin, 2021-2023, hourly, after 5 bp per side", c)
    return fig


# ---------------------------------------------------------------- funding


def chart_funding(theme):
    from datetime import date

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    f = load("funding")
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.12)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        monthly = f["coins"][symbol]["monthly_annualized"]
        months = [date.fromisoformat(k + "-15") for k in monthly]
        ax.plot(months, [100 * v for v in monthly.values()], color=c[COIN_COLORS[symbol]], lw=1.8,
                label=COIN_NAMES[symbol])
    ax.axhline(0, color=c["muted"], lw=0.8)
    ax.axhline(100 * 0.0001 * 3 * 365, color=c["muted"], lw=1, ls="--")
    ax.text(date(2025, 3, 1), 100 * 0.0001 * 3 * 365 + 1.5, "default: 0.01% per 8 hours", fontsize=8.5,
            color=c["muted"])
    ax.set_ylabel("funding, % a year")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _caption(fig, "What the cash-and-carry trade collected",
             "funding of Binance's perpetual futures, summed by month and scaled to a year", c)
    return fig


# ---------------------------------------------------------------- volatility


def chart_volatility(theme):
    from datetime import date

    import matplotlib.pyplot as plt

    c = _style(theme)
    v = load("volatility")
    btc = v["coins"]["BTCUSDT"]["scatter"]
    days = [date.fromisoformat(d) for d in btc["dates"]]
    first = days.index(date(2025, 7, 1))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [2.1, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.17, wspace=0.28)
    ax.plot(days[first:], btc["realized"][first:], color=c["data"], lw=1.2, label="realized")
    ax.plot(days[first:], btc["har"][first:], color=c["enc"], lw=1.8, label="forecast a day earlier")
    ax.set_ylabel("Bitcoin's volatility, % a year")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, 230)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelsize=9)
    names = [COIN_NAMES[s] for s in v["coins"]]
    x = np.arange(len(names))
    vol = [v["coins"][s]["r2_har"] for s in v["coins"]]
    ret = [v["coins"][s]["r2_return"] for s in v["coins"]]
    bx.bar(x - 0.2, vol, width=0.38, color=c["enc"], label="volatility")
    bx.bar(x + 0.2, ret, width=0.38, color=c["data"], label="return")
    for i in range(len(names)):
        bx.text(x[i] - 0.2, vol[i] + 0.02, f"{vol[i]:.2f}", ha="center", fontsize=8.5, color=c["ink"])
        bx.text(x[i] + 0.2, 0.02, f"{ret[i]:.3f}", ha="center", fontsize=8, color=c["ink"], rotation=90,
                va="bottom")
    bx.axhline(0, color=c["muted"], lw=0.8)
    bx.set_xticks(x, names)
    bx.set_ylim(-0.05, 0.8)
    bx.set_title("out-of-sample R², 2022-2026", fontsize=10, color=c["muted"], fontweight="normal")
    bx.legend(loc="upper left", frameon=False, fontsize=8.5)
    _caption(fig, "Forecasts of tomorrow's volatility and of tomorrow's return",
             "a linear HAR model on 5-minute data, refitted monthly; the same regression on past returns", c)
    return fig


def chart_targeting(theme):
    from datetime import date

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c = _style(theme)
    v = load("volatility")
    sol = v["coins"]["SOLUSDT"]["targeting"]
    days = [date.fromisoformat(d) for d in sol["rolling_dates"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    ax.plot(days, sol["rolling_vol_hold"], color=c["data"], lw=1.3, label="holding SOL")
    ax.plot(days, sol["rolling_vol_targeted"], color=c["enc"], lw=1.6,
            label="position sized by the forecast")
    ax.axhline(100 * sol["target"], color=c["muted"], lw=1, ls="--")
    ax.set_ylabel("volatility over 30 days, % a year")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _caption(fig, "Sizing a position by forecast volatility",
             f"Solana, target {100 * sol['target']:.0f}% a year, leverage at most {sol['max_leverage']:.0f}", c)
    return fig


CHARTS = {
    "memory-answers": chart_memory_answers,
    "memory-direction": chart_memory_direction,
    "memory-strategy": chart_memory_strategy,
    "arena": chart_arena,
    "foundation": chart_foundation,
    "depth": chart_depth,
    "overfitting": chart_overfitting,
    "noise": chart_noise,
    "funding": chart_funding,
    "reversal": chart_reversal,
    "growth": chart_growth,
    "fees": chart_fees,
    "tails": chart_tails,
    "acf": chart_acf,
    "breakeven": chart_breakeven,
    "volatility": chart_volatility,
    "targeting": chart_targeting,
}


def chart_alts():
    m = load("market")["coins"]["BTCUSDT"]
    v = load("volatility")["coins"]
    alts = {}
    alts["tails"] = (
        f"Histogram of Bitcoin's hourly returns in standard deviations on a log scale, against a normal law. "
        f"{m['tails']['beyond_6'] * m['hours']:.0f} of {m['hours']:,} hours moved more than six standard "
        f"deviations; a normal law expects {m['tails']['normal_beyond_6'] * m['hours']:.1g}"
    )
    alts["acf"] = (
        f"Autocorrelation of Bitcoin's hourly returns stays within noise at every lag up to a week (lag 1: "
        f"{m['acf_returns'][0]:.3f}), while that of absolute returns starts at {m['acf_absolute'][0]:.2f} and is "
        f"still {m['acf_absolute'][167]:.2f} after a week, with bumps at whole days"
    )
    rows = [f"{h}: average move {m['moves'][h]['mean_abs_bps']:.0f} bp, hit rate needed "
            f"{100 * m['moves'][h]['breakeven_hit_rate']['10']:.0f}% at 10 bp" for h in HORIZON_ORDER]
    alts["breakeven"] = "The hit rate a strategy trading every bar needs to cover its costs. " + "; ".join(rows)
    alts["volatility"] = (
        "Left: Bitcoin's realized daily volatility since July 2025 and the HAR forecast made a day earlier, which "
        "follows it closely. Right: out-of-sample R squared, volatility against return: " + "; ".join(
            f"{COIN_NAMES[s]} {v[s]['r2_har']:.2f} against {v[s]['r2_return']:.3f}" for s in v)
    )
    sol = v["SOLUSDT"]["targeting"]
    alts["targeting"] = (
        f"Volatility over 30 days of holding Solana, swinging from about 40 to over 150 percent a year, and of a "
        f"position sized by the forecast, which stays near the target of {100 * sol['target']:.0f} percent. "
        f"Realized volatility {100 * sol['hold']['volatility']:.0f}% against "
        f"{100 * sol['targeted']['volatility']:.0f}%, largest drawdown {100 * sol['hold']['max_drawdown']:.0f}% "
        f"against {100 * sol['targeted']['max_drawdown']:.0f}%"
    )
    a = load("arena")
    parts = [f"{a['models'][m]['name']} {100 * a['models'][m]['return']:+.0f}%" for m in ARENA_ORDER]
    alts["arena"] = (
        "Account values of six language models trading crypto perpetual futures with $10,000 each in Alpha Arena, "
        "season 1, 18 October to 3 November 2025, on a log scale, with Bitcoin held as a reference "
        f"({100 * a['models']['buynhold_btc']['return']:+.2f}%). Final returns: " + ", ".join(parts) + ". Grey "
        "bands show where a trader with no skill and the same volatility would end up; every model ends inside them"
    )
    d = load("direction")["coins"]
    m_all = load("market")["coins"]
    alts["reversal"] = (
        "Share of hours that move in the same direction as the previous hour, by year, for BTC, ETH and SOL. Every "
        "bar is below 50%, lowest in 2022 and 2023: " + "; ".join(
            f"{COIN_NAMES[s]} " + ", ".join(f"{y} {100 * v['same_sign']:.1f}%" for y, v in m_all[s]["reversal_by_year"]
                                            .items()) for s in COIN_NAMES)
    )
    btc = d["BTCUSDT"]["curves"]
    alts["growth"] = (
        f"Value of one dollar traded on the model's hourly forecasts for Bitcoin from January 2022 to October 2026, "
        f"log scale: without fees it ends at {np.exp(btc['fee_0'][-1]):.1f} dollars, holding Bitcoin at "
        f"{np.exp(btc['buy_and_hold'][-1]):.2f}, with 2 bp per side at {np.exp(btc['fee_2'][-1]):.3f}, with 5 bp per "
        f"side at practically zero"
    )
    rows = []
    for symbol in ("BTCUSDT", "SOLUSDT"):
        for st in d[symbol]["strategies"].values():
            gross, net = st["fees"]["0"]["annual_return"], st["fees"]["5"]["annual_return"]
            rows.append(f"{COIN_NAMES[symbol]} {st['position_changes_per_day']:.1f} changes a day: no fees "
                        f"{100 * gross:.0f}%, 5 bp {100 * net:.0f}%")
    alts["fees"] = ("Return per year against how often the strategy changes side, for four fee levels. "
                    + "; ".join(rows))
    dp = load("depth")
    alts["depth"] = (
        "Cost of a market buy against the mid price on Binance perpetual futures, by order size from $1,000 to $1 "
        "million, for contracts ranked by daily turnover, log scales. " + "; ".join(
            f"#{ct['rank']} {ct['symbol'].removesuffix('USDT')}: $100K costs {ct['costs_bps']['100000']:.2f} bp"
            for ct in dp["contracts"])
    )
    o = load("overfitting")
    alts["overfitting"] = (
        f"Scatter of {o['strategies']} strategies, Sharpe ratio in 2021-2023 against 2024-2026. The cloud is flatter "
        f"than the diagonal; the 20 best in the first period, highlighted, average "
        f"{np.mean([t['in_sample'] for t in o['top'][:20]]):.2f} then and "
        f"{np.mean([t['out_of_sample'] for t in o['top'][:20]]):.2f} later. Rank correlation "
        f"{o['rank_correlation']:.2f}"
    )
    n = o["noise"]
    alts["noise"] = (
        "Best Sharpe ratio among N random strategies against N on a log scale, with the expected maximum of Bailey "
        "and Lopez de Prado. "
        + "; ".join(f"N = {k}: {v:.2f}" for k, v in zip(n["counts"], n["best_empirical"], strict=True))
    )
    fu = load("funding")["coins"]
    alts["funding"] = (
        "Funding of Binance's BTC and ETH perpetual futures by month, scaled to a year, 2020 to October 2026. "
        "Yearly averages for BTC: " + ", ".join(f"{y} {100 * v['annualized']:.1f}%" for y, v in
                                              fu["BTCUSDT"]["yearly"].items())
    )
    mm = load("memory")["models"]
    panels = []
    for repo in MEMORY_PANELS:
        errors = price_errors(repo)
        early = [e for d, e in errors if months_between(d, mm[repo]["released"]) >= 12]
        panels.append(f"{MEMORY_NAMES[repo]}, released {mm[repo]['released'][:7]}: median error "
                      f"{100 * np.median(early):.0f}% on dates a year or more before its release")
    alts["memory-answers"] = (
        "Bitcoin's real closing price from 2018 to 2026 and four open models' answers for the 1st and 15th of each "
        "month. Each model's answers follow the price closely up to some point before its release and then stop, "
        "turn into refusals, zeros or a constant guess. " + "; ".join(panels)
    )
    rows = []
    for repo in memory_models():
        early, late, n_early, n_late = memory_accuracy(repo)
        rows.append(f"{MEMORY_NAMES[repo]} {100 * early:.0f}% of {n_early} months before, {100 * late:.0f}% of "
                    f"{n_late} after")
    alts["memory-direction"] = (
        "Share of months for which each model says correctly whether Bitcoin went up or down, for months that ended "
        "a year or more before the model's release and for months after it: " + "; ".join(rows)
    )
    calls = direction_calls(MEMORY_STRATEGY_MODEL, "months")
    released = mm[MEMORY_STRATEGY_MODEL]["released"][:7]
    before = sum(np.log1p(r) * (1 if up else -1) for d, _, up, r in calls if d < released)
    after = sum(np.log1p(r) * (1 if up else -1) for d, _, up, r in calls if d >= released)
    held = sum(np.log1p(r) for d, *_, r in calls if d < released)
    alts["memory-strategy"] = (
        f"A dollar traded monthly on {MEMORY_NAMES[MEMORY_STRATEGY_MODEL]}'s answers grows to "
        f"{np.exp(before):.0f} dollars before the model's release in {released}, against {np.exp(held):.0f} for "
        f"holding Bitcoin, and then goes to {np.exp(before + after):.0f}, a change of {100 * (np.exp(after) - 1):.0f}% "
        f"after the release"
    )
    fo = load("foundation")["models"]
    alts["foundation"] = (
        "Left: share of hours of 2026 where the direction of the next hour was right, for LightGBM, Chronos-2 and "
        "Kronos-small. " + "; ".join(
            f"{COIN_NAMES[s]} {100 * lightgbm_hit_rate(s):.1f}%, {100 * fo['chronos']['coins'][s]['hit_rate']:.1f}%, "
            f"{100 * fo['kronos']['coins'][s]['hit_rate']:.1f}%" for s in COIN_NAMES)
        + ". Right: R squared of tomorrow's log volatility, HAR against Chronos-2: " + "; ".join(
            f"{COIN_NAMES[s]} {fo['chronos']['coins'][s]['r2_log_rv_har']:.2f} against "
            f"{fo['chronos']['coins'][s]['r2_log_rv']:.2f}" for s in COIN_NAMES)
    )
    return alts


# ---------------------------------------------------------------- cover


def cover():
    a = load("arena")
    x0, x1, top, bottom = cv.MOTIF_X0 + 4, cv.MOTIF_X1, 150, 470
    low, high = np.log(2500), np.log(40000)

    def y(value):
        return bottom - (np.log(value) - low) / (high - low) * (bottom - top)

    hours = len(a["models"]["gpt-5"]["equity"])

    def x(i):
        return x0 + i / (hours - 1) * (x1 - x0)

    out = [cv.label(cv.MOTIF_X0, 118, "SIX LLMS, $10,000 EACH, 17 DAYS")]
    sigma = arena_noise_sigma(a)
    upper = [(x(i), y(1e4 * np.exp(2 * sigma * np.sqrt(i)))) for i in range(hours)]
    lower = [(x(i), y(1e4 * np.exp(-2 * sigma * np.sqrt(i)))) for i in reversed(range(hours))]
    points = " ".join(f"{px:.1f},{max(py, top - 20):.1f}" for px, py in upper + lower)
    out.append(f'<polygon points="{points}" fill="{cv.CHIP_FILL}" stroke="{cv.LINE}" stroke-width="1" '
               'stroke-dasharray="4 4"/>')
    out.append(f'<line x1="{x0}" y1="{y(1e4):.1f}" x2="{x1}" y2="{y(1e4):.1f}" stroke="{cv.LINE}" stroke-width="1"/>')
    for i, model in enumerate(ARENA_ORDER):
        equity = a["models"][model]["equity"]
        path = " ".join(f"{x(j):.1f},{y(v):.1f}" for j, v in enumerate(equity))
        out.append(f'<polyline points="{path}" fill="none" stroke="{cv.CHIP_INK}" stroke-opacity="{0.9 - 0.1 * i:.2f}" '
                   'stroke-width="1.6" stroke-linejoin="round"/>')
    btc = a["models"]["buynhold_btc"]["equity"]
    path = " ".join(f"{x(j):.1f},{y(v):.1f}" for j, v in enumerate(btc))
    out.append(f'<polyline points="{path}" fill="none" stroke="{cv.RED}" stroke-width="3" stroke-linejoin="round"/>')
    legend_y = 520
    out.append(f'<rect x="{x0}" y="{legend_y - 12}" width="22" height="4" fill="{cv.RED}"/>')
    out.append(cv.text(x0 + 32, legend_y - 4, "bitcoin, held", 16, cv.MUTED, cv.MONO))
    out.append(f'<rect x="{x0 + 220}" y="{legend_y - 12}" width="22" height="4" fill="{cv.CHIP_INK}"/>')
    out.append(cv.text(x0 + 252, legend_y - 4, "the models", 16, cv.MUTED, cv.MONO))
    cv.fits(x1)
    hook = ["What models can and cannot", "do in a market"]
    return cv.frame("blog / machine learning", ["AI in", "crypto trading"], hook, "".join(out),
                    footer="Forecasts, costs, backtests and agents")


COVER = cover

FIGURES = {
    "walk-forward": walk_forward,
    "agent": agent,
}
