"""Render two slide references and reusable charts from the reviewed EDA tables."""

import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ts-proposal-mpl"))

import fitz
from fontTools.ttLib import TTCollection
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = ROOT / "eda/results/tables"
OUT = HERE / "slide_mockups"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 1440, 810
BG = "#fff6ea"
DARK = "#3b332f"
MUTED = "#756c64"
BLUE = "#267485"
ORANGE = "#bd6547"
GRID = "#e8e3dc"
RULE = "#d4cbc0"
TARGETS = ["next10_input_tokens", "next10_final_output_tokens"]

target = pd.read_csv(TABLES / "08_active_sessions_and_demand.csv")
review = json.loads((HERE / "checks.json").read_text())
horizons = pd.read_csv(HERE / "horizon_context.csv")
sessions = pd.read_csv(TABLES / "06_session_quality.csv").iloc[0]
gaps = pd.read_csv(TABLES / "06_followup_gap_distribution.csv").set_index("statistic").gap_seconds
turns = pd.read_csv(TABLES / "06_input_length_by_observed_turn.csv").query("_turn <= 10")
selected_day = review["day_selection"]["zero_based_day"]
origin = int(target.timestamp.min() // 86400) * 86400
target["day"] = (target.timestamp // 86400 - origin // 86400).astype(int)
target["elapsed_days"] = (target.timestamp - origin) / 86400
one_day = target[target.day == selected_day]
assert len(one_day) == 1440
daily = target.groupby("day").agg(n=("timestamp", "size"), **{k: (k, "mean") for k in TARGETS})
daily = daily[daily.n == 1440]


def rgb(value):
    return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))


with tempfile.TemporaryDirectory(prefix="ts-slide-fonts-") as temporary:
    temporary = Path(temporary)
    collection = TTCollection("/System/Library/Fonts/AppleSDGothicNeo.ttc", lazy=True)
    font_files = {}
    for name, index in [("body", 0), ("bold", 6)]:
        path = temporary / f"{name}.ttf"
        collection.fonts[index].save(path)
        font_files[name] = path
    font_files["title"] = Path("/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf")
    fonts = {name: fitz.Font(fontfile=str(path)) for name, path in font_files.items()}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": RULE, "axes.labelcolor": DARK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.fonttype": "path", "axes.unicode_minus": False,
    })

    def save_chart(figure, name):
        figure.savefig(ASSETS / f"{name}.png", dpi=240, facecolor="white")
        figure.savefig(ASSETS / f"{name}.svg", facecolor="white")
        plt.close(figure)

    for mode in ["day", "full"]:
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.65), sharex=True)
        title = "10-min token demand: one day" if mode == "day" else "10-min token demand: 110 days"
        fig.suptitle(title, x=0.15, y=0.985, ha="left", fontsize=20, fontweight="bold", color=DARK)
        for row, (name, color) in enumerate(zip(TARGETS, [BLUE, ORANGE])):
            ax = axes[row]
            if mode == "day":
                ax.plot((one_day.timestamp % 86400) / 3600, one_day[name], color=color, lw=1.2)
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 6, 12, 18, 24])
            else:
                ax.plot(target.elapsed_days, target[name], color=color, alpha=0.45, lw=0.35)
                ax.plot(daily.index + 0.5, daily[name], color=DARK, lw=2.0, label="Daily mean")
                ax.set_xlim(0, 110)
                ax.set_xticks([0, 20, 40, 60, 80, 100])
            if mode == "day":
                ax.set_ylim(0, 65000 if row == 0 else 13000)
                ax.set_yticks([0, 30000, 60000] if row == 0 else [0, 5000, 10000])
            else:
                ax.set_ylim(0, 210000 if row == 0 else 47000)
                ax.set_yticks([0, 100000, 200000] if row == 0 else [0, 20000, 40000])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(labelsize=14, length=3)
            ax.set_ylabel("Input tokens" if row == 0 else "Final output tokens", fontsize=16)
        if mode == "full":
            axes[0].legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.4)
        axes[-1].set_xlabel("Hour of day (UTC+8)" if mode == "day" else "Relative day",
                            fontsize=16, labelpad=6)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.86, bottom=0.14, hspace=0.23)
        save_chart(fig, "12_one_day" if mode == "day" else "12_full_period")

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    fig.suptitle("Input length by observed request order", x=0.15, y=0.985, ha="left",
                 fontsize=20, fontweight="bold", color=DARK)
    ax.plot(turns._turn, turns.median_input, color=BLUE, lw=2.2, marker="o", ms=6)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 1500)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([0, 500, 1000, 1500])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(labelsize=14, length=3)
    ax.set_ylabel("Median input tokens", fontsize=16)
    ax.set_xlabel("Observed request order", fontsize=16, labelpad=6)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.81, bottom=0.22)
    save_chart(fig, "13_input_by_observed_order")

    document = fitz.open()
    text_records = []

    def new_page(number):
        page = document.new_page(width=WIDTH, height=HEIGHT)
        for name, path in font_files.items():
            page.insert_font(fontname=name, fontfile=str(path))
        page.draw_rect(page.rect, color=None, fill=rgb(BG))
        text(page, 64, 26, "EDA", 70, "title")
        end = 64 + fonts["title"].text_length("EDA", fontsize=70)
        cx, cy = end + 40, 72
        page.draw_circle((cx, cy), 29, color=rgb(DARK), width=2.1)
        digit = "1" if number == 12 else "2"
        digit_width = fonts["title"].text_length(digit, fontsize=58)
        text(page, cx - digit_width / 2, 33, digit, 58, "title")
        heading = "TARGET DEMAND" if number == 12 else "CONVERSATIONS"
        text(page, cx + 46, 26, heading, 70, "title")
        text(page, 1361, 771, str(number), 15, color=MUTED, width=20)
        return page

    def text(page, x, y, value, size=21, font="body", color=DARK, width=None, leading=1.4):
        for index, line in enumerate(value.split("\n")):
            line_width = fonts[font].text_length(line, fontsize=size)
            if width is not None and line_width > width + 0.5:
                raise ValueError(f"Text exceeds its box: {line!r}: {line_width:.1f} > {width}")
            assert x + line_width <= WIDTH - 20, line
            baseline = y + size + index * size * leading
            page.insert_text((x, baseline), line, fontname=font, fontsize=size, color=rgb(color))
            text_records.append({"page": page.number + 12, "text": line,
                                 "x": x, "y": y + index * size * leading,
                                 "size": size, "width": line_width})

    def rect(page, x, y, w, h, fill="white", stroke=None, thickness=0.8):
        page.draw_rect(fitz.Rect(x, y, x + w, y + h),
                       fill=rgb(fill) if fill.startswith("#") else (1, 1, 1),
                       color=rgb(stroke) if stroke else None, width=thickness)

    def line(page, x1, y1, x2, y2, color=RULE, width=0.8):
        page.draw_line((x1, y1), (x2, y2), color=rgb(color), width=width)

    def centered(page, x, y, w, value, size=20, font="body", color=DARK):
        string_width = fonts[font].text_length(value, fontsize=size)
        text(page, x + (w - string_width) / 2, y, value, size, font, color, width=w)

    def image(page, name, x, y, w, h):
        page.insert_image(fitz.Rect(x, y, x + w, y + h),
                          filename=str(ASSETS / f"{name}.png"), keep_proportion=True)

    # Slide 12: show the forecasting target rather than infer a horizon from ACF.
    page = new_page(12)
    text(page, 64, 123, "다음 10분 누적 토큰 수요: 하루의 변동과 전체 기간의 수준 차이", 23, width=1312)
    for x in [64, 736]:
        rect(page, x, 185, 640, 392)
    image(page, "12_one_day", 72, 198, 622, 366)
    image(page, "12_full_period", 744, 198, 622, 366)
    text(page, 64, 609, "토큰 수요는 하루 안에서도 변동하며, 전체 기간에서도 수준 차이가 있다.",
         28, "bold", width=1312)
    text(page, 64, 657, "최근 수요 이력과 하루 중 시각을 공통 입력으로 사용하고, 예측 효과는 시간순 검증으로 평가한다.",
         23, width=1312)

    # Slide 13: retain the conversation evidence and separate horizon choice.
    page = new_page(13)
    text(page, 64, 123, "반복 대화와 상태 정보를 추가하고, 후속 간격을 참고해 10분 지평을 선택한다.",
         23, width=1312)
    left, right, col_width = 64, 736, 640
    rect(page, left, 185, col_width, 216, fill=BG, stroke=RULE)
    rect(page, left, 185, col_width, 44, fill=DARK)
    text(page, left + 17, 193, "Conversation statistics", 22, "bold", color=BG, width=450)
    centered(page, left + 486, 193, 145, "Value", 22, "bold", color=BG)
    metrics = [
        ("Unique conversation IDs", f"{sessions.unique_sessions:,.0f}"),
        ("Conversations with 2+ requests", f"{sessions.repeat_sessions_pct:.2f}%"),
        ("Requests in repeated conversations", f"{sessions.rows_in_repeat_sessions_pct:.2f}%"),
        ("Median observed follow-up gap", f"{gaps['50%']:.0f} sec"),
    ]
    for index, (label, value) in enumerate(metrics):
        top = 229 + index * 43
        if index:
            line(page, left, top, left + col_width, top)
        text(page, left + 17, top + 9, label, 20, width=450)
        centered(page, left + 486, top + 8, 145, value, 22, "bold")

    rect(page, right, 185, col_width, 253)
    image(page, "13_input_by_observed_order", right + 8, 190, 624, 246)

    text(page, left, 424, "Horizon comparison", 23, "bold", width=640)
    table_y = 465
    widths = [120, 300, 220]
    starts = [left, left + widths[0], left + widths[0] + widths[1]]
    headers = ["Horizon", "Follow-up gaps ≤ horizon (%)", "Zero-demand windows (%)"]
    for x, width, label in zip(starts, widths, headers):
        centered(page, x, table_y, width, label, 17, "bold", MUTED)
    line(page, left, table_y + 30, left + 640, table_y + 30)
    for index, row in enumerate(horizons.itertuples(index=False)):
        y = table_y + 34 + index * 34
        chosen = row.horizon_minutes == 10
        if chosen:
            rect(page, left, y - 1, 640, 34, fill="#e8eeeb")
        label = f"{row.horizon_minutes} min"
        values = [label, f"{row.observed_followup_gap_coverage_pct:.1f}%", f"{row.zero_input_target_pct:.1f}%"]
        for x, width, value in zip(starts, widths, values):
            centered(page, x, y + 1, width, value, 21, "bold" if chosen else "body", BLUE if chosen else DARK)

    text(page, right + 23, 460, "대화 상태를 추가할 동기", 23, "bold", width=594)
    text(page, right + 23, 499,
         "관측 순서에 따라 입력 길이의 중앙값이 다르다.\n"
         "최근 대화의 수와 상태를 예측 후보로 사용한다.",
         21, width=594, leading=1.45)
    text(page, left, 611, "10분은 후속 요청 포괄 범위와 단기성을 고려한 설계 선택이다.",
         21, "bold", width=1312)
    line(page, 64, 651, 1376, 651, DARK, 1.0)
    text(page, 64, 663, "B", 25, "bold", BLUE, width=30)
    text(page, 108, 666, "A + 최근 5·15·30분 활동 대화 수", 22, width=1268)
    text(page, 64, 704, "C", 25, "bold", BLUE, width=30)
    text(page, 108, 707, "B + 최근성·관측 횟수·관측 연령·마지막 입력 길이·입력 길이 변화 요약", 22, width=1268)

    pdf_path = OUT / "slides_12_13_reference.pdf"
    document.save(pdf_path, garbage=4, deflate=True)
    for index, page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(4 / 3, 4 / 3), alpha=False).save(OUT / f"slide_{index + 12}.png")
    document.close()

    with fitz.open(pdf_path) as exported:
        assert len(exported) == 2
        for page in exported:
            assert page.rect == fitz.Rect(0, 0, WIDTH, HEIGHT)
            assert "\ufffd" not in page.get_text()
            for block in page.get_text("dict")["blocks"]:
                for paragraph in block.get("lines", []):
                    for span in paragraph.get("spans", []):
                        box = fitz.Rect(span["bbox"])
                        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= WIDTH and box.y1 <= HEIGHT, span["text"]
        assert "76.7%" in exported[1].get_text()
        assert "91.51%" in exported[1].get_text()
        combined_text = "\n".join(page.get_text() for page in exported).replace("\u00a0", " ")
        for expected in ["Conversation statistics", "131 sec", "10 min", "76.7%", "91.51%"]:
            assert expected in combined_text
        for removed in ["매 1분 시점", "좌우 y축", "하루 선정:", "인접 목표 구간", "¹", "²"]:
            assert removed not in combined_text
        for name, box in [
            ("13_conversation_statistics", fitz.Rect(64, 185, 704, 401)),
            ("13_horizon_comparison", fitz.Rect(64, 420, 704, 601)),
        ]:
            exported[1].get_pixmap(matrix=fitz.Matrix(2, 2), clip=box, alpha=False).save(ASSETS / f"{name}.png")
    with zipfile.ZipFile(OUT / "charts_english.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for asset in sorted(ASSETS.iterdir()):
            if asset.suffix in {".png", ".svg"}:
                bundle.write(asset, arcname=asset.name)
    (OUT / "layout.json").write_text(json.dumps(text_records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf_path), "pages": 2,
                      "previews": ["slide_12.png", "slide_13.png"],
                      "charts": 3, "statistical_tables": 2,
                      "text_boxes_checked": len(text_records)}, ensure_ascii=False))
