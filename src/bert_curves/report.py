"""Gráfico único com as quatro curvas e PDF com análise dos dados observados."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

LABELS = {"bow": "BoW + regressão logística", "scratch": "DistilBERT do zero",
          "frozen": "DistilBERT congelado", "finetuned": "DistilBERT fine-tuning"}
COLORS = {"bow": "#b66a16", "scratch": "#838b98", "frozen": "#356da6", "finetuned": "#168c77"}


def aggregate(frame):
    return frame.groupby(["model", "train_size"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        repetitions=("seed", "nunique")).reset_index()


def validate_results(frame, manifest):
    cfg = manifest["config"]
    keys = ["model", "train_size", "seed"]
    expected = {(m, n, s) for m in LABELS for n in cfg["sizes"] for s in cfg["seeds"]}
    actual = set(frame[keys].itertuples(index=False, name=None))
    if manifest["status"] != "complete" or actual != expected or frame.duplicated(keys).any():
        raise ValueError("Experimento incompleto ou com resultados duplicados")
    if not frame[["accuracy", "f1_macro"]].apply(lambda col: col.between(0, 1)).all().all():
        raise ValueError("Métricas inválidas")


def generate_report(folder):
    out = Path(folder)
    frame = pd.read_csv(out / "metrics.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    validate_results(frame, manifest)
    cfg = manifest["config"]
    summary = aggregate(frame)
    summary.to_csv(out / "summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#233448"})
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    for model, label in LABELS.items():
        points = summary[summary.model == model].sort_values("train_size")
        x = points.train_size.to_numpy()
        mean = points.accuracy_mean.to_numpy()
        std = points.accuracy_std.fillna(0).to_numpy()
        ax.plot(x, mean, marker="o", label=label, color=COLORS[model], linewidth=2)
        ax.fill_between(x, (mean-std).clip(0, 1), (mean+std).clip(0, 1),
                        color=COLORS[model], alpha=.13)
    ax.set(xlabel="Número de exemplos de treinamento", ylabel="Acurácia no teste",
           title=f"Curvas de aprendizado | {cfg['dataset'].upper()}", ylim=(0, 1))
    if max(cfg["sizes"]) / min(cfg["sizes"]) > 10:
        ax.set_xscale("log")
    ax.set_xticks(cfg["sizes"])
    ax.set_xticklabels(cfg["sizes"])
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.savefig(out / "learning_curves.png", dpi=200)
    fig.savefig(out / "learning_curves.pdf")
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPT", fontName="Helvetica", fontSize=10,
                              leading=13, spaceAfter=7,
                              textColor=colors.HexColor("#233448")))
    styles.add(ParagraphStyle(name="SmallPT", fontName="Helvetica", fontSize=8.5,
                              leading=11, spaceAfter=6,
                              textColor=colors.HexColor("#536273")))
    body, small = styles["BodyPT"], styles["SmallPT"]
    def p(text, style=body):
        return Paragraph(text, style)
    def number(value, places=1):
        return f"{value:.{places}f}".replace(".", ",")
    story = [
        Paragraph("Curvas de aprendizado com DistilBERT", styles["Title"]),
        p("Thiago Penha e Henrique Turco",
          ParagraphStyle(name="AuthorsPT", parent=body, alignment=1, spaceAfter=10)),
        p(f"<b>{cfg['dataset'].upper()}</b> | Treino: {', '.join(map(str, cfg['sizes']))} exemplos | "
          f"Teste fixo: {manifest['evaluated_test_size']} textos<br/>"
          f"{len(cfg['seeds'])} sementes; {cfg['epochs']} épocas; "
          f"batches de {cfg['batch_size']}; limite de {cfg['max_length']} tokens nos Transformers."),
        Image(str(out / "learning_curves.png"), width=17*cm, height=9.31*cm),
        p("Faixas: média ± 1 desvio-padrão entre sementes (não são intervalos de confiança).", small),
        Spacer(1, .12*cm),
    ]
    table_data = [["Treino", "BoW", "Do zero", "Congelado", "Fine-tuning"]]
    for n in cfg["sizes"]:
        row = [str(n)]
        for model in LABELS:
            point = summary[(summary.model == model) & (summary.train_size == n)].iloc[0]
            row.append(number(100*point.accuracy_mean, 2) + "%")
        table_data.append(row)
    table = Table(table_data, colWidths=[2*cm] + [3.75*cm]*4, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#233448")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#eff4f7"), colors.white]),
    ]))
    story += [p("<b>Acurácia média no teste</b>"), table, Spacer(1, .22*cm)]
    final = summary[summary.train_size == max(cfg["sizes"])].set_index("model")
    winner = final.accuracy_mean.idxmax()
    difference = 100*(final.loc["finetuned","accuracy_mean"]-final.loc["frozen","accuracy_mean"])
    story.append(p(
        f"<b>Conclusão.</b> No maior treino, <b>{LABELS[winner]}</b> obteve a melhor média. "
        f"A diferença entre fine-tuning e congelado foi de {number(difference, 2)} pontos percentuais. "
        "Congelar treina só a cabeça; o fine-tuning também adapta o encoder. "
        "O modelo do zero precisa aprender as representações com os poucos rótulos disponíveis."))
    scope = ("Execução exploratória, com amostra do teste oficial. "
             if cfg["test_size"] is not None else "Avaliação no teste oficial completo. ")
    story.append(p(
        "<b>Escopo.</b> " + scope +
        "Poucas épocas e sementes limitam a comparação. BoW recebe os textos completos; "
        "os Transformers usam textos truncados. Os resultados não representam o desempenho máximo dos métodos.",
        small))
    story.append(p(
        '<b>Fontes:</b> <link href="https://huggingface.co/datasets/stanfordnlp/imdb">IMDb (Stanford NLP)</link>'
        ' · <link href="https://huggingface.co/docs/transformers/v4.57.1/model_doc/distilbert">'
        'Documentação DistilBERT / Transformers</link>', small))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687785"))
        canvas.drawString(2*cm, 1.2*cm, "NLP | Resultados observados")
        canvas.drawRightString(19*cm, 1.2*cm, str(doc.page))
    SimpleDocTemplate(
        str(out / "relatorio.pdf"), author="Thiago Penha e Henrique Turco",
        title="Curvas de aprendizado com DistilBERT", pagesize=(21*cm,29.7*cm),
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm,
        bottomMargin=1.8*cm,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    generate_report(parser.parse_args().folder)
