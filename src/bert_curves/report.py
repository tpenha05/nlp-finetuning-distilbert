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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

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
    styles.add(ParagraphStyle(name="BodyPT", fontName="Helvetica", fontSize=10, leading=14,
                              spaceAfter=9, textColor=colors.HexColor("#233448")))
    body = styles["BodyPT"]
    def p(text):
        return Paragraph(text, body)
    story = [Paragraph("Fine-tuning de DistilBERT", styles["Title"]),
             p("Comparação de eficiência amostral em classificação de textos"),
             Spacer(1, .2*cm), Paragraph("Método e escopo", styles["Heading2"])]
    exploratory = cfg["test_size"] is not None
    story.append(p(
        f"Dataset: <b>{cfg['dataset'].upper()}</b>. Treino oficial: {manifest['official_train_size']:,} textos; "
        f"teste oficial: {manifest['official_test_size']:,}. Avaliação em {manifest['evaluated_test_size']:,} "
        f"textos fixos. Tamanhos de treino: {', '.join(map(str, cfg['sizes']))}; "
        f"sementes: {', '.join(map(str, cfg['seeds']))}. Cada ponto resume execuções independentes."))
    if exploratory:
        story.append(p("<b>Execução exploratória:</b> usa uma amostra estratificada do teste oficial e "
                       "um orçamento reduzido. O gráfico representa medições reais desta configuração; "
                       "não estima o desempenho de um treinamento longo no teste completo."))
    story.append(p(
        "BoW usa contagens de unigramas e bigramas e regressão logística. Os outros três métodos usam "
        "o mesmo encoder DistilBERT e uma cabeça linear sobre CLS: pesos aleatórios, pesos pré-treinados "
        "congelados e atualização de todos os pesos. Pesos e otimizadores são reiniciados em cada ponto."))
    story.append(Image(str(out / "learning_curves.png"), width=17*cm, height=9.31*cm))
    story.append(p("Faixas: média ± 1 desvio-padrão entre sementes; não são intervalos de confiança. "
                   "O eixo horizontal é o tamanho total do treino e o vertical é a acurácia no teste."))
    story.append(PageBreak())
    story.append(Paragraph("Resultados e interpretação", styles["Heading1"]))
    table_data = [["Treino", "BoW", "Do zero", "Congelado", "Fine-tuning"]]
    for n in cfg["sizes"]:
        row = [str(n)]
        for model in LABELS:
            pt = summary[(summary.model == model) & (summary.train_size == n)].iloc[0]
            std = 0 if pd.isna(pt.accuracy_std) else pt.accuracy_std
            row.append(f"{pt.accuracy_mean:.3f} ± {std:.3f}")
        table_data.append(row)
    table = Table(table_data, colWidths=[2*cm] + [3.75*cm]*4, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#233448")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8 if len(cfg["sizes"]) > 4 else 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#eff4f7"), colors.white])
    ]))
    story += [p("Acurácia média ± desvio-padrão por tamanho de treino."), table, Spacer(1, .4*cm)]
    nmax, nmin = max(cfg["sizes"]), min(cfg["sizes"])
    final = summary[summary.train_size == nmax].set_index("model")
    winner = final.accuracy_mean.idxmax()
    story.append(p(f"No maior tamanho ({nmax}), a maior acurácia média foi de <b>{LABELS[winner]}</b> "
                   f"({final.loc[winner, 'accuracy_mean']:.3f}). Isso descreve este orçamento e estas "
                   "amostras; não estabelece superioridade geral do método."))
    for model, label in LABELS.items():
        pts = summary[summary.model == model].set_index("train_size")
        change = pts.loc[nmax, "accuracy_mean"] - pts.loc[nmin, "accuracy_mean"]
        story.append(p(f"<b>{label}:</b> mudança entre {nmin} e {nmax} exemplos: "
                       f"{100*change:+.1f} pontos percentuais. F1 macro no maior tamanho: "
                       f"{pts.loc[nmax, 'f1_mean']:.3f}."))
    story.append(p(
        f"A diferença fine-tuning menos congelado no maior treino é "
        f"{100*(final.loc['finetuned','accuracy_mean']-final.loc['frozen','accuracy_mean']):+.1f} "
        "pontos percentuais. Congelar reduz o número de parâmetros ajustados e evita modificar "
        "as representações; fine-tuning pode adaptar as representações, mas exige mais otimização. "
        "O encoder aleatório precisa aprender essas representações com os poucos rótulos disponíveis."))
    story.append(PageBreak())
    story.append(Paragraph("Limitações e reprodução", styles["Heading1"]))
    story.append(p(
        f"Foram usadas {cfg['epochs']} épocas, batches de {cfg['batch_size']} e truncamento em "
        f"{cfg['max_length']} tokens. BoW recebe os textos completos. Essa diferença de contexto, "
        "o orçamento fixo de épocas e as taxas de aprendizado distintas limitam a comparação. "
        "As configurações são pré-definidas, sem ajuste com o teste. Há apenas "
        f"{len(cfg['seeds'])} sementes; não se deve inferir significância estatística destas faixas. "
        "Para estudar curvas mais estáveis, use configs/full.json e, se desejar ajustar hiperparâmetros, "
        "separe uma validação dentro do treino oficial."))
    story.append(p(
        "As curvas não precisam ser monotônicas. Subamostras, inicialização e otimização afetam os "
        "resultados; diferenças pequenas exigem mais repetições. O bônus AG News está implementado "
        "como opção, mas só pode ser interpretado depois de sua própria execução."))
    story.append(p(
        "Arquivos de auditoria: metrics.csv contém cada execução, summary.csv contém a agregação e "
        "manifest.json registra configurações, índices, hashes dos dados, revisão do modelo e ambiente."))
    story.append(Paragraph("Fontes", styles["Heading2"]))
    story.append(p(
        '<link href="https://tiagoft.github.io/nlp_course/language_models/07-finetuning_bert/">'
        'Enunciado: Fine-tuning BERT (Tiago F. Tavares)</link><br/>'
        '<link href="https://huggingface.co/datasets/stanfordnlp/imdb">Dataset IMDb (Stanford NLP)</link><br/>'
        '<link href="https://huggingface.co/docs/transformers/v4.57.1/model_doc/distilbert">'
        'Documentação oficial DistilBERT / Transformers</link>'))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687785"))
        canvas.drawString(2*cm, 1.2*cm, "NLP | Curvas de aprendizado | Resultados observados")
        canvas.drawRightString(19*cm, 1.2*cm, str(doc.page))
    SimpleDocTemplate(str(out / "relatorio.pdf"), pagesize=(21*cm,29.7*cm),
                      leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm,
                      bottomMargin=1.8*cm).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    generate_report(parser.parse_args().folder)
