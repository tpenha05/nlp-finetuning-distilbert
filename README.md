# Curvas de aprendizado: fine-tuning de DistilBERT

Solução da [atividade de NLP do professor Tiago F. Tavares](https://tiagoft.github.io/nlp_course/language_models/07-finetuning_bert/).

Compara, em **um único gráfico**, quatro estratégias de classificação de sentimento no IMDb:

| Método | Representação | O que é treinado |
|---|---|---|
| BoW + regressão logística | Contagem de unigramas/bigramas | Vocabulário e classificador, somente no treino |
| DistilBERT do zero | Mesmo encoder, pesos aleatórios | Encoder e cabeça linear |
| DistilBERT congelado | Encoder pré-treinado, token CLS | Apenas a cabeça linear |
| DistilBERT fine-tuning | Encoder pré-treinado, token CLS | Encoder e cabeça linear |

O código gera CSVs, PNG, gráfico vetorial e **PDF com resultados e interpretação**. O bônus com AG News também está implementado, mas precisa de uma execução própria.

## Execução local

Python 3.10 ou superior. GPU CUDA é recomendada para o experimento completo. O código usa CPU automaticamente quando CUDA não está disponível. O primeiro uso baixa os pesos de `distilbert/distilbert-base-uncased` e os splits oficiais do dataset; reserve alguns GB de disco e memória.

```bash
git clone https://github.com/tpenha05/nlp-finetuning-distilbert.git
cd nlp-finetuning-distilbert
python -m venv .venv
```

Ative o ambiente:

```powershell
# Windows / PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Depois:

```bash
python -m pip install -e .
python -m pytest -q
python -m bert_curves.experiment --config configs/quick.json --output results/custom/quick
```

Os resultados são gravados após cada treinamento. A pasta de saída deve estar vazia de métricas de execuções anteriores: o programa recusa sobrescrever `metrics.csv`.

## Notebook / Colab

Abra [notebooks/atividade.ipynb](notebooks/atividade.ipynb) no Jupyter ou [no Google Colab](https://colab.research.google.com/github/tpenha05/nlp-finetuning-distilbert/blob/main/notebooks/atividade.ipynb). No Colab, selecione um ambiente com GPU antes de executar.

## Configurações

| Configuração | Treino por ponto | Sementes | Teste | Tokens | Épocas |
|---|---|---|---|---|---|
| `quick.json` | 50, 100, 200 | 42, 43 | Amostra fixa de 200 | 64 | 2 |
| `full.json` | 50 a 5.000 | 42, 43, 44 | Split oficial completo | 256 | 3 |

A execução rápida é exploratória. Ela não equivale ao treinamento completo: poucos exemplos, truncamento e poucas atualizações podem fazer qualquer método superar outro. O relatório descreve o que foi observado, sem substituir resultados por expectativas teóricas.

Para o experimento maior:

```bash
python -m bert_curves.experiment --config configs/full.json --output results/custom/imdb_full
```

Para o bônus:

```bash
python -m bert_curves.experiment --config configs/full.json --dataset ag_news --output results/custom/ag_news_full
```

Cada execução gera um gráfico próprio. Não são incluídos resultados fictícios do bônus.

## Arquivos gerados

- `metrics.csv`: acurácia de treino, acurácia/F1 macro de teste, duração, perdas e parâmetros treináveis de cada execução.
- `summary.csv`: média e desvio-padrão por modelo e tamanho.
- `learning_curves.png` e `learning_curves.pdf`: as quatro curvas no mesmo gráfico.
- `relatorio.pdf`: método, gráfico, tabela e discussão.
- `manifest.json`: configurações, revisão exata do modelo, hashes dos dados, índices amostrados e ambiente.

Para gerar novamente o relatório a partir das medições, sem treinar:

```bash
python -m bert_curves.report results/imdb_quick
```

## Protocolo experimental

1. Usa os splits oficiais de treino/teste, sem escolher hiperparâmetros com o teste.
2. Seleciona uma única amostra de teste estratificada, independente das sementes do treino; `test_size: null` usa o teste inteiro.
3. Para cada semente, produz uma ordem estratificada: as amostras menores são prefixos das maiores. Os quatro métodos recebem os mesmos índices.
4. Reinicia pesos, otimizador e gerador dos batches a cada modelo/tamanho/semente. Não faz treinamento incremental entre pontos da curva.
5. Usa cabeça linear sobre CLS, como no enunciado. No congelado, desativa gradientes e dropout do encoder; no teste, usa `eval()` e `inference_mode()`.
6. Plota média e faixa de ±1 desvio-padrão entre sementes. A faixa **não** é intervalo de confiança.
7. Grava todas as execuções; o relatório rejeita resultados incompletos ou duplicados.

BoW recebe o texto inteiro; o Transformer recebe os primeiros tokens até o limite configurado. O número de épocas é fixo, portanto amostras maiores também recebem mais atualizações. O encoder do zero usa taxa 2e-4, fine-tuning 2e-5 e a cabeça usa 1e-3. Essas escolhas são pré-definidas; não foram otimizadas no teste. Uma comparação com orçamento de atualizações igual ou busca de hiperparâmetros requer outro protocolo com validação retirada do treino.

O tokenizer do modelo do zero continua sendo o WordPiece pré-treinado. “Do zero” se refere aos **pesos do encoder e classificador**, sem pré-treinamento de linguagem.

## Testes

Os testes não baixam modelos ou datasets. Usam um encoder minúsculo criado localmente para verificar que o congelamento impede atualizações do encoder, que fine-tuning permite atualizações e que a cabeça é treinada. Também verificam amostragem, agregação e rejeição de relatórios incompletos. Esse encoder é apenas uma ferramenta de teste; os experimentos usam DistilBERT base real.

## Entrega

Use `relatorio.pdf` da execução escolhida para a entrega no Blackboard. Revise o escopo exploratório ou execute a configuração maior antes de entregar se quiser uma comparação mais abrangente. O envio ao Blackboard é manual.

## Referências

- [Enunciado da atividade](https://tiagoft.github.io/nlp_course/language_models/07-finetuning_bert/)
- [DistilBERT na documentação oficial Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/distilbert)
- [Modelo DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- [IMDb / Stanford NLP](https://huggingface.co/datasets/stanfordnlp/imdb)
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)

Os dados e pesos não são redistribuídos neste repositório; consulte os termos dos respectivos mantenedores.
