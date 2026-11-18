# AV2 (Parte 2) — Manual de Uso

Este manual descreve como reproduzir a coleta de latência, a submissão dos conjuntos de Referência e Produção, o cálculo estatístico não paramétrico (percentis, boxplot e gráfico de violino), os testes de drift (PSI e KS) e o dashboard interativo em Streamlit na branch `av2-p2`.

## 1. Pré-requisitos e instalação

- Python **3.12** (`>=3.12,<3.13` definido em [`.python-version`](../../.python-version) e [`pyproject.toml`](../../pyproject.toml)).
- `uv` instalado no PATH.
- O modelo versionado [`models/adult-income-v1.pkl`](../../models/adult-income-v1.pkl) e sua metadata [`models/metadata.json`](../../models/metadata.json).

Na raiz do repositório:

```powershell
uv sync --frozen --python 3.12
```

> **Modo offline:** os arquivos [`docs/av2-p2/evidence/reference-inputs.jsonl`](evidence/reference-inputs.jsonl), [`production-inputs.jsonl`](evidence/production-inputs.jsonl), [`latency-benchmark.jsonl`](evidence/latency-benchmark.jsonl), [`reference-responses.jsonl`](evidence/reference-responses.jsonl), [`production-responses.jsonl`](evidence/production-responses.jsonl) e [`summary.json`](evidence/summary.json) já acompanham o repositório para permitir gerar gráficos, relatórios e abrir o dashboard sem conexão à internet.

## 2. Subir o serviço BentoML

Em um terminal na raiz do projeto:

```powershell
$env:CAREERPATH_LOG_PATH = 'logs/part2-inference.jsonl'
uv run --frozen bentoml serve careerpath.service:CareerPathService --host 127.0.0.1 --port 3000 --do-not-track
```

Em Linux/macOS:

```bash
export CAREERPATH_LOG_PATH=logs/part2-inference.jsonl
uv run --frozen bentoml serve careerpath.service:CareerPathService --host 127.0.0.1 --port 3000 --do-not-track
```

Confirme em outro terminal que `http://127.0.0.1:3000/health` responde HTTP `200`.

## 3. (Opcional) Baixar o dataset original UCI Adult

Se quiser reconstruir os conjuntos diretamente dos arquivos brutos em `data/raw/` (ignorados no Git) com validação SHA-256:

```powershell
uv run --frozen python scripts/download_data.py
```

Caso `data/raw/` não exista, [`scripts/part2_collect.py`](../../scripts/part2_collect.py) utiliza automaticamente as fixtures reproduzíveis em [`docs/av2-p2/evidence/`](evidence/).

## 4. Executar benchmark e submissão de Referência e Produção

Com o servidor BentoML ativo na porta `3000`, execute em outro terminal:

```powershell
uv run --frozen python scripts/part2_collect.py `
  --url http://127.0.0.1:3000 `
  --count 500 `
  --benchmark-count 500 `
  --seed 42 `
  --evidence-dir docs/av2-p2/evidence
```

### Parâmetros de `scripts/part2_collect.py`

| Opção | Padrão | Finalidade |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:3000` | URL base HTTP/HTTPS do serviço BentoML |
| `--count` | `500` | Quantidade de perfis válidos em Referência e Produção (mínimo `100`) |
| `--benchmark-count` | `500` | Quantidade de pedidos enviados no benchmark de latência (mínimo `100`) |
| `--seed` | `42` | Semente determinística para amostragem e perturbação realista |
| `--data-dir` | `data/raw` | Pasta com `adult.data` e `adult.test` (usa fallback se ausente) |
| `--metadata-path` | `models/metadata.json` | Metadata usada para conferir SHA-256 dos arquivos Adult |
| `--evidence-dir` | `docs/av2-p2/evidence` | Diretório de saída dos JSONLs gerados |
| `--prepare-only` | `false` | Gera apenas `reference-inputs.jsonl` e `production-inputs.jsonl` sem chamar HTTP |

Saídas gravadas em `--evidence-dir`:
- `reference-inputs.jsonl` e `production-inputs.jsonl`: inputs validados pelo schema [`Profile`](../../src/careerpath/schema.py).
- `latency-benchmark.jsonl`: tempos de resposta HTTP medidos no cliente para cada pedido.
- `reference-responses.jsonl` e `production-responses.jsonl`: entradas e predições devolvidas por `POST /predict`.

## 5. Calcular estatísticas não paramétricas, KS, PSI e gerar gráficos

Execute [`scripts/part2_report.py`](../../scripts/part2_report.py) (funciona inclusive **offline** sobre os arquivos coletados):

```powershell
uv run --frozen python scripts/part2_report.py `
  --evidence-dir docs/av2-p2/evidence `
  --inference-log logs/part2-inference.jsonl `
  --sla-ms 400
```

### Parâmetros de `scripts/part2_report.py`

| Opção | Padrão | Finalidade |
| --- | --- | --- |
| `--evidence-dir` | `docs/av2-p2/evidence` | Diretório com `latency-benchmark.jsonl`, `reference-responses.jsonl` e `production-responses.jsonl` |
| `--out-dir` | mesmo de `--evidence-dir` | Diretório onde `summary.json` e os gráficos `.png` são salvos |
| `--inference-log` | `None` | Caminho opcional do JSONL do servidor para cruzar latência ASGI com a latência HTTP cliente |
| `--sla-ms` | `400.0` | Limiar do SLA avaliado sobre `p95` (`p95 < sla_ms`) |
| `--epsilon` | `1e-6` | Suavização aplicada no PSI para categorias com contagem zero |

Artefatos gerados:
- `summary.json`: relatório completo em JSON (estatísticas de latência cliente/servidor, status SLA, PSI de `occupation`, KS de `probability_above_50k` e tabela complementar das 8 variáveis).
- `latency_boxplot.png`, `latency_violin.png` e `latency_boxplot_violin.png`.
- `drift_psi_occupation.png` e `drift_ks_probability.png`.

## 6. Abrir o dashboard interativo em Streamlit

Para explorar métricas, gráficos, tabelas detalhadas e simular outros limiares de SLA no navegador:

```powershell
uv run --frozen streamlit run scripts/dashboard.py
```

O dashboard abre em `http://localhost:8501` com três abas:
1. **Latência & SLA:** KPIs (`p50`, `p90`, `p95`, `p99`), status do SLA, gráficos boxplot e violino combinados/isolados e tabela completa de quartis/Tukey (com comparação ao log ASGI do servidor quando presente).
2. **Drift (PSI & KS):** PSI de `occupation` (com categoria explícita `(ausente)` para `null`), KS de `probability_above_50k` (ECDF + densidade comparada) e tabela complementar de todas as variáveis do contrato.
3. **Dados & Reprodução:** amostra dos perfis submetidos e comandos de reprodução.

Variável opcional: defina `CAREERPATH_PART2_EVIDENCE_DIR` para apontar o dashboard para outro diretório de evidências por padrão.

## 7. Atalhos opcionais com `just` e verificações de qualidade

```powershell
just part2-collect
just part2-report
just dashboard
just check
```

Verificações diretas sem `just`:

```powershell
uv run --frozen ruff check .
uv run --frozen pytest -q
uv run --frozen python -m compileall -q src scripts tests
```

## 8. Solução de problemas

| Situação | Ação |
| --- | --- |
| `Failed to reach http://127.0.0.1:3000/predict` | Confirme que o serviço BentoML foi iniciado antes de rodar `scripts/part2_collect.py` ou ajuste `--url`. |
| Mensagem *"Nenhuma coleta encontrada"* no Streamlit | Execute `part2_collect.py` (com o servidor ativo) ou aponte o diretório customizado na barra lateral para `docs/av2-p2/evidence`. |
| Erro de JSON inválido ao carregar arquivos | Verifique se o arquivo JSONL está completo e em UTF-8; não edite linhas parcialmente. |
| Porta `8501` ocupada no Streamlit | Rode `uv run --frozen streamlit run scripts/dashboard.py --server.port 8502`. |
