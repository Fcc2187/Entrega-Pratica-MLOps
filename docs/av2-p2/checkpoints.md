# AV2 (Parte 2) — Checkpoints

> Pasta separada de [`docs/av2-p1/`](../av2-p1/checkpoints.md) para distinguir os entregáveis da Parte 1 e da Parte 2 na branch `av2-p2`.

## Entrega da Parte 2 — 18/11/2026

Estado: concluída na branch `av2-p2` (base `origin/av2` `02c5276`).

### O que foi feito

- Renomeada a pasta `docs/av2/` para `docs/av2-p1/` com atualização de links no `README.md` e documentos afetados, e criada `docs/av2-p2/`.
- Implementado o subpacote [`src/careerpath/analytics/`](../../src/careerpath/analytics/__init__.py) (`io.py`, `latency.py`, `drift.py`, `plots.py`) com cálculo não paramétrico de latência (`min`, `Q1`, `mediana`, `Q3`, `IQR`, bigodes Tukey, `p90`, `p95`, `p99`, `max`, verificação de SLA `< 400 ms`), seleção KS vs PSI segundo a regra de `range > 100`, PSI categórico (divergência KL simetrizada/Jeffreys, `null` em `"(ausente)"` e `epsilon = 1e-6`), KS biamostral (`scipy.stats.ks_2samp`, curva $\delta(x) = |F_{\mathrm{ref}}(x) - F_{\mathrm{prod}}(x)|$ e limiar crítico $D_{\mathrm{crit}}(0{,}05)$), semáforos com regra mínima de `100` amostras, tabela complementar das 8 variáveis e builders compartilhados Matplotlib/Seaborn (`boxplot`, `violino`, `PSI`, `KS`), além de [`src/careerpath/http_utils.py`](../../src/careerpath/http_utils.py) para validação de URL e controle de redirecionamento nos scripts CLI.
- Criados [`scripts/part2_collect.py`](../../scripts/part2_collect.py) (amostras determinísticas de 500 registros de `adult.data` e `adult.test` + drift controlado, benchmark de 500 chamadas `POST /predict` e submissão dos dois conjuntos ao servidor) e [`scripts/part2_report.py`](../../scripts/part2_report.py) (cálculo de métricas e exportação de `summary.json` + 5 PNGs).
- Criado dashboard interativo [`scripts/dashboard.py`](../../scripts/dashboard.py) em Streamlit (3 abas: Latência & SLA, Drift PSI & KS com semáforos e limiares configuráveis na barra lateral, Dados & Reprodução, estados vazio/erro/sucesso, `width="stretch"` e textos acessíveis em PT-BR).
- Criados testes automatizados em [`tests/test_analytics.py`](../../tests/test_analytics.py) e [`tests/test_part2_scripts.py`](../../tests/test_part2_scripts.py) cobrindo regras KS/PSI, casos de borda (`null`, contagem zero, amostra vazia, `IQR=0`, servidor offline, SHA-256 inválido) e os 3 estados do Streamlit via `AppTest`.
- Executada coleta real contra servidor BentoML `1.4.39` ativo (`1.500` chamadas HTTP `POST /predict` registradas em `logs/part2-inference.jsonl`) e geradas evidências offline em [`docs/av2-p2/evidence/`](evidence/summary.json).
- Redigidos [`parte-2-relatorio.md`](parte-2-relatorio.md) e [`parte-2-manual.md`](parte-2-manual.md); atualizados [`README.md`](../../README.md), [`justfile`](../../justfile) e [`docs/documentacao-IA.md`](../documentacao-IA.md).

### Resultados observados contra o servidor real

- Benchmark cliente (`n=500`): `mediana = 31,64 ms`, `p90 = 40,10 ms`, `p95 = 42,40 ms` (SLA `< 400 ms` aprovado), `p99 = 59,01 ms`, `6` outliers de Tukey (máx. `144,56 ms`).
- Log interno ASGI (`n=1500`): `mediana = 14,00 ms`, `p95 = 17,38 ms`, `p99 = 27,65 ms`.
- Drift entrada (`occupation`, PSI): `0,1416` (`Atenção: 0,10 a 0,25`).
- Drift saída (`probability_above_50k`, KS): `D = 0,0860`, `p-valor = 0,0495` (`Estável < 0,10` em magnitude).
