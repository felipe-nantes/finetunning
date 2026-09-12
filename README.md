# decria-sec — fine-tuning QLoRA de um assistente de cibersegurança numa GTX 1060 6GB

> **English summary.** End-to-end QLoRA fine-tune of `Qwen/Qwen3-1.7B` into a bilingual
> (EN strong, PT functional) assistant for *authorized* penetration testing and defense,
> trained on a single 2016-era GTX 1060 6GB (Pascal). Public + locally-generated synthetic
> data, honest before/after evaluation, GGUF export for Ollama, everything reproducible from
> this repo. Zero paid APIs.

Projeto de portfólio: provar que dá pra fazer fine-tuning de um LLM open-source com hardware
de consumo antigo e ferramentas gratuitas, documentando cada decisão. O modelo final responde
em inglês e português sobre metodologia de pentest autorizado, ferramentas, interpretação de
saídas (nmap, gobuster...) e mitigações. Não gera malware nem exploits prontos; pedidos fora
de escopo são recusados (e isso é avaliado).

## Por que isso é difícil

| Restrição | Consequência |
|---|---|
| GTX 1060 6GB é Pascal (`sm_61`) | Sem bf16. fp16 roda a 1/64 da velocidade. **Todo compute em fp32.** |
| Compute capability 6.1 | Unsloth, flash-attention, Triton/Liger exigem 7.0+. **Nenhum deles é usado.** |
| Wheels PyTorch CUDA 13.x removeram Pascal | Pinado em `cu126`, cuja SASS `sm_60` roda no `sm_61`. Nunca `cu130+`. |
| 6 GB de VRAM | Modelos de 7B não cabem. Base primária: **Qwen3-1.7B** em NF4 (~3.2 GB de pico estimado). |
| Sem API paga | Dados sintéticos gerados localmente via Ollama (`qwen2.5:7b-instruct-q4_K_M`). |

Detalhes e alternativas descartadas: [spec de design](docs/superpowers/specs/2026-09-11-qlora-cybersec-design.md).
Plano de implementação com cada passo: [plano](docs/superpowers/plans/2026-09-11-qlora-cybersec.md).

## Pipeline

```
00_check_env      gate: GPU, arch list, NF4 round-trip, tok/s, pico de VRAM
01_build_dataset  Trendyol + AlicanKiraz0 (apache-2.0) -> filtros -> dedup -> split
02_gen_synthetic  ATT&CK + CWE Top 25 + OWASP WSTG + docs de ferramentas -> Ollama -> Q/A (EN/PT)
03_train          QLoRA: NF4 + LoRA r=16, fp32, loss so na resposta, checkpoint/resume
04_eval           CyberMetric-500 (MCQ), val loss, 12 prompts fixos, 5 de regressao
05_merge_export   merge em CPU -> GGUF f16 -> Q4_K_M / Q8_0 -> Modelfile -> ollama create
06_push_hub       adapter + GGUF + dataset no Hugging Face Hub, com cards e NOTICE
```

Cada script tem a lógica em `scripts/<nome>.py` (importável, testado) e um atalho numerado
`scripts/0N_<nome>.py`. Helpers compartilhados em `scripts/common.py`.

## Duas máquinas

| Máquina | Papel |
|---|---|
| Laptop (i5-1235U, sem GPU NVIDIA) | escrever código, rodar testes unitários, teste final do GGUF em CPU |
| Desktop (GTX 1060 6GB, Windows + WSL2 Ubuntu) | tudo que usa GPU: gate, geração sintética, treino, avaliação, export |

O `pyproject.toml` escolhe o torch certo sozinho: build CUDA 12.6 no Linux (WSL2), build CPU
no Windows. Por isso **no desktop, use sempre o terminal Ubuntu do WSL2**, nunca o PowerShell.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # instala uv (Linux/WSL)
git clone https://github.com/felipe-nantes/finetunning.git && cd finetunning
uv sync                                               # Python 3.11 + deps pinadas + torch
uv run pytest                                         # 46 testes, CPU, ~10 s
```

No Windows, se o `uv` falhar ao instalar o Python 3.11 com "Missing expected target directory",
defina `UV_PYTHON_INSTALL_DIR=C:\Users\<voce>\.uvpy` e rode `uv sync` de novo.

## Runbook no desktop (WSL2)

Ordem obrigatória. Cada passo é um gate: só siga se o anterior fechou.

```bash
nvidia-smi                                            # driver Windows >= 535; precisa listar a 1060
uv run python scripts/00_check_env.py                 # precisa imprimir PASS
```

```bash
uv run python scripts/01_build_dataset.py             # ~3k exemplos publicos -> data/processed/
```

> **Ollama e WSL2.** Instale o Ollama **dentro do WSL2** (`curl -fsSL https://ollama.com/install.sh | sh`)
> para que `localhost:11434` funcione sem configuração extra. Se em vez disso o Ollama rodar no
> Windows, suba-o com `OLLAMA_HOST=0.0.0.0` e, no WSL2, passe `--ollama-url http://<IP do Windows>:11434`
> (ou defina a variável de ambiente `OLLAMA_HOST` com o mesmo valor).

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M                # gerador local (4.7 GB, cabe na 1060)
uv run python scripts/02_gen_synthetic.py --fetch-seeds
uv run python scripts/02_gen_synthetic.py --target 1000   # ~5-6 h, rode de noite
```

```bash
uv run python scripts/03_train.py --config configs/smoke.yaml   # Qwen3-0.6B, 100 steps
uv run python scripts/03_train.py --config configs/1.7b.yaml    # run real, ~8-11 h estimadas
```

> **Checklist do smoke:** no log de treino, logo antes do primeiro passo, precisa aparecer
> `trainable params cast back to fp32: N` com `N > 0` e nenhum parâmetro treinável em bf16 — é
> o `force_fp32_trainable` desfazendo o cast pra bf16 que o `SFTTrainer` do TRL faz em modelos
> quantizados. Sem essa linha (ou com `N == 0`), o treino está rodando fora do fp32 que a 1060
> (Pascal) exige.

```bash
uv run python scripts/04_eval.py --config configs/1.7b.yaml     # base vs adapter -> docs/
```

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build && cmake --build ../llama.cpp/build -j --target llama-quantize
uv pip install -r ../llama.cpp/requirements.txt
uv run python scripts/05_merge_export.py --config configs/1.7b.yaml
cd outputs/qwen3-1.7b/gguf && ollama create decria-sec -f Modelfile && ollama run decria-sec "Explique IDOR de forma autorizada"
ollama show decria-sec --modelfile     # confirma que o TEMPLATE termina no prefixo do assistant
```

```bash
uv run huggingface-cli login
uv run python scripts/06_push_hub.py --user <HF_USER> --config configs/1.7b.yaml --hours <horas medidas>
```

Queda de energia no meio do treino: rode o mesmo comando de novo. O script retoma do último
checkpoint sozinho.

## Dados e licenças

| Fonte | Tipo | Licença | Uso |
|---|---|---|---|
| `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | instruções, EN, 53k | apache-2.0 | base pública (filtrada, cap 3k junto com a de baixo) |
| `AlicanKiraz0/Cybersecurity-Dataset-v1` | instruções, EN, 2.4k | apache-2.0 | base pública |
| MITRE ATT&CK (STIX) | técnicas + mitigações | licença MITRE (atribuição) | seed sintético |
| CWE Top 25 (API oficial) | fraquezas | termos MITRE (atribuição) | seed sintético |
| OWASP WSTG | guia de teste web | CC BY-SA 4.0 | seed sintético |
| nmap usage | doc de ferramenta | fatos de uso | seed sintético |
| CyberMetric-500 | benchmark MCQ | sem licença de redistribuição | **só avaliação**, citado, nunca publicado |

O dataset publicado sai como **cc-by-sa-4.0**: é o guarda-chuva que satisfaz o share-alike do
WSTG. `NOTICE` credita cada fonte com a licença original. Treino = publicado, então qualquer
pessoa reproduz exatamente o que o modelo viu.

Filtros aplicados à base pública, nesta ordem: só inglês, resposta entre 80 e 1.500 tokens,
sem recusas genéricas, dedup exato + MinHash (0.8), amostragem estratificada por tema
(recon, web, rede, privesc, defesa, CVE). Camada sintética: 70% EN / 30% PT, ancorada no
trecho-fonte (a resposta precisa citar termos do chunk), dedup contra a base.

## Resultados

Pendente do run no desktop. Esta seção vai receber, nesta ordem:

1. Saída literal do `00_check_env.py` (GPU, arch list, tok/s, pico de VRAM).
2. Horas de parede e pico de VRAM do run de 1.7B.
3. Tabela `docs/eval_results.json`: acurácia no CyberMetric-500, base vs adapter, IC 95%.
4. Trechos de `docs/samples.md`: 2 respostas EN, 1 PT e a recusa do prompt fora de escopo.
5. Links dos 3 repositórios no Hugging Face Hub.

Expectativa honesta, escrita antes de ver os números: ganho pequeno em MCQ. SFT com ~4k
exemplos ensina foco, formato e idioma, não conhecimento novo. O que deve mudar de forma
visível é a qualidade das respostas abertas e a recusa consistente de pedidos fora de escopo.

## Decisões registradas

- **Qwen3.5 existe e foi descartado.** Os modelos 2B/4B são multimodais com atenção linear
  híbrida e vocabulário de 248k; suporte imaturo em Pascal/llama.cpp e pior orçamento de VRAM.
  Fica como experimento futuro.
- **Loss só na resposta via prompt/completion**, não via `assistant_only_loss`: o template do
  Qwen3 não tem blocos `{% generation %}`. O que o template emite como prefixo de geração é
  exatamente o que o modelo vê na inferência.
- **`warmup_steps=0.03`** em vez de `warmup_ratio`: a transformers 5.x unificou os dois campos
  (fração `< 1` vira ratio). Descoberto por um teste unitário que constrói o `SFTConfig`.
- **Plano B documentado, não escondido:** se a 1060 ficar abaixo de ~40 tok/s no gate, o
  mesmo dataset e os mesmos scripts 04–06 rodam com treino no Colab/Kaggle T4 + Unsloth.

## Follow-ups antes do run de 1.7B

Itens reais, adiados deliberadamente na revisão final (não bloqueiam esta branch, mas devem
ser resolvidos antes do run completo de produção):

- Filtro de comprimento da base pública por tokens vs o `max_length` de 768 usado no treino
  (hoje `response_len_ok` conta tokens do próprio tokenizer, mas o corte não está alinhado
  1:1 com o truncamento aplicado em treino/eval).
- Flush incremental e resume na geração sintética (`02_gen_synthetic.py` só grava train/val
  no final do `--target`; uma queda de energia no meio da noite perde a sessão inteira).
- ~100–150 pares fora-de-escopo → recusa (EN/PT) na base de treino, para o modelo aprender a
  recusar de forma consistente (hoje só a avaliação testa isso; o treino não tem exemplos
  dedicados de recusa).
- Model card completo no Hugging Face Hub: fontes, licenças, val loss, IC 95%, amostras
  qualitativas e limitações conhecidas.

## Estrutura

```
configs/            smoke.yaml, 1.7b.yaml, 4b.yaml
scripts/            common.py + 7 modulos + 7 atalhos numerados
tests/              pytest, tudo roda em CPU
eval/               12 prompts fixos + 5 de regressao (congelados)
data/processed/     train/val jsonl (vai pro Hub, nao pro git)
data/manifest.json  manifest cumulativo: fontes publicas + sintetico, contagens train/val
docs/               spec, plano, eval_results.json, samples.md
outputs/            checkpoints, merges, gguf (fora do git)
NOTICE              atribuicao de cada fonte de dados
Modelfile           gerado pelo 05_merge_export
```
