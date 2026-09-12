# Design: QLoRA de um assistente de cibersegurança em GTX 1060 6GB

**Data:** 2026-09-11
**Status:** aprovado; pipeline implementado e revisado no laptop (2026-09-12); runs de GPU pendentes no desktop
**Projeto:** `finetunning_decria`

## 1. Objetivo

Fazer fine-tuning quantizado (QLoRA) de um LLM open-source pequeno para atuar como
assistente de cibersegurança / pentest autorizado, usando apenas hardware de consumo
(desktop com GTX 1060 6GB) e ferramentas gratuitas. O processo inteiro (dados, treino,
avaliação, export, publicação) deve ser reproduzível e documentado para servir de
portfólio.

Resultado esperado:

- Adapter LoRA publicado no Hugging Face Hub com model card completo.
- GGUF quantizado (Q4_K_M e Q8_0) rodando via Ollama, inclusive no laptop sem GPU.
- Dataset de treino publicado com manifesto de fontes e licenças.
- README com tabela de avaliação antes/depois, amostras qualitativas e horas reais de GPU.

## 2. Contexto e restrições

### Hardware

| Máquina | Papel | Especificação |
|---|---|---|
| Desktop | treino, geração sintética, avaliação, export | GTX 1060 **6GB** (Pascal, sm_61), Windows + WSL2 Ubuntu |
| Laptop | desenvolvimento dos scripts, teste final do GGUF em CPU | i5-1235U, 16GB RAM, Intel UHD (sem CUDA), 117GB livres, WSL2 Ubuntu |

### Consequências da GPU Pascal

- Sem bf16. fp16 em GP106 roda a 1/64 da velocidade de fp32. **Todo compute em fp32.**
- Unsloth exige compute capability 7.0+. **Não usar.**
- Flash-attention, Triton, Liger kernels: sem suporte a sm_61. **Não usar.** Atenção via SDPA padrão do PyTorch.
- PyTorch: as wheels **`cu126` x86_64 (2.14.x)** embarcam SASS para `sm_50;sm_60;sm_70;sm_75;sm_80;sm_86;sm_90` (verificado no `build_env_setup.py` da tag). O cubin `sm_60` roda no `sm_61` da GTX 1060 por compatibilidade de minor version (X.z executa em X.w com w ≥ z). As wheels **CUDA 13.x removeram `sm_50/60/70`** e por isso **não rodam** na 1060. **Pinar uma build `cu126`** (fallback `cu128`, que também tem `sm_60`; nunca `cu130+`). O gate `00_check_env` confirma empiricamente, não confia só na lista.
- bitsandbytes: a documentação oficial (até 0.50.x) lista **NF4/FP4 para Compute Capability 6.0+**, citando explicitamente a série GTX 10x0 (Pascal). Ou seja, Pascal **não** foi removido. **Pinar uma versão recente conhecida** (ex.: 0.48–0.50) e validar no smoke test, em vez de assumir que quebrou. LLM.int8() (8-bit) exige 7.5+, mas não é usado aqui.
- Modelos de 7B não cabem: NF4 ≈ 4GB + embeddings/lm_head fp32 + ativações estouram 6GB.
- **Armadilha descoberta na implementação:** a TRL 1.13 converte todos os parâmetros treináveis de um modelo 4-bit para **bf16** dentro de `SFTTrainer.__init__`, mesmo com `bf16=False`. Na Pascal isso significaria LoRA e otimizador em bf16. O `03_train.py` força os parâmetros treináveis de volta para fp32 depois de construir o trainer e aborta se sobrar algum não-fp32; o log do smoke deve mostrar `trainable params cast back to fp32: N`.

### Idioma

Treino majoritariamente em inglês (mais dado, melhor qualidade). O modelo deve responder
em português quando perguntado em português; qualidade em PT pode ser inferior.

### Custo

Zero. Nenhuma API paga. Geração sintética com modelo local via Ollama.

## 3. Não-objetivos

- Não treinar 7B+ localmente.
- Não gerar malware funcional, exploits weaponizados ou payloads prontos. O escopo é
  metodologia, ferramentas, interpretação de resultados, mitigação e defesa.
- Não fazer DPO/RLHF nesta fase. Só SFT.
- Não construir UI. A interface de demonstração é o Ollama.
- Não otimizar para benchmark. A métrica MCQ é honesta, não alvo.

## 4. Abordagem escolhida

**QLoRA local com script próprio** (`transformers` + `peft` + `trl` + `bitsandbytes`).

Alternativas consideradas e rejeitadas:

| Alternativa | Por que não |
|---|---|
| LLaMA-Factory / Axolotl | Mais dependências, mais pontos de falha em Pascal, portfólio mostra menos domínio |
| Colab/Kaggle T4 + Unsloth, 7B | Contradiz a premissa "hardware enxuto"; sessões caem. Mantido como **plano B** (seção 12) |
| CPU-only | 16GB RAM e sem GPU: inviável para treino de qualquer tamanho útil |

## 5. Modelo base

| Papel | Modelo | Licença | Uso |
|---|---|---|---|
| Smoke test | `Qwen/Qwen3-0.6B` | Apache 2.0 | 100 steps, validar pipeline |
| Primário | `Qwen/Qwen3-1.7B` | Apache 2.0 | Primeira rodada completa |
| Stretch | `Qwen/Qwen3-4B-Instruct-2507` | Apache 2.0 | Segunda rodada, só se VRAM/tempo permitirem |

Critérios: licença permissiva (publicação no Hub), suporte PT/EN, vocabulário e
tamanho que cabem em 6GB com margem.

Passo obrigatório no plano: antes de baixar, verificar se surgiu modelo pequeno mais
recente com licença permissiva e suporte a PT. Se sim, avaliar troca; a decisão fica
registrada no README.

**Decisão registrada (2026-09-11):** já existe a família `Qwen3.5` (2B/4B) em Apache 2.0,
mas são modelos **multimodais com atenção linear híbrida** (`Qwen3_5ForConditionalGeneration`,
vocab 248k, camadas de visão) — arquitetura nova, com suporte imaturo em `transformers`
pinado, `bitsandbytes` e `llama.cpp`, e vocabulário maior que piora o orçamento de VRAM em
6GB. Por isso a base primária continua **Qwen3-1.7B** (texto puro, vocab 151k, suporte
universal e comprovado em GGUF). Qwen3.5 fica anotado como experimento futuro no README.

### Orçamento de VRAM (Qwen3-1.7B, seq 768, batch 1)

| Item | ~GB |
|---|---|
| Corpo em NF4 | 0.8 |
| Embeddings / lm_head em fp32 (não quantizados) | 1.2 |
| LoRA + estados do otimizador | 0.3 |
| Ativações com gradient checkpointing | 0.5 |
| Pico de logits + loss | 0.4 |
| **Total estimado** | **~3.2** |

O pico de logits é baixo porque a TRL usa por padrão `loss_type="chunked_nll"`: a projeção do
`lm_head` é feita só nos tokens não-mascarados, em blocos, mantendo vivo apenas
`chunk_size × vocab` de logits por vez. Sem isso, o vocab de 151k dominaria a memória.
Para o 4B, seq cai para 512 e o run só acontece se o pico medido no 1.7B ficar abaixo de 5GB.

## 6. Ambiente

Desktop, dentro do WSL2 Ubuntu:

- Driver NVIDIA instalado **no Windows** (≥ 535, com suporte WSL). Nenhum driver dentro do WSL. `nvidia-smi` funcionando no WSL é o critério.
- Python 3.11 gerenciado por `uv`, venv isolado, deps pinadas em `pyproject.toml`.
- `torch` do índice `https://download.pytorch.org/whl/cu126`. Verificar que `torch.cuda.get_arch_list()` contém `sm_60` **ou** `sm_61` (o `sm_60` cobre a 1060). `dtype=torch.float32` no `from_pretrained` (kwarg `dtype`; `torch_dtype` está deprecado na `transformers` 5.x).
- `bitsandbytes` pinado numa versão recente conhecida (ex.: 0.48–0.50), validada no smoke test.
- `transformers`, `peft`, `trl`, `datasets`, `accelerate` pinados na versão vigente no início da implementação.
- `ollama` instalado no Windows do desktop (usa a GPU nativamente) para a geração sintética.
- `llama.cpp` clonado e compilado no WSL (só CPU basta) para conversão e quantização.

`scripts/00_check_env.py` é o gate de entrada. Ele:

1. Confirma CUDA disponível, nome da GPU, `sm_61` na arch list.
2. Carrega `Qwen3-0.6B` em NF4 e roda um forward+backward.
3. Roda 50 steps de treino fake (seq 768, batch 1) e reporta tok/s e pico de VRAM.
4. Falha explicitamente se qualquer etapa quebrar. Sem "provavelmente funciona".

## 7. Dados

### Formato canônico

JSONL, uma conversa por linha:

```json
{"messages": [
   {"role": "system", "content": "<system prompt fixo>"},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "..."}
 ],
 "source": "trendyol|synthetic-attack|synthetic-owasp|...",
 "lang": "en|pt"}
```

O chat template do modelo é aplicado no treino, nunca gravado no arquivo. System prompt
fixo e curto, definindo o papel (assistente de segurança ofensiva autorizada e defesa)
e o escopo (metodologia, ferramentas, análise, mitigação).

### Camada A: base pública (~3.000 exemplos, EN)

Fontes validadas no Hugging Face (existência, schema e licença conferidos em 2026-09-11):

- `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` — **apache-2.0**, 53.201 linhas, schema `{system, user, assistant}` em EN. Fonte principal.
- `AlicanKiraz0/Cybersecurity-Dataset-v1` — **apache-2.0**, 2.411 linhas, mesmo schema. Fonte secundária.

Descartados: `Isamu136/penetration_testing_scraped_dataset` (sem licença declarada) e
`Nitral-AI/Cybersecurity-ShareGPT` (não encontrado / 404). Os dois de cima já entregam o
schema exato e volume de sobra para o cap de 3.000.

Filtros, nesta ordem:

1. Idioma = EN (detector simples).
2. Resposta entre 80 e 1.500 tokens.
3. Remove recusas genéricas ("I cannot help", "As an AI") e respostas sem conteúdo técnico.
4. Dedup exato e near-dup (MinHash, threshold 0.8).
5. Amostragem estratificada por tema (recon, web, rede, privesc, defesa/blue, CVE/CWE), tema atribuído por keywords. Cap de 3.000 no total.

### Camada B: sintética (~1.000 exemplos, 70% EN / 30% PT)

- Gerador: `qwen2.5:7b-instruct` ou `qwen3:8b` (Q4_K_M) via Ollama no desktop. Roda **antes** do treino; a GPU não é compartilhada.
- Fontes seed com licença compatível:
  - MITRE ATT&CK (STIX JSON do GitHub oficial): técnicas, sub-técnicas, mitigações.
  - OWASP Web Security Testing Guide (CC BY-SA 4.0).
  - CWE Top 25 (termos de uso do MITRE permitem).
  - Man pages / help de nmap, sqlmap, hydra, gobuster.
- Pipeline por chunk: chunk da fonte (300–800 tokens) → prompt pede 2–3 pares Q/A ancorados
  no chunk, em JSON → validação de schema → filtro de tamanho → checagem de ancoragem
  (a resposta precisa citar ao menos 2 termos-chave do chunk) → dedup contra a camada A.
- Tipos de exemplo (mistura alvo):
  - explicação de técnica/vulnerabilidade (30%)
  - "como testar X em ambiente autorizado", passo a passo com ferramentas (25%)
  - interpretação de saída de ferramenta: saída sintética de nmap/nikto/gobuster gerada por template, seguida de análise (20%)
  - mitigação / visão blue team (15%)
  - fora de escopo ou pedido inadequado → redirecionamento educado (10%)
- Exemplos PT: pergunta em PT-BR → resposta em PT-BR, mesmo gerador, prompt em PT.
- Estimativa: ~300k tokens gerados, 5–6h a ~15 tok/s. Overnight.

### Split e manifesto

- 95% treino / 5% validação, estratificado por `source`. O val nunca é usado em nada além de loss.
- `data/manifest.json`: para cada fonte, URL, licença, contagem bruta, contagem após filtros, seed usada.
- Scripts `01_build_dataset.py` e `02_gen_synthetic.py` com seed fixa. `data/raw/` fica fora do git; `data/processed/` vai para o Hub.
- **Licença do dataset publicado: `cc-by-sa-4.0`**, escolhida como guarda-chuva que satisfaz o share-alike do OWASP WSTG. Um arquivo `NOTICE` credita cada fonte e sua licença original (Trendyol e AlicanKiraz0 apache-2.0; ATT&CK e CWE sob a licença permissiva do MITRE com atribuição; WSTG CC BY-SA 4.0). Manter treino = publicado preserva a reprodutibilidade, e a documentação da diligência de licença é um ponto a favor no portfólio.

## 8. Treino

`scripts/03_train.py`, parametrizado por `configs/{smoke,1.7b,4b}.yaml`.

### Quantização

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
# dtype=torch.float32 (kwarg novo) para embeddings, norms e lm_head
```

`prepare_model_for_kbit_training` + `gradient_checkpointing_enable`.

### LoRA

- r=16, alpha=32, dropout=0.05.
- Alvos: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Adapter salvo em safetensors. Base nunca modificada.

### Hiperparâmetros (config 1.7b)

| Parâmetro | Valor |
|---|---|
| max_seq_length | 768 |
| per_device_batch | 1 |
| grad_accum | 16 |
| lr | 2e-4, cosine, warmup 3% |
| weight_decay | 0 |
| epochs | 2 |
| optim | `adamw_torch` |
| fp16 / bf16 | False / False |
| loss | só nos tokens do assistant (completion-only) |

Config `4b`: seq 512, resto igual. Config `smoke`: 0.6B, 100 steps, 200 exemplos.

### Chat template Qwen3 e máscara de loss

O template do Qwen3 **não** tem blocos `{% generation %}`, então `assistant_only_loss=True`
da TRL não funciona direto. Em vez de editar o template, o `03_train.py` converte cada
exemplo `messages` em par **prompt/completion** no momento do load:

- `prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False)`
- `completion = messages[-1]["content"] + tokenizer.eos_token`

Com dataset prompt/completion, a TRL ativa `completion_only_loss` por padrão (loss só na
resposta). O que o template emite como prefixo de geração é, por definição, o mesmo que o
modelo verá na inferência, garantindo consistência treino/inferência. O smoke test imprime
um exemplo renderizado para conferir o prefixo (incluindo o bloco `<think>` vazio, se o
template o inserir).

### Robustez

- `save_steps` ≈ 30 min de treino, `save_total_limit=3`, resume automático do último checkpoint ao reiniciar.
- Log em CSV + TensorBoard local: loss train/val, lr, tok/s, pico de VRAM. W&B opcional, não obrigatório.
- `eval_steps` no val; parada manual se val loss subir por 2 avaliações seguidas.

### Gates, em ordem

1. `00_check_env.py` passa.
2. Smoke (0.6B, 100 steps) passa: treino roda, checkpoint salva, resume funciona, 1 geração sai coerente.
3. Estimativa de horas do 1.7B recalculada com tok/s real. Run 1.7B.
4. Run 4B só se o 1.7B fechou e o pico de VRAM medido ficou < 5GB.

Estimativa inicial (a substituir por medição): 4k exemplos × ~500 tokens × 2 épocas ≈ 4M
tokens; 1.7B a 100–150 tok/s ≈ 8–11h.

## 9. Avaliação

`scripts/04_eval.py`, sempre base vs. adapter, mesma seed.

1. **MCQ objetivo** — `CyberMetric-500` (JSON no GitHub, schema `{question, answers{A..D}, solution}`;
   fallback `zefang-liu/SecQA`). Zero-shot, pontuação por log-likelihood da letra da alternativa.
   Saída: acurácia + IC bootstrap 95%. Vai para README e model card. **CyberMetric não tem licença
   de redistribuição** (só pedido de citação), então é baixado em tempo de avaliação e **nunca
   entra no dataset publicado**; o paper é citado no README. Expectativa honesta: ganho pequeno;
   SFT de 4k exemplos ensina foco e estilo mais do que conhecimento.
2. **Val loss / perplexidade** no split de 5%.
3. **Qualitativo fixo** — `eval/prompts.jsonl` com 12 prompts congelados (8 EN, 4 PT): recon, web,
   privesc, interpretação de saída de nmap, mitigação, 1 fora de escopo. Geração com temperatura 0,
   max 400 tokens, base e adapter lado a lado em `docs/samples.md`.
4. **Regressão** — `eval/regression_prompts.jsonl` com 5 prompts gerais (não-segurança, EN/PT) para detectar perda de português ou colapso de tema.

Roda no desktop (~15–20 min para o MCQ em 1.7B fp32).

## 10. Export e publicação

`scripts/05_merge_export.py`:

1. Carrega base em fp16 na CPU, aplica adapter, `merge_and_unload`, salva HF fp16.
2. `llama.cpp/convert_hf_to_gguf.py` → GGUF f16.
3. `llama-quantize` → Q4_K_M e Q8_0.
4. Gera `Modelfile` (system prompt, template Qwen3, `num_ctx 4096`) e roda `ollama create decria-sec`.
5. Teste no laptop em CPU: 3 prompts (2 EN, 1 PT), confirma respostas coerentes.

`scripts/06_push_hub.py`, três repositórios:

| Repo | Conteúdo |
|---|---|
| `<user>/decria-sec-1.7b-lora` | adapter safetensors, adapter_config, model card |
| `<user>/decria-sec-1.7b-GGUF` | Q4_K_M, Q8_0, Modelfile |
| `<user>/decria-sec-dataset` | train/val JSONL, manifest.json, dataset card |

Model card obrigatoriamente cobre: base, licença (Apache 2.0), fontes de dado e licenças,
hiperparâmetros, hardware e horas reais, tabela de eval, amostras, limitações, uso pretendido
(teste autorizado, estudo, CTF) e uso não pretendido.

## 11. Estrutura do repositório

```
finetunning_decria/
  README.md                  # narrativa do projeto + resultados
  pyproject.toml             # deps pinadas (uv)
  .gitignore                 # data/raw, outputs, checkpoints, gguf
  configs/
    smoke.yaml
    1.7b.yaml
    4b.yaml
  scripts/
    00_check_env.py
    01_build_dataset.py
    02_gen_synthetic.py
    03_train.py
    04_eval.py
    05_merge_export.py
    06_push_hub.py
  eval/
    prompts.jsonl
    regression_prompts.jsonl
  data/
    raw/                     # gitignored
    processed/               # train.jsonl, val.jsonl, manifest.json
  docs/
    samples.md
    superpowers/specs/
  outputs/                   # gitignored: checkpoints, merges, gguf
  Modelfile
```

Fluxo: código escrito no laptop, `git push`; desktop faz `git pull` e executa no WSL2.
Nenhum script que use GPU roda no laptop. Checkpoints, merges e GGUFs ficam em `outputs/`
fora do git.

Cada script é independente: recebe caminhos e config por argumento, escreve saída em local
previsível, pode ser rodado isolado. Código compartilhado (leitura de config, formatação
de mensagens, system prompt) vive em um único módulo `scripts/common.py`.

## 12. Riscos e fallbacks

| Risco | Detecção | Resposta |
|---|---|---|
| Wheel `cu126` some ou não inclui sm_61 | `00_check_env` | Usar índice `cu118`; se necessário, `torch` 2.5–2.7 |
| bitsandbytes quebra em Pascal | `00_check_env` | Testar versões 0.47 → 0.45; se nenhuma funcionar, plano B |
| tok/s muito baixo (< 40) | `00_check_env` | Reduzir dataset para 2k, seq 512, 1 época; se ainda inviável, plano B |
| OOM no pico de logits | smoke ou início do run | seq 512; se persistir, 0.6B ou plano B |
| Ollama sem GPU no Windows | geração sintética lenta | Rodar Ollama no WSL; em último caso, gerar menos exemplos (500) |
| Dataset público sem licença clara | validação no passo 1 do plano | Descartar a fonte; compensar com mais sintéticos |
| Queda de energia em run longo | checkpoint | Resume automático |

**Plano B (Colab/Kaggle T4 + Unsloth):** mesmos dados, mesmos scripts 04–06. Só `03_train.py`
ganha uma variante. Documentado no README como desvio, não escondido.

## 13. Critérios de sucesso

1. `00_check_env.py` e smoke passam no desktop.
2. Run 1.7B completa 2 épocas com val loss caindo e sem OOM.
3. MCQ: acurácia do adapter ≥ base (empate técnico aceito e documentado).
4. Amostras qualitativas: respostas EN mais focadas e estruturadas que a base; PT funcional.
5. GGUF Q4_K_M roda no laptop em CPU via Ollama e responde em PT e EN.
6. Três repos no Hub publicados, model card completo, README com horas reais.

## 14. Fases (ordem de execução)

| Fase | Onde | Entregável |
|---|---|---|
| 0. Setup e validação | desktop | `00_check_env` verde, versões pinadas |
| 1. Dados públicos | laptop (CPU) | `train/val.jsonl` parcial, manifest |
| 2. Dados sintéticos | desktop (Ollama, overnight) | camada B mesclada, manifest final |
| 3. Smoke | desktop | pipeline validado, tok/s real |
| 4. Run 1.7B | desktop (overnight) | adapter, logs |
| 5. Eval | desktop | tabela MCQ, samples.md |
| 6. Export | desktop + laptop | GGUF, Modelfile, teste em CPU |
| 7. Publicação | laptop | 3 repos no Hub, README final |
| 8. (Opcional) Run 4B | desktop | repetir 4–7 |
