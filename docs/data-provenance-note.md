# Nota de proveniência: AlicanKiraz0/Cybersecurity-Dataset-v1 × Trendyol

Registro do achado que tirou a `AlicanKiraz0/Cybersecurity-Dataset-v1` das fontes ativas do
`01_build_dataset`. Verificado em 2026-09-16 e reconferido em 2026-09-17 a partir das cópias em
cache do HuggingFace (`datasets` 5.0.1, ambas com colunas `system`, `user`, `assistant`).

## Números

| Medida | Valor |
|---|---|
| Linhas brutas, Trendyol | 53.201 |
| Linhas brutas, AlicanKiraz0 | 2.411 |
| Linhas brutas da AlicanKiraz0 com `user+assistant` idêntico a alguma linha do Trendyol | 2.410 / 2.411 |
| Linhas da AlicanKiraz0 que passam nos filtros do pipeline (inglês, tamanho, sem recusa) | 2.390 |
| Dessas, duplicata exata (SHA256 de `user+assistant`) de uma linha do Trendyol | **2.390 / 2.390 (100%)** |
| Linhas da AlicanKiraz0 que sobrevivem ao dedup no build | 0 |
| Valores distintos do campo `system` bruto | Trendyol: 1; AlicanKiraz0: 2 (o das 2.410 linhas pareadas é o mesmo do Trendyol) |

Como o dedup exato mantém a primeira ocorrência e o Trendyol vem antes na lista de fontes, o pool
que chega ao MinHash, ao cap estratificado e ao split é idêntico com ou sem a AlicanKiraz0. Por isso
`data/processed/train.jsonl` e `val.jsonl` não mudam com a remoção: continuam com 2.850 + 150 linhas
`trendyol` na camada pública.

Nenhuma das duas dataset cards no HuggingFace cita a outra. Esta nota não afirma qual dataset derivou
de qual; registra apenas a sobreposição observada e a decisão de não contar a mesma linha duas vezes
nem creditar como contribuinte uma fonte que entrega 0 linhas.

## Como reproduzir

```python
import hashlib
from datasets import load_dataset

t = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train")
a = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-v1", split="train")
h = lambda r: hashlib.sha256((r["user"].strip() + " " + r["assistant"].strip()).encode()).hexdigest()
t_hashes = {h(r) for r in t}
print(sum(h(r) in t_hashes for r in a), "/", len(a))   # 2410 / 2411 no bruto
```

Para o número pós-filtro (2.390 / 2.390), aplique `scripts.build_dataset.normalize_row` a cada linha
da AlicanKiraz0 antes de hashear `user + " " + assistant` das mensagens resultantes.

## Pares idênticos de exemplo

Três pares, escolhidos como as três primeiras linhas da AlicanKiraz0 que passam nos filtros. Os
índices permitem recuperar as linhas completas de cada dataset.

### Par 1

- Índice no `AlicanKiraz0/Cybersecurity-Dataset-v1` (split `train`): 0
- Índice no `Trendyol/...-Instruction-Tuning-Dataset` (split `train`): 7266
- SHA256 de `user.strip() + " " + assistant.strip()` (igual nos dois): `62264b5e217ea59f0b3b617da142d1a94fc3a693af3205d5a5ec2d05a1c16690`
- Campo `system` bruto igual nos dois: sim

**user:**

> Explain the fundamental data flow of a Reflected XSS attack. Describe the journey of the malicious payload from the attacker to the victim's browser, detailing each step, including the role of the web server as a reflection point. What specific trust relationship is being exploited in this process?

**assistant (primeiros 400 caracteres):**

> Reflected XSS attacks occur when an attacker sends a request containing a malicious script to a vulnerable application that is then returned by the server and rendered on the victim’s browser without proper sanitization. The attack begins with the attacker crafting a URL that includes their payload, often embedded within query parameters or other user input fields. This crafted URL is distributed …
### Par 2

- Índice no `AlicanKiraz0/Cybersecurity-Dataset-v1` (split `train`): 1
- Índice no `Trendyol/...-Instruction-Tuning-Dataset` (split `train`): 18520
- SHA256 de `user.strip() + " " + assistant.strip()` (igual nos dois): `e4dec7c6a5b2df368aaf10d00a5392d1c2da61a43efd28a1ce003940e355ca01`
- Campo `system` bruto igual nos dois: sim

**user:**

> How does a Reflected XSS vulnerability in an API endpoint that returns a Content-Type of application/json differ from one in a standard HTML page? Explain the conditions under which a JSON response can be exploited for XSS and describe a potential attack scenario.

**assistant (primeiros 400 caracteres):**

> The difference between reflected cross-site scripting (XSS) vulnerabilities in a web page versus those in an API endpoint with a content type of application/json is primarily due to how browsers interpret and render the returned data. In standard HTML pages, any user-provided input that ends up within the document body can be interpreted as HTML if not properly sanitized. However, JSON responses a…
### Par 3

- Índice no `AlicanKiraz0/Cybersecurity-Dataset-v1` (split `train`): 2
- Índice no `Trendyol/...-Instruction-Tuning-Dataset` (split `train`): 38890
- SHA256 de `user.strip() + " " + assistant.strip()` (igual nos dois): `a8ecbe2eebf1b9604e9abad81ac581abc293074e23a165911790324017284a8b`
- Campo `system` bruto igual nos dois: sim

**user:**

> What is \"DOM Clobbering\" and how can it be used to escalate or enable a Reflected XSS payload? Provide a conceptual example of how an attacker might use clobbering to bypass sanitization logic or create a new vector for script execution.

**assistant (primeiros 400 caracteres):**

> DOM Clobbering occurs when an object property name in the document object model (DOM) is overwritten by a malicious property, allowing the attacker to bypass security checks and execute arbitrary code within the target page. In short, DOM clobbering is about replacing properties of JavaScript objects with attacker-controlled values. The most common scenario is using query string parameters to over…
