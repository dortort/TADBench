# Source attribution and license notices

TADBench transforms and combines third-party datasets with locally generated
examples. This collection is not offered under one blanket MIT license.
The `source` field identifies the origin of each example. Changes include field
normalization, label/attack-family mapping, word-count metadata, UUID assignment,
local splitting where documented, and composition of evaluation subsets.

| Source | Authors / attribution | Declared upstream terms |
|---|---|---|
| [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Dennis Jacob, Hend Alzahrani, Zhanhao Hu, Basel Alomair, David Wagner; *PromptShield: Deployable Detection for Prompt Injection Attacks* (2025) | Apache-2.0 |
| [deepset prompt injections](https://huggingface.co/datasets/deepset/prompt-injections) | deepset; original dataset card retained | Top-level card: Apache-2.0; nested `dataset_info.license`: CC-BY-4.0 |
| [SPML](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | Reshabh K Sharma, Vinayak Gupta, Dan Grossman; *SPML: A DSL for Defending Language Models Against Prompt Attacks* (2024) | MIT |
| TADBench synthetic mismatch and adversarial examples | [TADBench contributors](https://github.com/dortort/TADBench) | MIT; see `TADBench-LICENSE.txt` |

The deepset card contains conflicting license metadata. Both declarations are
preserved here and in `provenance/deepset-README.md`; this release does not
resolve that discrepancy or relicense deepset's data. Preserve attribution and
the source notices when redistributing derived copies.

Upstream cards are archived at the exact revisions in `manifest.json`, including
their citations, limitations, and disclaimers. Source-specific terms continue
to apply to normalized rows and combined subsets, including the threat bank.

License references:

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
- [MIT License](https://opensource.org/license/mit)

Complete Apache-2.0 and CC-BY-4.0 texts are included under `licenses/` in the
export. The SPML card's MIT declaration and authorship are retained verbatim in
`provenance/spml-README.md`; TADBench's own copyright/license notice is retained
separately in `TADBench-LICENSE.txt`.
