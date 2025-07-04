# AI4Soccer
This repository presents two significant research contributions in the AI4Soccer domain, advancing the state-of-the-art in football video understanding and commentary generation.

## Key Contributions
### Long Video QA Benchmark for Football Match Understanding
Current visual language models (VLMs) are primarily evaluated on short video clips, lacking comprehensive assessment methods for full-match understanding. We address this gap by introducing a new benchmark:
- Comprehensive Coverage
  - Covers 3 events (goal, corner, yellow card), 2020 QA pairs of 58 matches from 2017-18 Premier League
  - Covers 9 different levels of QA ![alt text](./image.png) 
- Evaluation Methodology: Question-answering tasks designed for complete football matches
- Dataset Organization: Benchmark data and evaluation code available in the Long-Video-QA directory

### Unanonymized Football Commentary Generation
We overcome the limitation of existing football commentary models that only generate anonymized content:

- Fine-tuning Framework: Built on the [TRL](https://github.com/huggingface/trl) framework
- Multi-model Optimization:
  - Domain expert model: MatchTime
  - Domain expert model: UniSoccer
  - State-of-the-art video large language model: Qwen2.5-VL 7B
- Technical Approach: LoRA-based supervised fine-tuning (SFT) to accurately reference player, coach, and club names
- Implementation: Code and models provided in the Unanonymize directory

## Repository Structure

```
AI4Soccer/
├── Long-Video-QA/         # Football long video understanding benchmark
│   ├── Data/              # Data organized by league and season
│   └── Tiny_QA/           # Compact test set
│
└── Unanonymize/           # Football commentary unanonymization
    ├── MatchTime/         # MatchTime model fine-tuning
    ├── UniSoccer/         # UniSoccer model fine-tuning
    └── Qwen/              # Qwen2.5-VL model fine-tuning
```

## Acknowledgements
Many thanks to the code bases from [MatchTime](https://github.com/jyrao/MatchTime/) and [UniSoccer](https://github.com/jyrao/UniSoccer), and dataset from [MatchTime](https://huggingface.co/datasets/Homie0609/MatchTime)