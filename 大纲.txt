transformer_project/
│
├── tokenization/
│   ├── bpe.py
│   ├── wordpiece.py
│   ├── sentencepiece.py
│   ├── unigram.py
│   ├── subword.py
│   └── __init__.py
│
├── embedding/
│   ├── nn_embedding.py
│   ├── pretrained_embedding.py
│   └── __init__.py
│
├── position_encoding/
│   ├── sinusoidal.py
│   ├── alibi.py
│   ├── rope.py
│   ├── cope.py
│   └── __init__.py
│
├── attention/
│   ├── standard_attention.py
│   ├── multi_head_attention.py
│   ├── sparse_attention.py
│   ├── long_range_attention.py
│   ├── kv_cache.py
│   └── __init__.py
│
├── ffn/
│   ├── standard_ffn.py
│   ├── moe.py
|   ├── glu.py
│   └── __init__.py
│
├── mask/
│   ├── pad_mask.py
│   ├── sequence_mask.py
│   └── __init__.py
│
├── encoder_decoder/
│   ├── standard_encoder.py
│   ├── standard_decoder.py
│   ├── bert.py
│   ├── gpt.py
│   ├── t5.py
│   ├── bart.py
│   └── __init__.py
│
├── lm_head/
│   ├── standard_lm_head.py
│   ├── adaptive_softmax.py
│   ├── hierarchical_softmax.py
│   └── __init__.py
│
├── training/
│   ├── data_processing.py
│   ├── training_script.py
│   ├── loss_functions.py
│   ├── optimizers.py
│   ├── lr_schedulers.py
│   ├── gradient_clipping.py
│   ├── mixed_precision.py
│   ├── distributed_training.py
│   ├── evaluation_metrics.py
│   ├── hyperparameter_tuning.py
│   └── __init__.py
│
├── inference/
│   ├── greedy_search.py
│   ├── beam_search.py
│   ├── top_k_sampling.py
│   ├── top_p_sampling.py
│   ├── temperature_sampling.py
│   ├── parallel_decoding.py
│   ├── autoregressive_decoding.py
│   ├── non_autoregressive_decoding.py
│   ├── kv_cache_inference.py
│   └── __init__.py
│
├── examples/
│   ├── text_classification.py
│   ├── machine_translation.py
│   ├── text_generation.py
│   ├── summarization.py
│   ├── question_answering.py
│   └── __init__.py
│
└── README.md
