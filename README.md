# JARVIS
Tatung System Technologies Inc JARVIS Demo

 * 目錄結構
<pre><code>
├── README.md
├── train.py             # NeMo fine-tuned code
└── dataset              # 資料集
    ├── train            # 訓練資料
    |   └── wav          # 音檔，格式：16bit 8000HZ 單聲道(Mono)
    ├── validation       # 驗證資料
    |   └── wav          # 音檔，格式：16bit 8000HZ 單聲道(Mono)
    ├── config.yaml      # NeMo quartznet Pipeline
    ├── train.json       # 訓練資料格式
    ├── validation.json  # 驗證資料格式   
    └── vocab.txt        # label
</code></pre>

資料使用Mozilla Common Voice <https://commonvoice.mozilla.org/zh-TW/datasets>，請自行前往下載
