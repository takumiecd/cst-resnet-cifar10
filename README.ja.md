# CIFAR-10向け CST残差CNN

[English](README.md) | 日本語

128→2,048→2,048→10の分類ヘッドを持つ小型残差CNNです。各版を一から全層学習しました。
CST 1層版では中央の2,048×2,048行列を65,536 Atomで表現し、2層版では入力側の
128→2,048も16,384 Atomで置き換えています。

| 方式 | 平均Top-1 | シード17 | シード23 | シード42 | 学習パラメータ数 | 1更新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 87.793% | 87.73% | 87.88% | 87.77% | 4,978,410 | 4.23 ms |
| CST 1層 | 88.040% | 88.04% | 88.11% | 87.97% | 1,046,250 | 5.99 ms |
| CST 2層 | 88.083% | 88.00% | 88.04% | 88.21% | 849,642 | 6.66 ms |
| CST 2層＋1×1 CSTショートカット | 87.923% | 87.68% | 87.98% | 88.11% | 842,676 | 9.08 ms |

CST 2層版は、3シード平均で学習パラメータを**82.9%削減**し、テスト精度が
**0.29ポイント**上がりました。1×1ショートカットは因子化実装を使った追試です。

![3シードの検証精度の平均と範囲](results/learning-curves.png)

[設定と集計](results/summary.json) · [シード別テスト結果](results/test-by-seed.csv) · [更新数ごとの検証結果](results/validation-by-step.csv)

RTX 4090一枚でバッチ256、4,096更新を実行しました。公式学習分割を学習45,000枚と検証
5,000枚に分け、最終チェックポイントを公式テスト10,000枚で評価しました。シード内で画像順、
画像拡張、共通層の初期値を揃えています。畳み込みはBF16、ヘッドはFP32です。更新時間は
元のCUDA学習処理で測った各シードの中央値の平均です。

## 推論

```bash
python -m pip install -r requirements.txt
python infer.py image.png --variant cst-two
python evaluate.py --data data --download --variant cst-two --device cuda
```

[Hugging Face](https://huggingface.co/takumiecd/cst-resnet-cifar10)ではシード17の
Dense、CST 1層、CST 2層を配布しています。推論時は展開済みの行列を取得します。
RTX 4090では、各版についてテスト全10,000枚で元のチェックポイントと同じ出力を確認しました。
同じBF16の畳み込み経路を使うには `--device cuda` を指定します。

2つの `*-atoms.safetensors` にはヘッドの学習済みAtomを物理座標で保存しています。
取得したファイルからは、以下のコマンドでCPU上で行列を復元できます。推論CLIに
`config.json` と復元したファイルのあるディレクトリを指定することもできます。

```bash
python expand_atoms.py cst-two-atoms.safetensors cst-two.safetensors
```
