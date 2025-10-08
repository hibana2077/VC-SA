# **ViT-GraphSampler：從影像預訓練出發的「可學式影格／Token 共選」與輕量時序 GNN 記憶庫，用於高效率影片分類**

> 一句話：用**影像預訓練的 ViT**當骨幹，學會在**影片層面同時選影格與選 token**，再把這些關鍵 token 串成**輕量時序圖（Graph）與記憶庫**做消息傳遞，達到**接近 VideoMAE/TimeSformer 的精度、遠低於其計算量**的影片分類。

---

## 研究核心問題

在**只用影像預訓練的 ViT**（如 DINOv2／DeiT/ViT-21k）前提下，如何在**有限計算預算**內，把影片中真正**時空關鍵**的資訊保留下來，並以**比全域時空注意力更便宜**的方式完成時序建模與分類？現有方法多半：

* 直接進行**全域或分解式時空注意力**（如 TimeSformer／Video Swin），仍昂貴；或
* 在**影格或 token**做**單一面向的動態稀疏化**（如 AdaFrame、DynamicViT/TokenLearner），但缺少與**時序記憶**的緊密耦合。 ([arXiv][1])

## 研究目標

1. 在**固定 FLOPs/延遲**預算下，提升 Kinetics-400、Something-Something V2 的 Top-1。
2. 在**長片段或稀疏關鍵動作**（如 Diving48、EG0-視角 EPIC-KITCHENS）上維持或提升精度。
3. 與 Token/Frame 稀疏化與 ToMe/Video ToMe 類方法相比，**以更低 token 數**達到同級精度。 ([arXiv][2])

## 貢獻

* **共選（co-selection）機制**：提出**可學式影格＋token 同步選擇器**，在**影格內**做 token 採樣、在**時間軸**挑選關鍵影格，兩者**共享一個效用（utility）目標**。相較僅選影格（AdaFrame）或僅選 token（DynamicViT/TokenLearner），能更精準控制計算。 ([CVF Open Access][3])
* **輕量時序 GNN 記憶庫（Graph-Based MemBank）**：將被保留的關鍵 token 作為節點，依**時序鄰近＋特徵相似**連邊，採**TGN 風格的記憶更新**做跨影格消息傳遞，複用**長期特徵庫（長期記憶）**的概念。 ([arXiv][4])
* **理論上界**：把共選視為**受限資訊瓶頸**（budgeted IB）問題，給出**拉格朗日放鬆**下的選取規則與**近似誤差—複雜度**權衡；再證明所建圖的**kNN 消息傳遞**可視為對**全域時空注意力**的低秩／稀疏近似，複雜度由 (O(TN^2)) 降為 (O(TNk))。 ([arXiv][5])
* **實證**：在多資料集上達成**更好的精度–效率曲線**，同時展示對**影像預訓練權重（DINOv2/IN21k）**的強韌遷移。 ([arXiv][6])

## 創新

1. **影格–token 共同決策**：把「看哪幾格、留哪幾個 token」統一成**同一個可微分目標**，不同於以往分開設計。 ([CVF Open Access][3])
2. **圖式時序建模取代重型注意力**：以**TGN 記憶＋GAT/鄰居聚合**近似時空關係，對長影片更友善。 ([arXiv][4])
3. **與 ToMe/VTM 互補**：我們的共選在**前端過濾**，ToMe/Video Token Merging 在**中段合併**，兩者可疊加。 ([arXiv][7])

## 理論洞見（精要）

* **命題 1（共選＝受限 IB 的拉格朗日最優策略）**：
  在總計算預算 (B) 下，最大化輸出與輸入子集的互資訊，可寫為 (\max_{S} I(Y;X_S)-\lambda|S|)。對每個影格與 token 的**重要性得分**收斂到該拉格朗日目標的**貢獻估計**；以**Gumbel-Softmax**給出可微近似的 Top-k 選擇。*（證明略）* ([arXiv][8])
* **命題 2（圖消息傳遞 ≈ 稀疏注意力）**：
  對被選 token 建 (k)-NN 圖並做 (L) 層消息傳遞，相當於在注意力矩陣上施加**稀疏遮罩／低秩近似**；在圖連通與混合條件下，對全域注意力的譜距離有上界，時間複雜度由 (O(TN^2)) 到 (O(TNk))。*（以 Linformer/Performer/Nyström 的近似觀點輔佐）* ([arXiv][5])

## 方法論

**Backbone**：凍結或微調**影像預訓練 ViT**（DINOv2 或 IN-21k 權重）。 ([arXiv][6])

**(A) Learnable 影格／token 共選**

* 影格層級：以輕量 policy（MLP＋全域池化）估計未來效用，參考 AdaFrame 思想但改為**可微分（Gumbel-Softmax）**而非純 RL。 ([CVF Open Access][3])
* 影格內 token 層級：以**DynamicViT/TokenLearner**式的重要度預測＋連續 Top-k 近似，輸出保留 mask。 ([NeurIPS Papers][9])

**(B) Graph-Based MemBank（輕量時序 GNN）**

* 節點＝被保留 token；邊＝時間近鄰＋語義 kNN。
* 記憶更新採**TGN**：對每個 token id 維護 memory state，訊息由相鄰節點聚合；GAT/加權平均作為消息函數。 ([arXiv][4])

**(C) 分類頭**：對最後一層記憶化節點做池化（CLS readout）→ 全連接分類。

**複雜度**：理論上 **(O(TN^2)\to O(TNk))**；實作上比 TimeSformer/Video Swin 的 divided attention/區域注意力更省記憶體於長序列情境。 ([arXiv][1])

## 數學推演與（草）證

1. **受限 IB 表述**：(\max_{S} I(Y;X_S)) s.t. (|S|\le B)；拉格朗日 (\mathcal{L}=I(Y;X_S)-\lambda |S|)。令每個元素 (e) 的**邊際貢獻** (\Delta_e) 估計為梯度／代理損（cross-entropy 減損）；用**Gumbel-Softmax**近似 one-hot 選擇，溫度退火至近似 Top-k。*（參見 Concrete/Gumbel-Softmax 之可微抽樣）* ([arXiv][10])
2. **子模性近似與貪婪保證（直觀）**：當 (\Delta_e) 由**覆蓋／設施位置**等項構成時具**遞減報酬**特性，可借用子模最大化的 (1-1/e) 近似保證（為理論動機，實作仍端到端學習）。*（參考子模選取文獻）* ([機器學習研究期刊][11])
3. **圖消息傳遞與注意力近似**：設全域注意力矩陣 (A)；以 Nyström/線性注意力觀點，把 (A\approx \tilde A)（低秩或核近似）。我們的圖稀疏遮罩對應將注意力限制在**局部鄰域**，其圖拉普拉斯 (L) 與 (A) 的譜差界與 (k)、節點覆蓋率相關，據此推得**誤差–成本**權衡。*（以 Linformer/Performer/Nyström 作理論支點）* ([arXiv][5])

## 預計使用資料集

* **Kinetics-400**（通用人類動作）。 ([arXiv][2])
* **Something-Something V2**（需強時序推理）。 ([Hugging Face][12])
* **Diving48**（稀疏關鍵片段、細粒度動作）。 ([UC San Diego Service][13])
* **EPIC-KITCHENS-100**（自我視角、長期依賴）。 ([EPIC Kitchens][14])

## 與現有研究之區別

* 相對 **TimeSformer／Video Swin**：我們以**圖消息傳遞＋記憶**近似時空關係，避免全域注意力成本。 ([arXiv][1])
* 相對 **AdaFrame**：不只選影格，也**同時**選 token，且**可微**訓練。 ([CVF Open Access][3])
* 相對 **DynamicViT／TokenLearner**：不只做**空間稀疏**，還結合**時序記憶**與**跨格圖**。 ([NeurIPS Papers][9])
* 相對 **ToMe / Video Token Merging（VTM）**：ToMe/VTM 側重**合併相似 token**；我們在前端做**選擇與記憶**，兩者可疊加。 ([arXiv][7])
* 只用**影像預訓練**（如 DINOv2／IN-21k）起步，證明不依賴昂貴影片預訓練（對比 VideoMAE）。 ([arXiv][6])

## 實驗設計

**骨幹與權重**：ViT-B/16（DINOv2 或 IN-21k）作主幹；所有方法共用同起點。 ([arXiv][6])

**比較方法**：

* 全域/分解注意力：TimeSformer、Video Swin。 ([arXiv][1])
* 影格選擇：AdaFrame。 ([CVF Open Access][3])
* token 稀疏：DynamicViT、TokenLearner；再加**ToMe/VTM**做後處理。 ([NeurIPS Papers][9])

**評估指標**：Top-1/Top-5、GFLOPs、吞吐（clips/s）、端到端延遲、記憶體峰值、能耗（選擇性）。

**Ablation**：

* 影格選擇器 vs token 選擇器；
* kNN 圖的 (k)、層數 (L)、記憶尺寸；
* 不同預算 (B) 的**精度–效率曲線**（Pareto 前緣）；
* 與 ToMe 疊加的增益；
* 不同預訓練（DINOv2 vs IN-21k）遷移性。 ([arXiv][6])

**長片段測試**：把影片時長拉長（或用有長期依賴的子集），比較本法與 TimeSformer 的記憶體與延遲。 ([arXiv][1])

**訓練細節**（建議）：

* clip 採樣：訓練時使用多尺度時間抖動；測試 1×或 3× 視 budget。
* 損失：分類交叉熵＋**稀疏正則**（對選擇器）＋**一致性正則**（共選在不同增強下輸出一致）。
* 選擇器溫度退火（Gumbel-Softmax）。 ([arXiv][8])

---

如果你想直接動手做，我可以幫你把**方法段落**轉成**最小可行原型（PyTorch 版）**的模組接口與訓練腳本架構，包含：共選器 head、TGN-style 記憶庫層、以及 TimeSformer/Video Swin/TokenLearner/ToMe 的對照訓練配置。

[1]: https://arxiv.org/abs/2102.05095 "Is Space-Time Attention All You Need for Video Understanding?"
[2]: https://arxiv.org/abs/1705.06950 "[1705.06950] The Kinetics Human Action Video Dataset"
[3]: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html "AdaFrame: Adaptive Frame Selection for Fast Video Recognition"
[4]: https://arxiv.org/abs/2006.10637 "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
[5]: https://arxiv.org/abs/2006.04768 "Linformer: Self-Attention with Linear Complexity"
[6]: https://arxiv.org/abs/2304.07193 "[2304.07193] DINOv2: Learning Robust Visual Features ..."
[7]: https://arxiv.org/abs/2210.09461 "Token Merging: Your ViT But Faster"
[8]: https://arxiv.org/pdf/1611.01144 "arXiv:1611.01144v5 [stat.ML] 5 Aug 2017"
[9]: https://papers.neurips.cc/paper_files/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf "DynamicViT: Efficient Vision Transformers with Dynamic ..."
[10]: https://arxiv.org/pdf/1611.00712 "The Concrete distribution"
[11]: https://www.jmlr.org/papers/v21/19-467.html "Submodular selection for data summarization in Python"
[12]: https://huggingface.co/datasets/HuggingFaceM4/something_something_v2 "HuggingFaceM4/something_something_v2 · Datasets at ..."
[13]: https://www.svcl.ucsd.edu/projects/resound/dataset.html "Diving48 Dataset"
[14]: https://epic-kitchens.github.io/ "EPIC-KITCHENS Dataset"
