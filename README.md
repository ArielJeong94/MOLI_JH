# 🧬 MOLI materialization & Extension

Multi-Omics 기반 항암제 반응 예측 모델 구현 및 성능 개선 프로젝트

> Multi-omics 데이터를 활용하여 암세포의 약물 반응을 예측하는 딥러닝 모델 MOLI를 직접 구현하고,
<br>다양한 학습 전략 시도를 통해 일반화 성능을 개선해 본 프로젝트

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/ArielJ94/egfri-predictor)
[![Velog](https://img.shields.io/badge/Velog-블로그-green)](https://velog.io/@jkh3043/series/논문-구현해보기)

---

## 📌 프로젝트 개요

**MOLI(Multi-Omics Late Integration)** 는 암 환자의 유전체 프로파일을 기반으로 
<br>EGFRi 항암제의 반응(Sensitive / Resistant)을 예측하는 딥러닝 모델입니다.

이 프로젝트는 Sharifi-Noghabi et al. (2019) 논문을 직접 구현하는 것에서 시작하여, 
<br>**Drug Embedding 도입**, **하이퍼파라미터 최적화**, **클래스 불균형 처리**, **오분류 분석**, **모델 서비스 배포**까지 수행한 개인 실습 프로젝트입니다.

- **서비스 데모**: [ariel-jeong94-portfolio.vercel.app](https://ariel-jeong94-portfolio.vercel.app/) → 모델 서비스 탭
- **추론 서버**: [HuggingFace Spaces](https://huggingface.co/spaces/ArielJ94/egfri-predictor)
- **개발 블로그**: [Velog 시리즈](https://velog.io/@jkh3043/series/논문-구현해보기)

---

## 🤖 하이퍼파라미터 & 모델 성능 정리
<img width="1542" height="186" alt="image" src="https://github.com/user-attachments/assets/e65cd945-3724-48f5-9072-bf4ed227f2d4" />
<img width="1589" height="732" alt="image" src="https://github.com/user-attachments/assets/e357b681-cb3f-48a6-9af0-cd9c35efe373" />

---

## 🗂️ 프로젝트 구조
```
MOLI_JH/
├── notebook/
│   ├── MOLI_Pandrug_EDA.ipynb          # 데이터 EDA 및 전처리
│   ├── MOLI_Pandrug_EDA_v2.ipynb       # VarianceThreshold 추가
│   ├── MOLI_pandrug_modelA.ipynb       # 기본 구현 (WeightedRandomSampler)
│   ├── MOLI_pandrug_modelB.ipynb       # Drug_id embedding 제거 실험
│   ├── MOLI_pandrug_modelC.ipynb       # VarianceThreshold 적용 데이터 사용
│   ├── MOLI_pandrug_modelD.ipynb       # siames-triplet selector 사용
│   ├── MOLI_pandrug_modelE.ipynb       # SMOTE 적용 실험
│   ├── MOLI_pandrug_modelF.ipynb       # SMOTE + Focal Loss 실험
│   └── MOLI_pandrug_modelG.ipynb       # 최종 모델 (WRS + Focal Loss)
└── outputs/
    ├── EGFRi/
    │   ├── model_G.pth                 # 최종 모델 가중치
    │   ├── model_config.json           # 모델 구조 및 하이퍼파라미터
    │   ├── gene_config.json            # 학습에 사용된 유전자 목록
    │   ├── scaler_params.json          # StandardScaler 파라미터
    │   └── gene_symbol_map.json        # ENTREZ ID → Gene Symbol 매핑
    └── figures/                        # 학습 곡선, EDA 시각화
```

---

## 🧬 데이터셋

| 데이터셋 | 설명 | 역할 | 샘플 수 |
|---|---|---|---|
| **GDSC** | 암세포주 × 약물 반응 데이터 | Train | 3,258 |
| **PDX** | 환자 유래 이종이식(마우스) 모델 | Validation | 81 |
| **TCGA** | 실제 암환자 유전체 데이터 | External Validation | 15 |

**데이터 출처**: [Zenodo - MOLI Dataset](https://zenodo.org/records/4036592)

**입력 오믹스 3종**:
- **Expression** (18,232 genes): 유전자 발현량, StandardScaler 정규화
- **Mutation** (14,447 genes): 체세포 변이 이진값 (0/1)
- **CNA** (20,502 genes): 복제수 변이 (-1/0/1)

**타겟 약물 (EGFRi 5종)**:
`Cetuximab`, `Afatinib`, `Erlotinib`, `Gefitinib`, `Lapatinib`

---

## 🏗️ 모델 아키텍처

```
Expression (18,232) → OmicsEncoder → z_expr ( 32) ─┐
Mutation   (14,447) → OmicsEncoder → z_mut  (  8) ─┼─→ Concat (184) → L2 Normalize → Classifier → (R/S)
CNA        (20,502) → OmicsEncoder → z_cna  (128) ─┤
Drug ID    (5 types) → Embedding   → z_drug ( 16) ─┘
```

**핵심 설계 결정**:

1. **Late Integration**: 각 오믹스 타입별 독립 인코딩 후 결합 → 각 오믹스 고유 분포 보존
2. **Drug Embedding**: 논문 원본 대비 추가 도입 → 5개 약물을 단일 Pan-drug 모델로 학습
3. **Combined Loss**: `γ × TripletMarginLoss + FocalLoss`
   - Triplet Loss: Sensitive 샘플끼리는 가깝게, Resistant와는 멀게 embedding space 구성
   - Focal Loss (α=0.6, γ=1.5): 클래스 불균형(S:R = 1:5.4) 처리

---

## 🔬 실험 과정 및 주요 의사결정

### 1단계: 논문 재현 시도

논문 하이퍼파라미터(Table S3)를 그대로 적용하여 학습.

| 시도 | 변경 사항 | PDX Cetuximab AUROC |
|---|---|---|
| AllTripletSelector | 논문 원본 구현 | 0.709 |
| HardTripletSelector | Easy Triplet 희석 방지 | 0.724 |
| VarianceThreshold(0.05) | 저분산 유전자 제거 | 오히려 하락 |

**결론**: Drug Embedding 도입으로 인해 논문과 동일한 하이퍼파라미터가 최적이 아님을 확인.

### 2단계: 하이퍼파라미터 최적화 (Optuna)

```python
# 탐색 공간
SEARCH_SPACE = {
    'lr_expr', 'lr_mut', 'lr_cna', 'lr_cls',     # Learning Rate
    'weight_decay',                              # Regularization
    'dropout_expr/mut/cna/cls',                  # Dropout
    'hidden_expr/mut/cna',                       # Hidden Dimension (추가)
    'drug_emb_dim',                              # Drug Embedding Dim (추가)
    'gamma', 'margin', 'batch_size'              # Loss & Training
}
```

60 trials × 3-Fold CV, TPE Sampler + Median Pruner 적용.

**Optuna 최적 결과**: 5-Fold CV 평균 AUROC **0.7258 ± 0.0258**

### 3단계: 클래스 불균형 처리 비교

| 방법 | PDX Cetuximab AUROC | PDX Cetuximab F2 | TCGA AUROC |
|---|---|---|---|
| ModelA: WeightedRandomSampler + BCE | 0.5964 | 0.3125 | 0.4167 |
| ModelF: SMOTE + Focal Loss | 0.6655 | 0.0000 | 0.2917 |
| **ModelG: WRS + Focal Loss** | **0.6145** | **0.3125** | **0.5833** |

**핵심 발견**: SMOTE + Focal Loss 동시 적용 시 이중 과보정(double correction) 발생.
<br>고차원 오믹스 데이터(50K+ features)에서 SMOTE 합성 샘플이 생물학적으로 의미없는 공간에 생성됨을 확인.

### 4단계: 오분류 분석 (Error Analysis)

<img width="1173" height="720" alt="image" src="https://github.com/user-attachments/assets/5729ee8e-15f7-496b-8634-33b2dcfb0ebf" />


Threshold Sweep 결과, **threshold=0.40** 채택:
- 의료 도메인 특성상 FN(치료 기회 손실) > FP(불필요한 치료) 비용
- TCGA F2 = 0.789 달성

---

## 📊 최종 성능 (ModelG)

| 데이터셋 | 약물 | AUROC | Recall | Precision |
|---|---|---|---|---|
| PDX | 전체 | 0.740 | 0.625 | 0.156 |
| PDX | **Erlotinib** | **0.907** | **1.000** | 0.231 |
| PDX | Cetuximab | 0.640 | 0.400 | 0.105 |
| TCGA | 전체 | 0.589 | 0.143 | 0.333 |

> **Threshold = 0.40** (Recall 우선 설계)

---

## 🚀 모델 서비스 아키텍처

```
[포트폴리오 Vercel]
        ↓  POST /gradio_api/upload + /call/predict (SSE)
[HuggingFace Spaces - Gradio]
        ├── 전처리: Gene alignment + Mean Imputation + StandardScaler
        ├── 추론: MOLI forward pass
        └── 시각화:
            ├── Sample × Drug Heatmap
            ├── Best Drug per Sample
            └── Omics Feature Explanation
                ├── Expression: Perturbation Importance Top 10
                ├── Mutation: 변이 빈도 Top 10 Genes
                └── CNA: 증폭/결실 빈도 Top 10 Genes
```

**전처리 파이프라인** (학습 환경과 동일):

```python
# 1. Gene alignment (Train ∩ Val 교집합 기준)
# 2. 누락 gene → Mean Imputation (Expression) / 0-padding (Mutation/CNA)
# 3. StandardScaler (Train fit → transform only)
# 4. 무분산 Mutation gene 제거 (8개)
```

### 결과 반환
#### 1. 예측 테이블
<img width="963" height="522" alt="image" src="https://github.com/user-attachments/assets/67e7751d-5bee-43ea-8719-7b0830f0bac6" />

- Threshold = 0.40 기준으로 Sensitive / Resistant 판정

#### 2. 시각화 패널 해석
<img width="1389" height="868" alt="image" src="https://github.com/user-attachments/assets/b5506372-cdf7-4ea1-a303-9d40f7b7cc25" />

- Heatmap: 샘플 × 약물 민감도 확률을 색상으로 표현. 녹색에 가까울수록 Sensitive 가능성 높음.
- Sensitive Probability by Drug (Boxplot): 전체 샘플의 예측 확률 분포. 중앙값이 0.40 threshold 근처에 위치해 있어 경계 샘플이 많음을 보여줌.
- Best Drug Prediction per Sample: 샘플별 가장 높은 민감도 예측 약물. X-3029, X-3237, X-1658 등이 상대적으로 높은 확률(0.57~0.59)을 기록하여 Erlotinib 반응 가능성이 더 높음을 시사.

#### 3. Omics Feature Explanation 해석
<img width="1568" height="523" alt="image" src="https://github.com/user-attachments/assets/7d7b6635-0f36-4d0d-81bb-9da9dde9b704" />

- Expression — Top 10 Influential Genes (Perturbation Importance)
`CST1, EMID1, PCDHA13, TMEM213, UBXN11 ...`
   - 각 유전자를 0으로 마스킹했을 때 예측 확률 변화량을 측정한 값.
   - 변화량이 클수록 해당 유전자의 발현이 모델 판단에 더 큰 영향을 미쳤음을 의미.

- Mutation — Top 10 Mutated Genes (변이 빈도)
`SERINC2 (0.87), PHLDA1 (0.84), TP53 (0.80), ORAI1 (0.80) ...`
   - 입력 샘플들에서 변이가 자주 관찰된 유전자.
   - TP53은 비소세포폐암(NSCLC)에서 가장 빈번하게 변이되는 유전자로, 이 데이터셋의 생물학적 특성과 일치.
   - EGFR pathway 억제제(EGFRi) 반응과 TP53 변이 상태의 상관관계는 여러 연구에서 보고된 바 있다.

- CNA — Top 10 Copy Number Changed Genes (증폭/결실 빈도)
`MOSPD2, FAM47A, CSPG4P1Y ... (주로 결실 방향)`
   - 입력 샘플에서 복제수 변화가 많은 유전자.
   - 대부분 결실(deletion, 파란색) 방향으로 나타나며, NROB1과 같이 일부 증폭(amplification, 빨간색)도 관찰.

❗️ 해석 주의사항: 
- Expression Perturbation Importance는 모델의 예측 근거를 사후 설명(post-hoc explanation)하는 것으로, 생물학적 인과관계를 직접 의미하지는 않음.
- Mutation/CNA 패널은 입력 데이터의 특성 요약이며 모델 가중치 기반 해석이 아님.

---

## ⚠️ Limitations

| 한계 | 설명 |
|---|---|
| **도메인 갭** | GDSC(세포주) 데이터로 학습 → TCGA(실제 환자) 일반화 제한 |
| **샘플 부족** | TCGA Erlotinib 검증 샘플 3개 → 통계적 신뢰도 낮음 |
| **낮은 Precision** | Recall 우선 설계 + S/R 확률 분포 겹침 → FP 다수 발생 |
| **단일 치료** | 복합 약물 상호작용 미고려 |
| **임상 적용 불가** | 연구/학습 목적 한정 |

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| **ML/DL** | PyTorch, scikit-learn, imbalanced-learn |
| **최적화** | Optuna (TPE Sampler, Median Pruner) |
| **서버** | Gradio, HuggingFace Spaces |
| **프론트엔드** | React (Vite), TypeScript, Tailwind CSS |
| **배포** | Vercel (프론트), HuggingFace Spaces (추론) |
| **시각화** | Matplotlib |

---

## 📖 참고 문헌

- Sharifi-Noghabi, H. et al. (2019). *MOLI: multi-omics late integration with deep neural networks for drug response prediction.* Bioinformatics.
- Schroff, F. et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR.
- Dataset: [Zenodo MOLI](https://zenodo.org/records/4036592)
- Triplet Selector: [siamese-triplet](https://github.com/adambielski/siamese-triplet)

---

## 📝 개발 블로그

전체 구현 과정을 4편의 시리즈로 정리했습니다.

1. [논문 리뷰 — MOLI 핵심 개념 및 아키텍처 분석](https://velog.io/@jkh3043/MOLI-Multi-Omics-data-Late-Integration을-통한-약물-반응-예측-모델)
2. [데이터 준비 및 EDA](https://velog.io/@jkh3043/MOLI-2-MOLI-직접-구현해보기데이터-준비.-EDA)
3. [모델 학습 — 논문 재현 시도 및 한계 분석](https://velog.io/@jkh3043/MOLI-3-MOLI-직접-구현해보기모델-학습편)
4. [하이퍼파라미터 최적화 및 클래스 불균형 처리](https://velog.io/@jkh3043/MOLI-4-MOLI-직접-구현해보기모델-학습편2)

---

<p align="center">
  <i>본 프로젝트는 연구/학습 목적으로 제작되었으며 임상에 적용할 수 없습니다.</i>
</p>
