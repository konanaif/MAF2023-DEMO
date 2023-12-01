# MAF-DEMO
MAF-DEMO는 MAF의 기능을 담고있는 웹데모입니다. 
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/675ab84c-20c3-48fa-bed1-4b3e1d41a7ee)

메인 화면의 구성은 다음과 같습니다. 
- DATA SELECTION: 데이터 선택
- STATE OF THE ART, DESCRIPTION OF SOTA ALGORITHMS: 컨소시엄에서 제안한 알고리즘에 대한 간략한 소개 
- ADVENTAGES: 프레임 워크의 특징 소개 


## About


MAF-DEMO 에는 현재 3개의 tablu 데이터와 1개의 이미지 데이터가 포함되어 있습니다. 또한 14개의 알고리즘을 포함하고 있으며 앞으로도 계속 보완할 예정입니다.
* Data : COMPAS, German credit scoring, Adult census income, Public Figures Face Database(Image)
* Algorithm : Disparate_Impact_Remover, Learning_Fair_Representation, Reweighing, Gerry_Fair_Classifier, Meta_Fair_Classifier, Prejudice_Remover, FairBatch, FairFeatureDistillation(Image only), FairnessVAE(Image only), KernelDensityEstimator, LearningFromFairness(Image only)

## Setup
```bash
git clone https://github.com/konanaif/MAF-DEMO.git
```

## How to use
### 0. Dataset
Sample 폴더를 생성한 후, 해당 폴더에 데이터를 다운로드 합니다. 이 때, MAF 프레임워크와 Sample 폴더, MAF-DEMO의 소스코드 hierarchy는 다음과 같습니다. 
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/daf86714-8f60-4626-9d1a-f3c3b474e9f3)

#### A.	COMPAS
다음 링크를 통해 compas-scores-two-years.csv 파일을 다운로드 합니다. 
- 링크: https://github.com/propublica/compas-analysis/
#### B.	German credit scoring
다음 링크를 통해 german.data 파일을 다운로드 합니다.
- 링크: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
#### C.	Adult census income
다음 링크를 통해 adult.data 파일을 다운로드 합니다.
- 링크: https://archive.ics.uci.edu/dataset/2/adult
#### D.	Public Figures Face Database
Public Figures Face Database 옵션을 클릭하는 경우, 다음 링크에서 자동으로 다운로드를 수행합니다. 
다운로드가 완료되면 Pubfig 폴더, dev_urls.txt, pubfig_attributes.txt,
pubfig_attr_merged.csv 파일이 생성됩니다. 이미지 파일은 Pubfig 폴더 안에 저장됩니다.
- 링크: https://www.cs.columbia.edu/CAVE/databases/pubfig/download/


### 1. Data selection
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/2385e86d-68ff-4fbb-9060-6c0514aacc9d)

샘플 데이터 선택 화면입니다. 현재 Sample 디렉토리에 적합한 파일이 있어야 제대로 실행되며, 데이터는 Preset sample 4가지, Custom dataset 1가지 선택 가능합니다.
* Custom dataset 선택 시 제한사항
  * csv 파일만 업로드 가능하며, 데이터에는 Target, Bias 열이 반드시 하나씩 존재해야합니다.

### 2. Metric
Data 자체 Bias measures와 Base model (SVM) bias measures, T-SNE analysis를 차트와 테이블로 표현합니다.

#### Data metrics
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/d7d86c4c-0e59-4907-823d-19faa75eb7a0)
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/5e3e7483-f10c-4e51-8335-d00b431708aa)


#### Performance
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/417cc3fc-8021-455b-886a-5ce2226051fc)
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/857a1ce0-b9f7-4c5d-a874-0f551506cd06)

#### Classification metrics
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/6a88d634-c432-44fa-a943-ac60661ca9ea)
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/89e574ae-5da1-47f0-9b91-4b17bb349baa)



### 3. algorithm select
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/76e348b9-9d2d-4f11-a0fc-201b4335cdaf)

편향성 완화 알고리즘 선택 화면입니다. ⓘ 버튼을 클릭하여 각 모델에 대한 간략한 설명을 확인할 수 있습니다. 
현재 AIF360의 알고리즘과 컨소시엄에서 개발한 알고리즘을 포함하고 있으며, 향후 추가할 예정입니다. SOTA 알고리즘 중 일부는 Image data로만 활용 가능하며, 현재 Image data는 Pubfig 데이터만 가능합니다. Pubfig 데이터가 아닐 경우 해당 알고리즘들은 disabled 됩니다.


### 4. compare models
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/11ad9fbd-8d83-417b-9f00-2ac7d2533fe8)
![image](https://github.com/konanaif/MAF2023-DEMO/assets/96036352/07d59f75-033e-490b-9eca-fcbadf911c70)

알고리즘을 선택하면 base model 과 mitigated model 간의 결과를 비교합니다.
