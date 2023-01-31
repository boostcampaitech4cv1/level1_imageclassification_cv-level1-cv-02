# 마스크 착용 상태 분류 대회

## Competition Overview

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

### 문제 정의

- 우리가 풀어야 할 문제는?
    - Input이 주어졌을 때 성별, 나이, 마스크 착용 여부에 따른 클래스를 정의하여 반환해야 합니다.
- Input / Output
    - Input : 이미지, Output : 성별, 나이, 마스크 착용 여부에 따른 18개의 클래스
- 우리가 문제를 푼 방식
    - one vs all 전략(하나의 모델로 multi label classification)
    - one vs one 전략(여러개의 모델로 multi class classification)

### 프로젝트 팀 구성 및 역할
  - 박시형 : 나이 분류
  - 정혁기 : 마스크 분류
  - 김형석 : 성별 분류
  - 노순빈 : 전체 분류
  - 장국빈 : 전체 분류
  
## EDA

전체 이미지 수 : 4500    
한 사람당 사진의 개수 : 7 (마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장)  
이미지 크기 : [384, 512]  

### Class Discription
<p align=center>
<img width="500" src="https://user-images.githubusercontent.com/77565951/215813850-9b7172b0-e4a0-4e65-a0e4-27be6ac84bd8.png"/>
</p>

- Labeling
    - Mask
        - Wear : 마스크를 쓴 경우
        - Incorrect : 마스크를 이상하게 쓴 경우 (코스크, 턱스크 등)
        - Not Wear : 마스크를 쓰지 않은 경우
    - Gender
        - Male : 남자
        - Female : 여자
    - Age
        - < 30 : 나이가 30세 미만
        - ≥30 and < 60 : 나이가 30세 이상 60세 미만
        - ≥60 : 나이가 60세 이상

### 나이분포
<img width="1034" alt="스크린샷 2022-11-02 오후 4 11 51" src="https://user-images.githubusercontent.com/77565951/215831085-b491a52d-421a-470d-9c2c-b774a6f0f534.png">
나이 카테고리 3개로 나누어 시각화
<img width="1034" alt="스크린샷 2022-11-02 오후 4 13 16" src="https://user-images.githubusercontent.com/77565951/215831058-4c5d9794-c3a5-490e-a7ce-10f5720fe285.png">

### 이미지 RGB 분포
<img width="1034" src="https://user-images.githubusercontent.com/77565951/215832620-38b4387e-c14c-40cc-ad5d-23365b654a4d.png"/>

### 최종 class 분포
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/77565951/215833124-e1edbe28-c64b-41e2-965f-0f0d969a3400.png"/>
</p>

### class별 픽셀 평균 이미지
<img src="https://user-images.githubusercontent.com/77565951/215833554-ea55713f-8369-475b-a085-e346127faa13.png"/>

### 전체 이미지 픽셀 평균 이미지
<p align=center>
<img src="https://user-images.githubusercontent.com/77565951/215833632-d2a2997b-2460-4925-9ccc-b5ec72545396.png"/>
</p>

## Strategy
- Data over-sampling : 클래스 이미지 개수가 적은 60대 이상 이미지를 복제하여 분포 맞춤
- Mask 분류 모델
  - Focal Loss
  - pretrained_ResNext50
  - F1-score : 0.98+
  - Best model 찾는 기준을 classification_report를 이용, 여러 F1-score 기준으로 고름
  
 - 나이 분류 모델
    - Data augmentation : 사진에서 보통 사람들이 중앙에 위치하기 때문에 center crop을 통해 얼굴에 집중할 수 있도록 함
    - 나이를 3개의 카테고리로 나눠서 진행
    - resnet101, resnext, regnet_x_16f, efficientnet_b3
    - F1Loss, CrossEntropyLoss
    - CutMix 적용했으나 성능 저하
    - SGD, ADAM 사용해서 실험
    - 앙상블 적용
    - Griddropout 사용하여 머리카락 등 여러 일반화 성능이 낮아질 가능성을 낮추고자 함
    - 나이 class의 범위를 바꿔보면서 비교(성능 차이 없음)
 
 - 성별 분류 모델
    - Pretrained ResNext(224x224)
    - No Augmentation, BCE Loss, ADAM, lr 0.001, Epoch20
    - accuracy : 98%
 
 - 전체 분류 모델
    - Pretrained ViT(384x384), SGD, Crop((50,50),(334,400))
    - F1 Score : 0.7417
    - Accuracy : 79.7460%

### Hard Voting
---
## Setup

1. Run DataDownloader.ipynb
2. Run DataMaker.ipynb

## Submission

Evaluation data에 대한 inference 값을 submission.csv로 만들어 제출

* column format : ImageID,ans

>a41280fcf20d5bb68550876c36b63e9d030b2324.jpg,0
2b721e63790fd041b5440f05647afc9266fa05bd.jpg,0
3931a1e7ee6fd45f313436ab68d0f556a25e4d25.jpg,0
4b30021def42c080bb7744d15b50b3a381d9cb4f.jpg,0
b57eca823bdbf49272c75354bf0e0d3d8fc119d7.jpg,0
075bbf401dd04ad6154bc508875d8910e08116e8.jpg,0
f740f5a296b8d5331ae47907bd51126bb0e70697.jpg,0
c90c83435d9c8c3f1fdc3ee6c7f65478f72f4967.jpg,0
b7e19ad2552e1f27a4b2c8a93c1284bfa2d5176e.jpg,0
b5930a21e7a24290cbf63cc93798705cfce09d16.jpg,0
b862db486a2f118412858a1369ef30b14b90cbee.jpg,0
d80bd7fff0e6bfc64092fac7a20c14ecdb8fda3f.jpg,0
...

## Evaluation

F1 Score 사용

![img](https://forums.fast.ai/uploads/default/original/3X/c/c/cca1b3ad72fc927fbf3d3690f01d2e3b5a31dd2e.png)

---

## Rule

1. 변수명 : snake_case
2. 함수명 : camelCase
3. 클래스 : PascalCase
4. utils 함수 작성시 annotation 작성!
5. docs 작성 권장.  
6. wandb 사용.  
7. model 이나 train 함수 따로 python 파일로 작성해서 폴더에 넣기.  
8. 모델 파일 이름 backbone명_추가내용_이름이니셜.py.  
