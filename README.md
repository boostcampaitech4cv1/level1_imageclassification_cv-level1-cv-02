# 마스크 착용 상태 분류 대회

---
> [AI Stages](https://stages.ai) / [Naver Boost Camp AI Tech 4th](https://www.boostcourse.org)

## 개요
COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

## 제출
test data에 대한 inference 값을 submission.csv로 만들어 제출
column format : ImageID,ans
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

## 평가
F1 Score 사용

![img](https://forums.fast.ai/uploads/default/original/3X/c/c/cca1b3ad72fc927fbf3d3690f01d2e3b5a31dd2e.png)

---

## rule

> 변수명 : snake_case
> 함수명 : camelCase
> 클래스 : PascalCase
> 
> utils 함수 작성시 annotation 작성!
> docs 작성 권장
> wandb 사용
> model 이나 train 함수 따로 python 파일로 작성해서 폴더에 넣기
> 모델 파일 이름 backbone명_추가내용_이름이니셜.py
