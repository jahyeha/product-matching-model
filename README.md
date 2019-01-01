
# product-matching-model

2018 [SNU-FIRA AI Agent Course](http://bdi.snu.ac.kr/academy/portal/index.php/ai_intro/), Final Capstone Project (18.11.06-12.14)

This project is supported by **Seoul National University& Ministry of Employment and Labor**.

🏅Our team won **1st place** in the final project presentation.

## Goal

#### ▪ Project Background
<img src="https://github.com/jahyeha/product-matching-model/blob/master/img/intro.jpg" width="85%">

#### ▪ 딥러닝을 활용한 상품매칭 모델 개발
　(Building Product Matching Models for an Ecommerce Platform(KOR) using Deep Learning Methods.)

## Team Members
<img src="https://github.com/jahyeha/capstone-project/blob/master/img/members.png" width="28%">

- [Jahye Ha](https://github.com/jahyeha) 
- [Yoonna Jang](https://github.com/YOONNAJANG) 
- [Yongju Ahn](https://bhi-kimlab.github.io/) (T.A.)

## Requirements
Initial requirements are as follows.
```
 python 3.6.5
 gensim 3.6.0
 keras 2.2.4
 tensorflow 1.12.0
 numpy 1.15.4
 pandas 0.23.4
```

## Result 
<img src="https://github.com/jahyeha/capstone-project/blob/master/img/res.png" width="100%">

```
*---main.py---*
✔ | 히말라야 고수분크림 (인텐시브) 150ml ===> 인텐시브 모이스처라이징 크림@150ml
✔ | [HANYUL]어린쑥 수분 진정크림 50ml ===> 어린쑥 수분 진정크림@50ml
✔ | 마몽드 모이스처 세라마이드 인텐스 크림 ===> 모이스처 세라마이드 인텐스 크림@50ml
✔ | 토니모리 더 촉촉 그린 티 수분 크림 60ml ===> 더 촉촉 그린티 수분 크림@60ml
✔ | 아이오페 더마 리페어 시카크림 ===> 더마 리페어 시카크림@50ml 
✔ | 헤라 아쿠아볼릭 하이드로-젤 크림 50ml ===> 어린쑥 수분진정 젤@250ml
✖ | 히말라야 정품-50ml 히말라야인텐시브수분크림/히말라야인텐시 ===> 너리싱 스킨 크림@50ml | ✪정답: 인텐시브 모이스처라이징 크림 ...
```

## 

### Notice

░░░░CLOSED DATASETS░░░░ 프로젝트 협력기관과의 비밀 유지 협약으로 데이터를 공개할 수 없습니다.

### References
- Joulin, Armand, et al. "Fasttext. zip: Compressing text classification models." arXiv preprint arXiv:1612.03651 (2016).
- Shah, Kashif, Selcuk Kopru, and Jean David Ruvini. "Neural Network based Extreme Classification and Similarity Models for Product Matching." Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 3 (Industry Papers). Vol. 3. 2018.
- [How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
