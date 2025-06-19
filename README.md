# 비(雨)추다 with 스쿨어택 
AI를 도와 빗물받이에 있는 쓰레기를 찾고 제일 잘 도와준 학교를 가려보자!

## 프로젝트 소개
빗물받이 막힘으로 생기는 침수 피해를 줄이기 위한 초등학생 대상 참여형 교육 시스템

## 개발 기간
### 2025.05.30 ~ 2025.06.16

## 팀원

| [이현지](https://github.com/Merongmerongmerona) | [진소희](https://github.com/soheejin) | [오제현](https://github.com/jaehyun-96) | [김진희](https://github.com/jjjinn-hee) | [김현진](https://github.com/hyzi45) | [이은아](https://github.com/eunalee) |
|:----------------------------------------------:|:------------------------------------:|:---------------------------------------:|:------------------------------------:|:---------------------------------------:|:---------------------------------------:|
| <img src="https://github.com/user-attachments/assets/54843451-8bcb-4bbe-9e6c-eb9c53fdd26e" width=200px alt="_"/>| <img src="https://github.com/user-attachments/assets/687d052e-a421-4645-bec3-25e5a8b612b5" width=200px alt="_"/> | <img src="https://github.com/user-attachments/assets/ffc379cc-29c9-45dd-852f-69bbc7acb4e6" width=200px alt="_"/> | <img src="https://github.com/user-attachments/assets/9869727c-b144-45f1-a8e4-9888c105904e" width=200px alt="_"/> | <img src="https://github.com/user-attachments/assets/2aad4c49-53b4-4db1-b088-898f392857f3" width=200px alt="_"/> | <img src="https://github.com/user-attachments/assets/0d43f5e4-5eec-4ff2-83ea-3b02c45e73d6" width=200px alt="_"/> |
| 데이터 수집 및 전처리<br>모델 제작<br>프로젝트 매니저 | 데이터 수집 및 전처리<br>모델 제작<br>시연영상 및 발표자료 제작 | 데이터 수집 및 전처리<br>모델 제작<br>아이디어 기획 | 데이터 수집 및 전처리<br>모델 제작<br>그래프 시각화 | 데이터 수집 및 전처리<br>모델 제작<br>웹 서비스 개발 | 데이터 수집 및 전처리<br>모델 제작<br>웹 서비스 개발 |


## 사용 기술 스택
[![My Skills](https://skillicons.dev/icons?i=py,opencv,pytorch,azure,git)](https://skillicons.dev)

## 프로세스
### 데이터 수집
직접 수집한 데이터(약 880장) + 외부 데이터(웹 검색)

### 데이터 전처리
이미지 크롭 / 좌우반전 / 랜덤 회전 / 밝기·대비·채도 변화
```python
# 증강 파이프라인
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),         # 랜덤 크롭
    transforms.RandomHorizontalFlip(),         # 좌우 반전
    transforms.RandomRotation(15),             # 회전
    transforms.ColorJitter(0.2, 0.2, 0.2),     # 밝기/대비/채도 변화
])
```

### 모델링
#### 분류 모델
| 항목         | 빗물받이 형태 구분 (격자형 / 비격자형) | 빗물받이 오염도 구분 (깨끗 / 깨끗하지X) |
|--------------|-----------------------------------------|------------------------------------------|
| 모델 학습 횟수 | 4회                                      | 12회                                      |
| 최종 선정 모델 | Iteration 3                              | Iteration 9                               |
| 학습데이터     | Service(400장) + Not Service(400장)     | Clean(519장) + Heavy(638장) + 증강 390장 |
| Precision     | 95.2%                                    | 95.7%                                     |
| Recall        | 98.8%                                    | 95.7%                                     |
| AP            | 99.7%                                    | 99.3%                                     |
| 비고          | 격자형 누락 방지 중요 → Recall 우선     | 깨끗함 예측 정확도 중요 → Precision 우선 |

#### 개체 감지 모델
| 항목           | 담배꽁초 감지 |
|----------------|----------------|
| 모델 학습 횟수   | 6회             |
| 최종 선정 모델   | Iteration 6     |
| 학습데이터       | Medium Heavy(300장) + 증강 100장 |
| Precision       | 83.1%           |
| Recall          | 85.6%           |
| AP              | 89.2%           |
| 비고            | 가장 성능이 우수한 Iteration 채택 |


### 웹 서비스 구현
Gradio 기반의 웹 서비스를 통해 AI 모델 분석, 사용자 참여, 데이터 시각화 제공  

#### 📸 이미지 업로드 및 AI 분석
---
1. **격자형 빗물받이 여부 판별**
    - 업로드된 이미지가 격자형 빗물받이인지 아닌지 판별

2. **빗물받이 오염도 판별**
    - 격자형인 경우, 깨끗한 상태인지 판별
      
3. **담배꽁초 탐지**
    - 오염된 빗물받이 이미지에서 담배꽁초 감지하여 표시

<table>
    <tbody>
        <tr>
            <td><img src="https://github.com/user-attachments/assets/e942674c-6fb6-441d-81f4-b4fe1c9454c9" width=400px alt="_"/></td>
            <td><img src="https://github.com/user-attachments/assets/8f80cf73-5126-48a6-9069-f40872544778" width=400px alt="_"/></td>
            <td><img src="https://github.com/user-attachments/assets/6ffdb0b9-3cae-49cc-aed8-7f3a471f971b" width=400px alt="_"/></td>
        </tr>
    </tbody>
</table>  


#### ✍️ 사용자 참여
---
1. **쓰레기 태깅**
    - 이미지 내 쓰레기를 찾아 항목별로 태그
      
2. **결과 비교**
    - 모델 감지와 사용자 태깅 결과를 IoU(교집합 비율)로 비교 제공
    - 항목별 태깅된 쓰레기 갯수 표시
      
3. **학교 정보 입력**
    - 공공데이터 기반의 초등학교명 선택 후, 태깅 결과 저장
      
4. **안전신문고 연동**
    - 아이들이 직접적으로 사회문제 해결에 참여하게끔 유도하는 안전신문고 링크 제공

<table>
    <tbody>
        <tr>
            <td><img src="https://github.com/user-attachments/assets/8802a976-bbac-4d7d-9aab-064d3ea5099c" width=400px alt="_"/></td>
            <td><img src="https://github.com/user-attachments/assets/15b31e43-6230-41e9-bea0-d5821b1e2380" width=400px alt="_"/></td>
            <td><img src="https://github.com/user-attachments/assets/b4e11374-11b1-4fd8-8851-f603d8b19394" width=400px alt="_"/></td>
        </tr>
    </tbody>
</table>  


#### 📊 수집된 데이터 시각화
---
1. **학교별 참여도 랭킹**
    - 태깅 활동이 활발한 학교 Top3 그래프로 시각화
      
2. **사회적 가치 환산**
    - 태깅 활동으로 아낄 수 있는 사회적 비용의 가치를 아이들이 이해할 수 있는 직관적인 수치로 설명

<table>
    <tbody>
        <tr>
            <td><img src="https://github.com/user-attachments/assets/d1650017-1386-475d-94d6-cbebfcc0213e" width=400px alt="_"/></td>
            <td><img src="https://github.com/user-attachments/assets/5ed276e6-be1b-4116-8ae3-b5bcbd6e83fd" width=400px alt="_"/></td>
        </tr>
    </tbody>
</table>


## 서비스
* Hugging Face Space [https://eunalee-project1.hf.space/](https://eunalee-project1.hf.space/)  
* 데모 영상 [https://your-demo-link.com](https://your-demo-link.com)