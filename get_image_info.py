## 실행 전, Python Image Library 설치
## pip install image 

import PIL.Image
from pprint import pprint

# 이미지 가져오기
img = PIL.Image.open('data/IMG_9311.JPG')
# 이미지 메타정보 추출
metadata = img._getexif()

pprint(metadata)