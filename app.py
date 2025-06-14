import gradio as gr
import folium
import io
import os
import requests
import ast
import json
import locale
import numpy as np
import pandas as pd
import altair as alt
from collections import Counter
from PIL import Image, ImageDraw, ImageOps
from gradio_modal import Modal
from gradio_image_annotation import image_annotator
from PIL.ExifTags import TAGS
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials


#──────────────────────────────────────────────────────────────
# UI 요소 숨기기
#──────────────────────────────────────────────────────────────
def hide_components(n) :
    return [gr.update(visible=False)] * n

#──────────────────────────────────────────────────────────────
# UI 요소 보이기
#──────────────────────────────────────────────────────────────
def show_components(*components) :
    return [gr.update(visible=True) for _ in components]

#──────────────────────────────────────────────────────────────
# 이미지 처리
#──────────────────────────────────────────────────────────────
def process_image(image_path) :
    # 이미지가 삭제된 경우, 모든 셋팅 초기화
    if image_path is None :
        return '', *hide_components(10)

    # 이미지 빗물받이 여부 판단
    service_or_not_label = predict_with_api(image_path)
    is_valid = service_or_not_label == 'service'
    validation_msg = f'✅이건 와플모양 배수구야!' if is_valid else '🚫이건 와플모양 배수구가 아니야! 다시 올려줘'

    # 빗물받이가 아닌 경우,
    if not is_valid :
        return validation_msg, *hide_components(10)
    
    # 빗물받이인 경우, 오염도 예측    
    severity_label = predict_with_api(image_path, 'severity')
    is_clean = severity_label == 'Clean'
    result_msg = f'🟢 깨끗해! 다른 배수구도 확인해볼래?' if is_clean else f'🟡 더러워! 커비랑 같이 얼마나 더러운지 확인해볼까?'

    return validation_msg, gr.update(value=result_msg, visible=True), *hide_components(1), gr.update(visible=False) if is_clean else gr.update(visible=True), *hide_components(7)

#──────────────────────────────────────────────────────────────
# 학습 모델 결과 반환
#──────────────────────────────────────────────────────────────
def predict_with_api(image_path, type='service_or_not') :
    # Azure Custom Vision API 연결 정보
    PREDICTION_CONFIG = {
        'service_or_not' : {
            'key' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
            'url' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/58b52583-2cfb-4767-b9e0-8e83032f9d95/classify/iterations/Iteration3/image',
        },
        'severity' : {
            'key' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
            'url' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/651b700e-e582-45f1-9eee-4ea3aa9670f7/classify/iterations/Iteration1/image',
        }
    }

    config = PREDICTION_CONFIG[type]

    # API 호출 시, 사용할 헤더 셋팅
    headers = {
        'Prediction-Key' : config['key'],
        # 바이너리 이미지 전송
        'Content-Type' : 'application/octec-stream'
    }

    # 전송할 이미지 (바이너리 형태)
    byte_data = pil_to_binary(image_path)

    # API 호출
    response = requests.post(config['url'], headers=headers, data=byte_data)
    predictions = response.json()['predictions']

    # 확률이 가장 높은 예측 항목 선택
    top_prediction = max(predictions, key=lambda x : x['probability'])
    label = top_prediction['tagName']
    probability = top_prediction['probability']

    return label

#──────────────────────────────────────────────────────────────
# PIL 이미지 객체 -> JPEG 형식의 바이너리 데이터로 변환
#──────────────────────────────────────────────────────────────
def pil_to_binary(image_path) :
    image = Image.open(image_path)
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_data = buf.getvalue()

    return byte_data

#──────────────────────────────────────────────────────────────
# IoU 계산 함수 - 두 바운딩 박스가 얼마나 겹치는지를 나타냄
# IoU = (겹친 영역 넓이) / (전체 영역 넓이)
# 결과값 - 0.0 ~ 1.0 사이 (0.0 : 전혀 겹치지 않음, 1.0 : 완전히 동일)
#──────────────────────────────────────────────────────────────
def calculate_iou(boxA, boxB):
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])
    
    # 겹치는 영역 (교집합)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # 전체 영역 (합집합)
    unionArea = float(
        (boxA["xmax"] - boxA["xmin"]) * (boxA["ymax"] - boxA["ymin"]) +
        (boxB["xmax"] - boxB["xmin"]) * (boxB["ymax"] - boxB["ymin"]) - interArea
    )

    return interArea / unionArea if unionArea != 0 else 0

#──────────────────────────────────────────────────────────────
# EXIF Orientation 에 따른 개체 감지 바운딩 박스 좌표 설정
#──────────────────────────────────────────────────────────────
def transform_box(box, width, height, orientation):
    xmin, ymin, xmax, ymax = box
    
    # 회전 없음
    if orientation == 1:
        return xmin, ymin, xmax, ymax
    
    # 좌우 반전
    elif orientation == 2:
        new_xmin = width - xmax
        new_xmax = width - xmin
        return new_xmin, ymin, new_xmax, ymax
    
    # 180도 회전
    elif orientation == 3:
        new_xmin = width - xmax
        new_xmax = width - xmin
        new_ymin = height - ymax
        new_ymax = height - ymin
        return new_xmin, new_ymin, new_xmax, new_ymax
    
    # 180도 회전 + 좌우 반전
    elif orientation == 4:
        new_ymin = height - ymax
        new_ymax = height - ymin
        return xmin, new_ymin, xmax, new_ymax
    
    # 90도 반시계방향 회전 + 좌우 반전
    elif orientation == 5:
        new_xmin = ymin
        new_xmax = ymax
        new_ymin = width - xmax
        new_ymax = width - xmin
        return new_xmin, new_ymin, new_xmax, new_ymax
    
    # 90도 시계방향 회전 (270도 반시계방향)
    elif orientation == 6:
        new_xmin = height - ymax
        new_xmax = height - ymin
        new_ymin = xmin
        new_ymax = xmax
        return new_xmin, new_ymin, new_xmax, new_ymax
    
    # 90도 시계방향 회전 + 좌우 반전
    elif orientation == 7:
        new_xmin = ymin
        new_xmax = ymax
        new_ymin = xmin
        new_ymax = xmax
        return new_xmin, new_ymin, new_xmax, new_ymax
    
    # 90도 반시계방향 회전 (270도 시계방향)
    elif orientation == 8:
        new_xmin = ymin
        new_xmax = ymax
        new_ymin = width - xmax
        new_ymax = width - xmin
        return new_xmin, new_ymin, new_xmax, new_ymax
    
    return xmin, ymin, xmax, ymax

#──────────────────────────────────────────────────────────────
# AI 감지
#──────────────────────────────────────────────────────────────
def detect_with_boxes(image_path):
    byte_data = pil_to_binary(image_path)
    image = Image.open(image_path)

    # 이미지 자동 회전
    transform_image = ImageOps.exif_transpose(image)

    # 메타데이터 orientation 정보
    orientation = image._getexif()
    if orientation is not None:
        orientation = orientation.get(274, 1)
    
    # Custom Vision API 설정
    PREDICTION_KEY = "BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC"
    ENDPOINT_URL = "https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/e81e8daf-2a54-4f41-9c8f-581d45e49ee9/detect/iterations/Iteration1/image"

    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }

    # Prediction 클라이언트 생성
    credentials = ApiKeyCredentials(in_headers={'Prediction-Key' : PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT_URL, credentials=credentials)
    response = requests.post(ENDPOINT_URL, headers=headers, data=byte_data)
    results = response.json()
 
    ai_boxes = []
    image_with_boxes = transform_image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
 
    for pred in results["predictions"]:
        if pred["probability"] > 0.5:
            w, h = image.width, image.height
            box = pred["boundingBox"]
            left = int(box["left"] * w)
            top = int(box["top"] * h)
            right = int(box["width"] * w)
            bottom = int(box["height"] * h)
            
            # 회전에 따른 바운딩 박스 좌표 변환
            box_info = transform_box(
                (left, top, left+right, top+bottom), w, h, orientation
            )
 
            ai_boxes.append({
                "label": pred["tagName"],
                "xmin": box_info[0],
                "ymin": box_info[1],
                "xmax": box_info[2],
                "ymax": box_info[3]
            })
 
            draw.rectangle([box_info[0], box_info[1], box_info[2], box_info[3]], outline="red", width=20)
            draw.text((box_info[0], box_info[1]), f"{pred['tagName']} ({pred['probability']:.2f})", fill="black")

    return image_with_boxes, ai_boxes

#──────────────────────────────────────────────────────────────
# 업로드 처리
#──────────────────────────────────────────────────────────────
def handle_upload(image_path):
    image = Image.open(image_path)
    # 이미지 자동 회전
    transform_image = ImageOps.exif_transpose(image)

    ai_img, ai_boxes = detect_with_boxes(image_path)
    annotator_input = {
        "image": transform_image,
        "annotations": []
    }
    return ai_img, annotator_input, ai_boxes, image_path

#──────────────────────────────────────────────────────────────
# 사용자 vs AI 바운딩 박스 비교
#──────────────────────────────────────────────────────────────
def compare_boxes(user_data, ai_boxes):
    if not user_data or "boxes" not in user_data:
        return "❌ 네가 찾은 쓰레기가 없어! 쓰레기 찾는걸 도와줄래?", None, []
 
    img_array = user_data["image"]
    user_boxes = user_data["boxes"]
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
 
    # 일치한 갯수
    matched_count = 0
    results_to_save = []
    used_ai = set()
    used_user = set()
    labels = []

    # 사용자가 입력한 바운딩 박스 정보
    for u_idx, ubox in enumerate(user_boxes):
        # 사용자가 태깅한 바운딩 박스 정보
        if ubox['label'] :
            labels.append(ubox['label'])

        user = {
            "xmin": ubox["xmin"],
            "ymin": ubox["ymin"],
            "xmax": ubox["xmax"],
            "ymax": ubox["ymax"]
        }
 
        best_iou = 0
        matched_ai_idx = -1
        for i, abox in enumerate(ai_boxes) :
            iou = calculate_iou(user, abox)
            # IoU 최댓값 셋팅
            if iou > best_iou :
                best_iou = iou
                matched_ai_idx = i
 
        # IoU 값이 0.5 이상이면 일치한 갯수 카운팅
        # AI, 사용자 모두 감지하면 초록색 바운딩 박스 표시
        if best_iou >= 0.5 :
            matched_count += 1
            used_ai.add(matched_ai_idx)
            used_user.add(u_idx)
            draw.rectangle([user["xmin"], user["ymin"], user["xmax"], user["ymax"]], outline="green", width=20)
        else:
            # 사용자만 감지하면 노란색 바운딩 박스 표시
            draw.rectangle([user["xmin"], user["ymin"], user["xmax"], user["ymax"]], outline="yellow", width=20)
    
    # 태그별 바운딩 박스 갯수
    label_counts = Counter(labels)
    # 전체 태그 갯수
    total_tag = sum(label_counts.values())
 
    # AI 감지 바운딩 박스 정보
    for idx, abox in enumerate(ai_boxes):
        if idx not in used_ai:
            # AI만 감지하면 주황색 바운딩 박스 표시
            draw.rectangle([abox["xmin"], abox["ymin"], abox["xmax"], abox["ymax"]], outline="orange", width=20)
 
    # 사용자만 감지한 갯수
    user_only = len(user_boxes) - matched_count
    # AI만 감지한 갯수
    ai_only = len(ai_boxes) - len(used_ai)

    # 저장할 데이터
    results_to_save.append({
        'total_tag' : total_tag,
        'total_label_tag' : dict(label_counts)
    })

    # 태그별 바운딩 박스 갯수 UI 노출
    label_summary_html = ''.join(f'<li><b>{label} :</b> {count}개</li>' for label, count in label_counts.items())

    result_html = f'''
    <div style="font-family: sans-serif; line-height: 1.5;">
        <h3>📋 결과</h3>
        <ul style="list-style: none;">
            <li>🟩 우리 둘 다 쓰레기 {matched_count}개를 똑같이 찾았어!</li>
            <li>🟨 네가 찾은 쓰레기는 {user_only}개야!</li>
            <li>🟧 커비가 찾은 쓰레기는 {ai_only}개야!</li>
        </ul>
        <h4>📦 내가 찾은 쓰레기</h4>
            <ul>
                <li><b>총 갯수 :</b> {total_tag}개</li>
                {label_summary_html}
            </ul>
    </div>
    '''

    return result_html, img, results_to_save

#──────────────────────────────────────────────────────────────
# 전체 태그 삭제
#──────────────────────────────────────────────────────────────
def reset_boxes(img_path):
    image = Image.open(img_path)
    transform_image = ImageOps.exif_transpose(image)
    img_np = np.array(transform_image)
    return {
        "image": img_np, 
        "annotations": [], 
        "boxes": []
    }

#──────────────────────────────────────────────────────────────
# 마지막 태그 삭제
#──────────────────────────────────────────────────────────────
def remove_last_box(data, img_path):
    if not isinstance(data, dict):
        return gr.update()

    ann = data.get("annotations", [])
    boxes = data.get("boxes", [])
    
    return {
        "image": img_path,  
        "annotations": ann[1:],
        "boxes": boxes[1:]
    }

#──────────────────────────────────────────────────────────────
# 초등학교 선택
#──────────────────────────────────────────────────────────────
def get_school_list() :
    with open('전국초중등학교위치표준데이터.json', 'r', encoding='utf-8') as f :
        json_data = json.load(f)
    school_names = [record['학교명'] for record in json_data['records'] if record['학교급구분'] == '초등학교']

    locale.setlocale(locale.LC_COLLATE, 'ko_KR.UTF-8')
    return sorted(school_names, key=locale.strxfrm)

#──────────────────────────────────────────────────────────────
# 결과 저장
#──────────────────────────────────────────────────────────────
def submit_form(school_name, image_path, tag_info) :
    result_msg = ''
    error_msg = ''

    # 초등학교명 입력값 유효성 검사
    if not school_name :
        error_msg = '너의 학교를 선택해줘'
        return gr.update(value=error_msg, visible=True), gr.update(visible=True), gr.update(visible=True)


    # 이미지 저장
    image = Image.open(image_path)
    transform_image = ImageOps.exif_transpose(image)
    os.makedirs("saved_images", exist_ok=True)
    filename = f"saved_images/image_{np.random.randint(100000)}.jpg"
    transform_image.save(filename)
    
    # 데이터 저장
    row = {
        'school' : school_name,
        'image' : filename,
        'tag_info' : tag_info
    }
    print(row)

    csv_file = 'school_attack.csv'
    header = not os.path.exists(csv_file)

    df = pd.DataFrame(row)
    df.to_csv(csv_file, mode='a', header=header, index=False, encoding='utf-8')
    
    result_msg = f"💾 저장 완료: {filename}"
 
    return gr.update(value=result_msg, visible=False), gr.update(value='', visible=True), *hide_components(1)

#──────────────────────────────────────────────────────────────
# 스쿨어택 데이터 가져오기
#──────────────────────────────────────────────────────────────
def get_school_attck_data() :
    csv_file = 'school_attack.csv'

    # 파일이 없으면 빈 DataFrame 반환
    if not os.path.exists(csv_file):
        return pd.DataFrame(columns=[
            '학교명', '총 태그 갯수', '배수구 수', '살린 금액',
            '요아정', '마라탕', '아이스크림'
        ])
    
    try :
        df = pd.read_csv('school_attack.csv')

        # 태깅한 정보(문자열 형태 딕셔너리)를 딕셔너리로 변경
        df['tag_info'] = df['tag_info'].apply(ast.literal_eval)

        # 총 태깅한 갯수 정보
        df['total_tag'] = df['tag_info'].apply(lambda x: x.get('total_tag', 0))

        # 쓰레기 종류별 태깅한 갯수 정보
        df['total_label_tag'] = df['tag_info'].apply(lambda x: x.get('total_label_tag', {}))

        # 모든 라벨 추출
        all_labels = set()
        for label_dict in df['total_label_tag']:
            all_labels.update(label_dict.keys())

        # 쓰레기 종류별 컬럼 생성
        for label in all_labels:
            df[label] = df['total_label_tag'].apply(lambda x: x.get(label, 0))

        # 사용하지 않는 컬럼 삭제
        df.drop(columns=['tag_info', 'total_label_tag'], inplace=True)

        # 태깅 정보 합계
        school_tag_info = df.groupby('school').sum(numeric_only=True).reset_index()

        # 점검한 배수구 수 (학교수)
        school_counts = df.groupby('school').size().reset_index(name='count')

        # 모든 데이터 병합
        df = pd.merge(school_tag_info, school_counts, on='school')

        # 컬럼명 변경
        df = df.rename(columns={
            'school' : '학교명',
            'count': '배수구 수',
            'total_tag': '총 태그 갯수'
        })

        # 총 태깅 갯수가 많은 순, 그 다음 점검한 배수구 수가 많은 순 정렬
        df = df.sort_values(by=['총 태그 갯수', '배수구 수'], ascending=[False, False]).reset_index(drop=True)

        # 살린 배수구 금액 추출
        price = 10000
        df['살린 금액'] = df['배수구 수'] * price

        # 아이템별 단가
        item_prices = {
            '요아정': 4500,
            '마라탕': 13000,
            '아이스크림': 1500
        }

        for items, price in item_prices.items():
            df[items] = (df['살린 금액'] / price).astype(int)

        return df
    
    except Exception as e :
        return pd.DataFrame(columns=[
            '학교명', '총 태그 갯수', '배수구 수', '살린 금액',
            '요아정', '마라탕', '아이스크림'
        ])

#──────────────────────────────────────────────────────────────
# 스쿨어택 그래프 
#──────────────────────────────────────────────────────────────
def get_ranked_chart(df) :
    # ✅ df가 비어 있으면 빈 차트 반환
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_point().encode()
    
    # 총 태깅수 기준으로 내림차순 정렬
    df = df.sort_values(by='총 태그 갯수', ascending=False).reset_index(drop=True).head(5)

    # 순위 부여
    df['순위'] = ''
    if len(df) > 0: df.loc[0, '순위'] = '🥇'
    if len(df) > 1: df.loc[1, '순위'] = '🥈'
    if len(df) > 2: df.loc[2, '순위'] = '🥉'

    # 색상 지정
    def color_func(rank):
        if rank == '🥇': return '#FFD700'
        elif rank == '🥈': return '#C0C0C0'
        elif rank == '🥉': return '#CD7F32'
        else: return '#a9a9a9'
    df['color'] = df['순위'].apply(color_func)

    # 정렬 기준 리스트로 직접 지정
    school_order = df['학교명'].tolist()

    # 막대 차트 (세로 막대그래프)
    bar = alt.Chart(df).mark_bar().encode(
        x=alt.X('학교명:N', sort=school_order, title='학교'),
        y=alt.Y('총 태그 갯수:Q', axis=alt.Axis(title='총 태그 갯수', titleAngle=0)),
        color=alt.Color('color:N', scale=None, legend=None),
        tooltip=['학교명', '총 태그 갯수'],
    ).properties(
        title='커비와 함께 쓰레기를 많이 찾은 학교 Top5'
    )

    # 이모지 텍스트
    text = alt.Chart(df[df['순위'] != '']).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,  # 막대 위에 위치
        fontSize=24,
        fontWeight='bold'
    ).encode(
        x=alt.X('학교명:N', sort=school_order),
        y='총 태그 갯수:Q',
        text='순위:N'
    )

    # 전체 차트 구성
    chart = (bar + text).properties(
        width=100 * len(df),  # 반응형 너비
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16,
        grid=False
    ).configure_axisX(
        labelAngle=0,       # x축 라벨 가로로 표시
        labelLimit=200      # 한 레이블당 너비 제한
    ).configure_view(
        stroke=None
    ).configure_title(
        fontSize=20,
    )

    return chart

#──────────────────────────────────────────────────────────────
# 우리가 살린 배수구
#──────────────────────────────────────────────────────────────
def display_save_price() :
    df = get_school_attck_data()

    # 조회된 데이터가 없는 경우
    if df.empty :
        return '''
        <div style="font-family: sans-serif; font-size: 16px; color: gray;">
            ⚠️ 우리가 살린 배수구가 아직 없어! 커비랑 같이 숨은 쓰레기를 찾아볼래?
        </div>
        '''
    
    price = 130000
    total = df['배수구 수'].sum()
    total_price = total * price

    # 항목별 금액
    price_list = {
        '요아정' : 4500,
        '마라탕' : 13000,
        '아이스크림' : 1500
    }

    # 항목별 단위
    units = {
        '요아정': '개',
        '마라탕' : '그릇',
        '아이스크림' : '개'
    }

    # 아이콘
    icons = {
        '요아정' : '🍨',
        '마라탕' : '🍲',
        '아이스크림' : '🍦',
    }

    # 항목별 금액 환산 결과
    item_counts = []
    for item, item_price in price_list.items() :
        # 점검한 배수구 총 갯수
        count = total_price // item_price
        # 아이콘
        icon = icons.get(item, '🎁')
        # 단위
        unit = units.get(item, '')
        item_counts.append({
            'item' : item,
            'count' : count,
            'icon' : icon,
            'unit' : unit
        })

        # count 기준으로 내림차순 정렬
        item_counts.sort(key=lambda x: x['count'], reverse=True)

        # 정렬된 항목으로 HTML 생성
        item_lines = ''
        for entry in item_counts:
            item_lines += f"<div>{entry['icon']} {entry['item']} <b>약 {entry['count']}{entry['unit']}</b></div>\n"

    # 전체 HTML 문자열
    html_output = f'''
    <style>
    .info-list {{
        list-style: none;
        padding: 0;
        margin: 0;
        font-size: 16px;
    }}

    .info-item {{
        list-style: none;
        margin-bottom: 24px;
        padding-bottom: 20px;
    }}

    .info-title {{
        font-weight: bold;
        margin-bottom: 6px;
        padding-bottom: 6px;
        border-bottom: 1px solid #ccc;
    }}

    .info-item > div:not(.info-title) {{
        margin-top: 6px;
    }}
    </style>

    <ul class="info-list">
        <li class="info-item">
            <div class="info-title">배수구 수</div>
            <div>{total}개</div>
        </li>
        <li class="info-item">
            <div class="info-title">우리가 아낀 금액 (배수구 1개당 금액은 130,000원이야!)</div>
            <div>{total_price:,}원</div>
        </li>
        <li class="info-item">
            <div class="info-title">이 돈으로 살 수 있는 것</div>
            {item_lines}
        </li>
    </ul>
    '''

    return html_output

#──────────────────────────────────────────────────────────────
# 새로고침 시, 데이터 업데이트
#──────────────────────────────────────────────────────────────
def refresh_school_attck() :
    # 차트 그릴 데이터 가져오기
    df = get_school_attck_data()

    # 조회된 데이터가 없는 경우
    if df.empty :
        html =  '''
        <div style="font-family: sans-serif; font-size: 16px; color: gray;">
            ⚠️ 우리가 살린 배수구가 아직 없어! 커비랑 같이 숨은 쓰레기를 찾아볼래?
        </div>
        '''
        return alt.Chart(pd.DataFrame()).mark_point().encode(), html

    # 조회된 데이터가 있는 경우
    return get_ranked_chart(df), display_save_price()
    

#──────────────────────────────────────────────────────────────
# Gradio UI
#──────────────────────────────────────────────────────────────
with gr.Blocks() as demo :
    gr.Markdown('## 💧비추다 with 스쿨어택')
    gr.Markdown('우리의 AI 커비를 도와 빗물받이에 있는 쓰레기를 찾고 제일 잘 도와준 학교를 가려보자!')

    with gr.Tabs() :
        # 개체 감지 (담배꽁초) 탭
        with gr.Tab('🔍 숨은 쓰레기 찾기') :
            # 이미지 메타정보를 사용하기 위해서 type='filepath' 로 지정
            gr.Markdown('#### 📸 와플 모양 배수구 사진만 올려줘!')
            image_input = gr.Image(type='filepath', label='사진을 올려줘')
            validation = gr.Textbox(label='네가 찍은 사진')
            prediction = gr.Textbox(label='네가 찍어준 배수구', visible=False)
            detect_btn = gr.Button('🟦 커비랑 게임 시작', visible=False)

            # global 변수
            temp_ai_result = gr.State()
            image_path = gr.State()
            temp_save_result = gr.State()

            # 사용자 vs AI 이미지 비교
            notice = gr.Markdown('''
                        #### 📢 커비는 담배를 찾았어!
                        커비가 못 찾은 쓰레기를 같이 찾아볼래? <b>담배꽁초, 낙엽, 기타 쓰레기를 찾아줘!</b>
                        ''', visible=False)
            with gr.Row(visible=False) as detect :
                ai_result = gr.Image(label="커비가 찾은 담배꽁초")
                annotator = image_annotator(
                    label='내가 찾는 쓰레기',
                    label_list=['담배꽁초', '낙엽', '기타'],
                    label_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                )

            with gr.Row(visible=False) as button_row :
                clear_btn = gr.Button("❌ 태그한거 전부 지울래")
                remove_btn = gr.Button("⛔ 마지막 태그만 지울래")

            # 전체 태그 삭제
            clear_btn.click(
                fn=reset_boxes,
                inputs=[image_path],
                outputs=[annotator]
            )

            # 마지막 태그 삭제
            remove_btn.click(
                fn=remove_last_box,
                inputs=[annotator, image_path],
                outputs=[annotator]
            )

            compare_btn = gr.Button("📐 게임 결과", visible=False)
            
            # AI 감지 및 태깅
            detect_btn.click(
                fn=handle_upload,
                inputs=image_input,
                outputs=[ai_result, annotator, temp_ai_result, image_path]
            )

            detect_btn.click(
                fn=lambda: show_components(detect, compare_btn, notice, button_row),
                inputs=None,
                outputs=[detect, compare_btn, notice, button_row]
            )

            # 비교 결과 노출
            with gr.Row(visible=False) as compare :
                compare_result = gr.Image(label="게임 결과")
                html_output = gr.HTML()
            save_btn = gr.Button("💾 학교 친구들에게 자랑하기", visible=False)       

            # 학교명 조회
            school_names = get_school_list()
            
            # 학교 이름 입력창
            with gr.Row(visible=False) as school_form :
                school_input = gr.Dropdown(choices=school_names, label='초등학교 선택', value=None)
                modal_alert = gr.Textbox(visible=False, label='알림')
                submit_btn = gr.Button('우리 학교 점수 올리기')

            report_btn = gr.HTML('''
                            <a href="https://www.safetyreport.go.kr" target="_blank" style="display: block; border-radius: 6px; padding: 15px; background: #033075; color: white; font-weight: bold; text-align: center; text-decoration: none;">
                            안전신문고에 신고하러 가기
                            </a>
                        ''', visible=False)
            
            # 결과 저장 버튼 클릭 시,
            save_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=[school_form]
            )

            # 사용자 vs AI 비교
            compare_btn.click(
                fn=compare_boxes,
                inputs=[annotator, temp_ai_result],
                outputs=[html_output, compare_result, temp_save_result]
            )

            compare_btn.click(
                fn=lambda: show_components(compare, save_btn, report_btn),
                inputs=None,
                outputs=[compare, save_btn, report_btn]
            )

            # 제출 버튼 클릭 시,
            submit_btn.click(
                fn=submit_form,
                inputs=[school_input, image_path, temp_save_result],
                outputs=[modal_alert, school_input, school_form]
            )

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, report_btn, detect_btn, notice, detect, button_row, compare_btn, compare, save_btn, school_form]
            )

        # 저장한 데이터 시각화
        with gr.Tab('🏫 스쿨어택') :
            # 새로고침
            refresh_btn = gr.Button('🔄 새로고침')

            # 스쿨어택 차트
            gr.Markdown('## 🏅 우리 학교는 몇 등 ?')

            df = get_school_attck_data()
            df['학교명'] = df['학교명'].astype(str).str.replace(r'\s+', '', regex=True)
            
            with gr.Row() :
                plot = gr.Plot(get_ranked_chart(df), show_label=False)
            gr.HTML("<div style='height: 40px;'></div>")

            # 우리가 살린 배수구
            gr.Markdown('## 💸 커비와 함께 살린 배수구')
            html_output = gr.HTML(value=display_save_price())

            # 새로고침 시, 데이터 업데이트
            refresh_btn.click(
                fn=refresh_school_attck,
                inputs=None,
                outputs=[plot, html_output]
            )
        
        # 맨 위로 이동
        scroll_button = gr.HTML(''' 
            <style>
            #scrollToTop {
                position: fixed;
                bottom: 40px;
                right: 40px;
                z-index: 9999;
                background-color: #fed7aa;
                color: #ea580c;
                width: 48px;
                height: 48px;
                border: none;
                border-radius: 50%;
                font-size: 30px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>

            <button id="scrollToTop" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">↑</button>
            ''')

demo.launch()
