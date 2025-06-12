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
from collections import Counter
from PIL import Image, ImageDraw, ImageOps
from gradio_modal import Modal
from gradio_image_annotation import image_annotator
from PIL.ExifTags import TAGS
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials


#──────────────────────────────────────────────────────────────
# 이미지 처리
#──────────────────────────────────────────────────────────────
def process_image(image_path) :
    # 이미지가 삭제된 경우, 모든 셋팅 초기화
    if image_path is None :
        return '', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # 이미지 빗물받이 여부 판단
    service_or_not_label, service_or_not_probability = predict_with_api(image_path)
    is_valid = service_or_not_label == 'service'
    validation_msg = f'✅유효한 사진입니다. (예측 : {(service_or_not_probability * 100) :.0f}%)' if is_valid else '🚫유효하지 않은 사진입니다.'

    # 빗물받이가 아닌 경우,
    if not is_valid :
        return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # 빗물받이인 경우, 오염도 예측    
    severity_label, severity_probability = predict_with_api(image_path, 'severity')
    is_clean = severity_label == 'clean'
    result_msg = f'🟢 깨끗 ({(severity_probability * 100) :.0f}%)' if is_clean else f'🟡 주의 요망 ({severity_label} : {(severity_probability * 100) :.0f}%)'
    
    # 안전신문고 버튼
    report_btn = '''
        <a href="https://www.safetyreport.go.kr" target="_blank" style="display: block; border-radius: 6px; padding: 15px; background: #e4e4e7; color: black; font-weight: bold; text-align: center; text-decoration: none;">
            안전신문고에 신고하러 가기
        </a>
    '''

    return validation_msg, gr.update(value=result_msg, visible=True), gr.update(value=report_btn, visible=False), gr.update(visible=False) if is_clean else gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


#──────────────────────────────────────────────────────────────
# 학습 모델 결과 반환
#──────────────────────────────────────────────────────────────
def predict_with_api(image_path, type='service_or_not') :
    # Custom Vision Predictioin 정보
    PREDICTION_KEY = {
        'service_or_not' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
        'severity' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
    }
        
    ENDPOINT_URL = {
        'service_or_not' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/58b52583-2cfb-4767-b9e0-8e83032f9d95/classify/iterations/Iteration3/image',
        'severity' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/ab4cf356-d250-44f4-9221-12c8560bbee1/classify/iterations/Iteration9/image',
    }

    # API 호출 시, 사용할 헤더 셋팅
    headers = {
        'Prediction-Key' : PREDICTION_KEY[type],
        # 바이너리 이미지 전송
        'Content-Type' : 'application/octec-stream'
    }

    # 전송할 이미지 (바이너리 형태)
    byte_data = pil_to_binary(image_path)

    # API 호출
    response = requests.post(ENDPOINT_URL[type], headers=headers, data=byte_data)
    predictions = response.json()['predictions']

    # 확률이 가장 높은 예측 항목 선택
    top_prediction = max(predictions, key=lambda x : x['probability'])
    label = top_prediction['tagName']
    probability = top_prediction['probability']

    return label, probability


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
    print(user_data)
    if not user_data or "boxes" not in user_data:
        return "❌ 사용자 태깅 없음", None, []
 
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
        <ul>
            <li><b>AI랑 나랑 똑같이 찾은 쓰레기🟩 :</b> {matched_count}/{len(user_boxes)}개</li>
            <li><b>나만 찾은 쓰레기🟨 :</b> {user_only}개</li>
            <li><b>AI만 찾은 쓰레기🟧 :</b> {ai_only}개</li>
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
    print(school_name)
    result_msg = ''
    error_msg = ''

    # 초등학교명 입력값 유효성 검사
    if not school_name :
        error_msg = '초등학교를 선택해주세요.'
        return gr.update(value=error_msg, visible=True), gr.update(visible=True)

    # 이미지 저장
    image = Image.open(image_path)
    os.makedirs("saved_images", exist_ok=True)
    filename = f"saved_images/image_{np.random.randint(100000)}.jpg"
    image.save(filename)

    # 입력 데이터 저장
    # row = {
    #     'school' : school_name,
    #     'image' : filename,
    #     'score' : score,
    #     'lat' : lat,
    #     'lon' : lon
    # }
    
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
 
    return gr.update(value=result_msg, visible=False), gr.update(visible=False)


#──────────────────────────────────────────────────────────────
# 스쿨어택 데이터 가져오기
#──────────────────────────────────────────────────────────────
def get_school_attck_data() :
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
        'total_tag': '총 태깅수'
    })

    # 총 태깅 갯수가 많은 순, 그 다음 점검한 배수구 수가 많은 순 정렬
    df = df.sort_values(by=['총 태깅수', '배수구 수'], ascending=[False, False])

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


#──────────────────────────────────────────────────────────────
# 우리가 살린 배수구
#──────────────────────────────────────────────────────────────
def display_save_price() :
    df = get_school_attck_data()

    cols = ['요아정', '마라탕', '아이스크림']

    # 살린 금액이 높은 순 정렬
    df = df.sort_values(by='살린 금액', ascending=False).reset_index(drop=True)

    # 최대값 및 컬럼명 추출
    df['항목'] = df[cols].idxmax(axis=1)
    df['항목 갯수'] = df[cols].max(axis=1)

    school_df = df[['학교명', '살린 금액', '항목', '항목 갯수']]

    # 항목별 단위
    unit_map = {
        '아이스크림' : '개',
        '마라탕': '그릇',
        '닌텐도' : '대',
        '치킨' : '마리'
    }

    # 아이콘
    icon_map = {
        '아이스크림' : '🍦',
        '마라탕' : '🍲',
        '닌텐도' : '🎮',
        '치킨' : '🍗'
    }

    # HTML 테이블 생성
    table_rows = ''
    for _, row in school_df.iterrows():
        name = row['학교명']
        item = row['항목']
        price = row['살린 금액']
        item_cnt = row['항목 갯수']

        emoji = icon_map.get(item, '🎁')
        unit = unit_map.get(item, '')

        table_rows += f'''
            <tr>
                <td style="padding: 6px 0;">{name}</td>
                <td style="padding: 6px 0;">{price} 원</b></td>
                <td style="padding: 6px 0;">{emoji} {item} 약 <b>{item_cnt}{unit}</b></td>
            </tr>
        '''

    # 전체 HTML 템플릿
    html_output = f'''
        <table style="width: 100%; font-size: 16px; border: none; border-collapse: collapse;">
            <thead>
                <tr>
                <th style="text-align: left; padding-bottom: 8px;">학교명</th>
                <th style="text-align: left; padding-bottom: 8px;">살린 금액</th>
                <th style="text-align: left; padding-bottom: 8px;">아껴준 금액 환산 결과</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    '''

    return html_output


#──────────────────────────────────────────────────────────────
# Gradio UI
#──────────────────────────────────────────────────────────────
with gr.Blocks() as demo :
    gr.Markdown('## 🚧 격자형 빗물받이에 특화된 시범 서비스입니다.')

    with gr.Tabs() :
        # 개체 감지 (담배꽁초) 탭
        with gr.Tab('🔎') :
            gr.Markdown("## 🧪 담배꽁초 감지 비교 (사용자 vs AI)")

            # 이미지 메타정보를 사용하기 위해서 type='filepath' 로 지정
            image_input = gr.Image(type='filepath', label='사진을 올려주세요.')
            validation = gr.Textbox(label='이미지 확인')
            prediction = gr.Textbox(label='오염 심각도 확인', visible=False)
            detect_btn = gr.Button('🟦 AI 감지 및 태깅 시작', visible=False)

            # global 변수
            temp_ai_result = gr.State()
            image_path = gr.State()
            temp_save_result = gr.State()

            # 사용자 vs AI 이미지 비교
            with gr.Row(visible=False) as detect :
                ai_result = gr.Image(label="🤖 AI 감지 결과")
                annotator = image_annotator(
                    label='이미지 업로드',
                    label_list=['아래 항목에서 선택하세요.(선택X)', '담배꽁초', '종이', '재활용', '낙엽'],
                    label_colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
                )
            compare_btn = gr.Button("📐 비교", visible=False)
            
            # AI 감지 및 태깅
            detect_btn.click(
                fn=handle_upload,
                inputs=image_input,
                outputs=[ai_result, annotator, temp_ai_result, image_path]
            )

            detect_btn.click(
                fn=lambda: (gr.update(visible=True),)*2,
                inputs=None,
                outputs=[detect, compare_btn]
            )

            # 비교 결과 노출
            with gr.Row(visible=False) as compare :
                compare_result = gr.Image(label="📊 사용자 vs AI 비교 결과")
                html_output = gr.HTML()
            save_btn = gr.Button("💾 결과 저장", visible=False)       
            report_btn = gr.HTML()
            
            # 사용자 vs AI 비교
            compare_btn.click(
                fn=compare_boxes,
                inputs=[annotator, temp_ai_result],
                outputs=[html_output, compare_result, temp_save_result]
            )

            compare_btn.click(
                fn=lambda: (gr.update(visible=True),)*3,
                inputs=None,
                outputs=[compare, save_btn, report_btn]
            )

            # 학교명 조회
            school_names = get_school_list()
            
            # 학교 이름 입력창
            with Modal(visible=False) as school_form :
                school_input = gr.Dropdown(choices=school_names, label='초등학교 선택', value=None)
                modal_alert = gr.Textbox(visible=False, label='알림')
                submit_btn = gr.Button('제출')
            
            # 결과 저장 버튼 클릭 시,
            save_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=[school_form]
            )

            # 제출 버튼 클릭 시,
            submit_btn.click(
                fn=submit_form,
                inputs=[school_input, image_path, temp_save_result],
                outputs=[modal_alert, school_form]
            )

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, report_btn, detect_btn, detect, compare_btn, compare, save_btn, school_form]
            )

        # 스쿨어택
        with gr.Tab('📊') :
            gr.Markdown("## 🏫 스쿨어택")

            df = get_school_attck_data()
            
            compatition = gr.BarPlot(
                df,
                x='학교명',
                y='총 태깅수',
                x_title='학교',
                y_title='우리가 찾은 쓰레기 갯수',
                color='총 태깅수',
                tooltip='none',
            )

            gr.Markdown('## 💸 우리가 살린 배수구')
            gr.HTML(value=display_save_price())


demo.launch()