import gradio as gr
import folium
import io
import os
import re
import requests
import ast
import numpy as np
import pandas as pd
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
        # return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ''
        return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # 빗물받이인 경우,
    # 1. 심각도 예측    
    severity_label, severity_probability = predict_with_api(image_path, 'severity')
    is_clean = severity_label == 'clean'
    result_msg = f'🟢 깨끗 ({(severity_probability * 100) :.0f}%)' if is_clean else f'🟡 주의 요망 ({severity_label} : {(severity_probability * 100) :.0f}%)'
    
    # 2. GPS 정보 추출
    gps = get_image_gps(image_path)
    # 서울 중심
    map = folium.Map(location=[37.566535, 126.9779692], zoom_start=11)
    folium.Marker(location=[gps[0], gps[1]], icon=folium.Icon(color='red', icon='star')).add_to(map)
    map_html = map._repr_html_()

    # 안전신문고 버튼
    report_btn = '''
        <a href="https://www.safetyreport.go.kr" target="_blank" style="display: block; border-radius: 6px; padding: 15px; background: #e4e4e7; color: black; font-weight: bold; text-align: center; text-decoration: none;">
            안전신문고에 신고하러 가기
        </a>
    '''

    return validation_msg, gr.update(value=result_msg, visible=True), gr.update(value=report_btn, visible=True), gr.update(visible=False) if is_clean else gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


#──────────────────────────────────────────────────────────────
# 이미지 위치 정보
#──────────────────────────────────────────────────────────────
def get_image_gps(image_path) :
    # 기본값 (서울 중심)
    lat, lon = 37.566535, 126.9779692

    # 이미지가 삭제된 경우, 모든 셋팅 초기화
    if image_path is None :
        return lat, lon

    # 이미지 불러올 때, 오류가 발생한 경우 기본값 사용
    try :
        image = Image.open(image_path)
        metadata = image._getexif()
    except Exception :
        return lat, lon

    # 메타정보가 없는 경우, 기본값 사용
    if not metadata : 
        return lat, lon

    # 메타정보가 있는 경우, 이미지 위치정보 추출
    for tag, value in metadata.items() :
        decoded = TAGS.get(tag, tag)

        if decoded == 'GPSInfo' :
            # 위도 (도, 분, 초)
            gps_lat = value.get(2)
            # 경도 (도, 분, 초)
            gps_lon = value.get(4)

    try :
        if gps_lat and gps_lon : 
            # 위도
            lat = (((gps_lat[2] / 60.0) + gps_lat[1]) / 60.0) + gps_lat[0]
            # 경도
            lon = (((gps_lon[2] / 60.0) + gps_lon[1]) / 60.0) + gps_lon[0]
    except Exception : 
        pass

    return lat, lon


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
 
    # 사용자가 입력한 바운딩 박스 정보
    for u_idx, ubox in enumerate(user_boxes):
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
 
    # AI 감지 바운딩 박스 정보
    for idx, abox in enumerate(ai_boxes):
        if idx not in used_ai:
            # AI만 감지하면 주황색 바운딩 박스 표시
            draw.rectangle([abox["xmin"], abox["ymin"], abox["xmax"], abox["ymax"]], outline="orange", width=20)
 
    # 사용자만 감지한 갯수
    user_only = len(user_boxes) - matched_count
    # AI만 감지한 갯수
    ai_only = len(ai_boxes) - len(used_ai)
 
    # 점수 계산
    # 사용자와 AI 태그 영역이 일치하면 가중치 0.5
    # 사용자만 태그하면 가중치 0.3
    # AI만 태그하면 가중치 0.2
    score_match = matched_count * 0.5
    score_user = user_only * 0.3
    score_ai = ai_only * 0.2
    total_score = score_match + score_user + score_ai

    results_to_save.append({
        'user_tag' : int(len(user_boxes)),
        'ai_only' : int(len(ai_boxes)),
        "matched_count": int(matched_count),
        "user_only": int(user_only),
        "ai_only": int(ai_only),
        "score_user": round(score_user, 1),
        "score_ai": round(score_ai, 1),
        "total_score": round(total_score, 1)
    })
    print(results_to_save)

    html = f'''
    <div style="font-family: sans-serif; line-height: 1.5;">
        <h3>📋 결과</h3>
        <ul>
            <li><b>🟩 일치한 태그:</b> {matched_count}/{len(user_boxes)}개</li>
            <li><b>🟨 사용자만 태깅한 박스:</b> {user_only}개</li>
            <li><b>🟧 AI만 감지한 박스:</b> {ai_only}개</li>
        </ul>
        <h3>📊 총점: {total_score:.1f}점</h3>
        <ul>
            <li><b>일치 항목 점수:</b> {score_match:.1f}점</li>
            <li><b>사용자만 태깅한 점수:</b> {score_user:.1f}점</li>
            <li><b>AI만 감지한 점수:</b> {score_ai:.1f}점</li>
        </ul>
    </div>
    '''

    return html, img, results_to_save


#──────────────────────────────────────────────────────────────
# 결과 저장
#──────────────────────────────────────────────────────────────
def submit_form(school_name, image_path, score) :
    print(school_name, image_path, score)
    result_msg = ''
    error_msg = ''

    # 초등학교명 유효성 검사
    pattern = r'초등학교'
    if not re.search(pattern, school_name) :
        error_msg = '초등학교 이름을 올바른 형식(예: xx초등학교)으로 입력해주세요.'
        return gr.update(value=error_msg, visible=True), gr.update(visible=True)
    
    lat, lon = get_image_gps(image_path)

    # 이미지 저장
    image = Image.open(image_path)
    os.makedirs("saved_images", exist_ok=True)
    filename = f"saved_images/image_{np.random.randint(100000)}.jpg"
    image.save(filename)

    # 입력 데이터 저장
    row = {
        'school' : school_name,
        'image' : filename,
        'score' : score,
        'lat' : lat,
        'lon' : lon
    }

    csv_file = 'school_attack.csv'
    header = not os.path.exists(csv_file)

    df = pd.DataFrame(row)
    df.to_csv(csv_file, mode='a', header=header, index=False, encoding='utf-8')

    # with open("saved_annotations.json", "a", encoding="utf-8") as f:
    #     json.dump({'school' : school_name, "image": filename, "score": score, 'lat' : lat, 'lon' : lon}, f, ensure_ascii=False)
    #     f.write("\n")
    
    result_msg = f"💾 저장 완료: {filename}"
 
    return gr.update(value=result_msg, visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value='', visible=False)


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
            # map = gr.HTML(visible=False)
            report_btn = gr.HTML(visible=False)
            detect_btn = gr.Button('🟦 AI 감지 및 태깅 시작', visible=False)

            # global 변수
            temp_ai_result = gr.State()
            image_path = gr.State()
            temp_save_result = gr.State()

            # 사용자 vs AI 이미지 비교
            with gr.Row(visible=False) as detect :
                ai_result = gr.Image(label="🤖 AI 감지 결과")
                annotator = image_annotator(
                    label_list=['담배'],
                    label_colors=[(255, 0, 0)]
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
            
            # 사용자 vs AI 비교
            compare_btn.click(
                fn=compare_boxes,
                inputs=[annotator, temp_ai_result],
                outputs=[html_output, compare_result, temp_save_result]
            )

            compare_btn.click(
                fn=lambda: (gr.update(visible=True),)*2,
                inputs=None,
                outputs=[compare, save_btn]
            )
            
            # 학교 이름 입력창
            with Modal(visible=False) as school_form :
                school_input = gr.Textbox(label='학교이름 (예: xx초등학교)')
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
                outputs=[modal_alert, school_form, school_input, modal_alert]
            )

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, report_btn, detect_btn, detect, compare_btn, compare, save_btn, school_form]
            )

        # 스쿨어택
        with gr.Tab('🏫') :
            gr.Markdown("## 🏫 초등학교별 순위")
            df = pd.read_csv('school_attack.csv')

            # 점수 딕셔너리 값 추출
            df['score'] = df['score'].apply(ast.literal_eval)
            score_df = df['score'].apply(pd.Series)
            df = pd.concat([df.drop(columns=['score']), score_df], axis=1)

            # 학교별 종합 점수가 가장 높은순으로 정렬
            sorted_df = df.sort_values(by=['school', 'total_score', 'score_user'], ascending=[True, False, False]).groupby('school', as_index=False).first()
            sorted_df = sorted_df[['school', 'total_score']]
            sorted_df.columns = ['학교명', '최고 점수']

            # 학교별 태그한 갯수가 많은순으로 정렬
            user_tag_df = df.groupby('school')['user_tag'].sum().reset_index()
            user_tag_df.columns = ['학교명', '총 수거량']

            # 데이터 병합 및 타입 변환
            school_attack_df = pd.merge(sorted_df, user_tag_df, on='학교명')
            school_attack_df['최고 점수'] = school_attack_df['최고 점수'].astype(float)
            school_attack_df['총 수거량'] = school_attack_df['총 수거량'].astype(int)

            # 병합한 데이터 정렬
            school_attack_df = school_attack_df.sort_values(by=['최고 점수', '총 수거량'], ascending=[False, False]).reset_index(drop=True)
            
            # 단위 표시
            school_attack_df['최고 점수'] = school_attack_df['최고 점수'].apply(lambda score : f'{score}점')
            school_attack_df['총 수거량'] = school_attack_df['총 수거량'].apply(lambda count : f'{int(count)}개')

            # 1~3순위 표시
            medals = ['🥇', '🥈', '🥉']
            for i in range(3) :
                school_attack_df.loc[i, '학교명'] = f'{medals[i]} {school_attack_df.loc[i, "학교명"]}'
            
            gr.DataFrame(value=school_attack_df)

demo.launch()