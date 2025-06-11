import gradio as gr
import folium
import io
import os
import requests
import json
import numpy as np
from PIL import Image, ImageDraw
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
        return '', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # 이미지 빗물받이 여부 판단
    service_or_not_label, service_or_not_probability = predict_with_api(image_path)
    is_valid = service_or_not_label == 'service'
    validation_msg = f'✅유효한 사진입니다. (예측 : {(service_or_not_probability * 100) :.0f}%)' if is_valid else '🚫유효하지 않은 사진입니다.'

    # 빗물받이가 아닌 경우,
    if not is_valid :
        return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ''
    
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

    return validation_msg, gr.update(value=result_msg, visible=True), gr.update(value=map_html, visible=True), gr.update(value=report_btn, visible=True)


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
# IoU 계산 함수
#──────────────────────────────────────────────────────────────
def calculate_iou(boxA, boxB):
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = float(
        (boxA["xmax"] - boxA["xmin"]) * (boxA["ymax"] - boxA["ymin"]) +
        (boxB["xmax"] - boxB["xmin"]) * (boxB["ymax"] - boxB["ymin"]) - interArea
    )
    return interArea / unionArea if unionArea != 0 else 0

#──────────────────────────────────────────────────────────────
# AI 감지
#──────────────────────────────────────────────────────────────
def detect_with_boxes(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    
    # Custom Vision API 설정
    PREDICTION_KEY = "5k8oJDDDmqLn5Yy9n1Q16CHetW6H0pvTjFPj1Q4JpQl7dAVJE0WhJQQJ99BEACYeBjFXJ3w3AAAIACOGZmg4"
    ENDPOINT_URL = "https://cv7934-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/92adf90f-3b67-4923-b2eb-1804da244279/detect/iterations/Iteration1/image"

    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }

    # Prediction 클라이언트 생성
    credentials = ApiKeyCredentials(in_headers={'Prediction-Key' : PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT_URL, credentials=credentials)
    response = requests.post(ENDPOINT_URL, headers=headers, data=buffered.getvalue())
    results = response.json()
 
    ai_boxes = []
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
 
    for pred in results["predictions"]:
        if pred["probability"] > 0.5:
            w, h = image.width, image.height
            box = pred["boundingBox"]
            left = int(box["left"] * w)
            top = int(box["top"] * h)
            right = int((box["left"] + box["width"]) * w)
            bottom = int((box["top"] + box["height"]) * h)
 
            ai_boxes.append({
                "label": pred["tagName"],
                "xmin": left,
                "ymin": top,
                "xmax": right,
                "ymax": bottom
            })
 
            draw.rectangle([left, top, right, bottom], outline="red", width=5)
            draw.text((left, top), f"{pred['tagName']} ({pred['probability']:.2f})", fill="red")
 
    return image_with_boxes, ai_boxes

#──────────────────────────────────────────────────────────────
# 업로드 처리
#──────────────────────────────────────────────────────────────
def handle_upload(image: Image.Image):
    ai_img, ai_boxes = detect_with_boxes(image)
    annotator_input = {
        "image": np.array(image.convert("RGB")),
        "annotations": []
    }
    return ai_img, annotator_input, ai_boxes, image

#──────────────────────────────────────────────────────────────
# 박스 비교 및 시각화
#──────────────────────────────────────────────────────────────
def compare_boxes(user_data, ai_boxes):
    if not user_data or "boxes" not in user_data:
        return "❌ 사용자 태깅 없음", None, []
 
    img_array = user_data["image"]
    user_boxes = user_data["boxes"]
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
 
    matched_count = 0
    results_to_save = []
    used_ai = set()
    used_user = set()
 
    for u_idx, ubox in enumerate(user_boxes):
        user = {
            "xmin": ubox["xmin"],
            "ymin": ubox["ymin"],
            "xmax": ubox["xmax"],
            "ymax": ubox["ymax"]
        }
 
        best_iou = 0
        matched_ai_idx = -1
        for i, abox in enumerate(ai_boxes):
            iou = calculate_iou(user, abox)
            if iou > best_iou:
                best_iou = iou
                matched_ai_idx = i
 
        if best_iou >= 0.5:
            matched_count += 1
            used_ai.add(matched_ai_idx)
            used_user.add(u_idx)
            draw.rectangle([user["xmin"], user["ymin"], user["xmax"], user["ymax"]], outline="green", width=5)
        else:
            draw.rectangle([user["xmin"], user["ymin"], user["xmax"], user["ymax"]], outline="yellow", width=5)
 
        results_to_save.append({
            "label": ubox["label"],
            "xmin": ubox["xmin"],
            "ymin": ubox["ymin"],
            "xmax": ubox["xmax"],
            "ymax": ubox["ymax"],
            "matched": best_iou >= 0.5,
            "iou": round(best_iou, 2)
        })
 
    for idx, abox in enumerate(ai_boxes):
        if idx not in used_ai:
            draw.rectangle([abox["xmin"], abox["ymin"], abox["xmax"], abox["ymax"]], outline="orange", width=5)
 
    user_only = len(user_boxes) - matched_count
    ai_only = len(ai_boxes) - len(used_ai)
 
    # 점수 계산
    score_match = matched_count * 0.5
    score_user = user_only * 0.3
    score_ai = ai_only * 0.2
    total_score = score_match + score_user + score_ai
 
    msg = (
        f"✅ 비교 완료!\n"
        f"- 일치한 태그: {matched_count}/{len(user_boxes)}개\n"
        f"- 사용자만 태깅한 박스: {user_only}개\n"
        f"- AI만 감지한 박스: {ai_only}개\n"
        f"\n"
        f"📊 총점: {total_score:.1f}점 (일치: {score_match:.1f}, 사용자만: {score_user:.1f}, AI만: {score_ai:.1f})"
    )
 
    return msg, img, results_to_save

#──────────────────────────────────────────────────────────────
# 결과 저장
#──────────────────────────────────────────────────────────────
def save_results(image: Image.Image, results_to_save):
    os.makedirs("saved_images", exist_ok=True)
    filename = f"saved_images/image_{np.random.randint(100000)}.jpg"
    image.save(filename)
 
    with open("saved_annotations.json", "a", encoding="utf-8") as f:
        json.dump({"image": filename, "annotations": results_to_save}, f, ensure_ascii=False)
        f.write("\n")
 
    return f"💾 저장 완료: {filename}"


#──────────────────────────────────────────────────────────────
# Gradio UI
#──────────────────────────────────────────────────────────────
with gr.Blocks() as demo :
    gr.Markdown('## 🚧 격자형 빗물받이에 특화된 시범 서비스입니다.')

    with gr.Tabs() :
        # 분류 (clean/heavy) 탭
        with gr.Tab('📸') :
            gr.Markdown('## 🧹 빗물받이 청결도 판별 (AI)')

            # 이미지 메타정보를 사용하기 위해서 type='filepath' 로 지정
            image_input = gr.Image(type='filepath', label='사진을 올려주세요.')
            validation = gr.Textbox(label='이미지 확인')
            prediction = gr.Textbox(label='오염 심각도 확인', visible=False)
            map = gr.HTML(visible=False)
            report_btn = gr.HTML(visible=False)

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, map, report_btn]
            )

        # 개체 감지 (담배꽁초) 탭
        with gr.Tab('🔎') :
            gr.Markdown("## 🧪 담배꽁초 감지 비교 (사용자 vs AI)")

            image_input = gr.Image(type="pil", label="이미지 업로드")
            start_btn = gr.Button("🟦 AI 감지 및 태깅 시작")
        
            with gr.Row(visible=False) as tag_row:
                ai_result = gr.Image(label="🤖 AI 감지 결과")
                annotator = image_annotator(
                    label_list=['cigarette', 'plastic waste', 'paper waste', 'natural object', 'other trash'],
                    label_colors=[(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 255)]
                )
        
            compare_btn = gr.Button("📐 비교", visible=False)
        
            # 사용자 vs AI 비교 영역
            with gr.Row(visible=False) as compare_row:
                compare_result = gr.Image(label="📊 사용자 vs AI 비교 결과")
        
            compare_text = gr.Textbox(label="결과 메시지", visible=False, lines=6)
            save_btn = gr.Button("💾 결과 저장", visible=False)
            save_text = gr.Textbox(label="저장 메시지", visible=False)
        
            # global 변수
            hidden_ai_boxes = gr.State()
            original_image = gr.State()
            temp_save_result = gr.State()
        
            # AI 감지 및 태깅
            start_btn.click(
                fn=handle_upload,
                inputs=image_input,
                outputs=[ai_result, annotator, hidden_ai_boxes, original_image]
            )

            start_btn.click(
                lambda: (gr.update(visible=True),)*2,
                None,
                [tag_row, compare_btn]
            )
        
            # 사용자 vs AI 비교
            compare_btn.click(
                fn=compare_boxes,
                inputs=[annotator, hidden_ai_boxes],
                outputs=[compare_text, compare_result, temp_save_result]
            )

            compare_btn.click(
                lambda: (gr.update(visible=True),)*3,
                None,
                [compare_text, compare_row, save_btn]
            )
        
            # 결과 저장
            save_btn.click(
                fn=save_results,
                inputs=[original_image, temp_save_result],
                outputs=save_text
            )

demo.launch()