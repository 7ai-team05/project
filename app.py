import gradio as gr
import folium
import io
import os
import requests
import numpy as np
import pandas as pd
import PIL.Image
from PIL import ImageDraw
from gradio_image_annotation import image_annotator
from PIL.ExifTags import TAGS
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials



# 이미지 처리
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
        return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
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

    return validation_msg, gr.update(value=result_msg, visible=True), gr.update(value=map_html, visible=True), gr.update(visible=True)


# 이미지 위치 정보
def get_image_gps(image_path) :
    # 기본값 (서울 중심)
    lat, lon = 37.566535, 126.9779692

    # 이미지가 삭제된 경우, 모든 셋팅 초기화
    if image_path is None :
        return lat, lon

    # 이미지 불러올 때, 오류가 발생한 경우 기본값 사용
    try :
        image = PIL.Image.open(image_path)
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


# 학습 모델 결과 반환
def predict_with_api(image_path, type='service_or_not') :
    # Custom Vision Predictioin 정보
    PREDICTION_KEY = {
        'service_or_not' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
        'severity' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC',
        'object_detect' : 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC'
    }
        
    ENDPOINT_URL = {
        'service_or_not' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/58b52583-2cfb-4767-b9e0-8e83032f9d95/classify/iterations/Iteration3/image',
        'severity' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/ab4cf356-d250-44f4-9221-12c8560bbee1/classify/iterations/Iteration9/image',
        'object_detect' : 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3e17ba3a-0a1e-44a6-8c7d-237db4c93280/detect/iterations/Iteration1/url'
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


# PIL 이미지 객체 -> JPEG 형식의 바이너리 데이터로 변환
def pil_to_binary(image_path) :
    image = PIL.Image.open(image_path)
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_data = buf.getvalue()

    return byte_data



def detect_image(image_path) :
    # Custom Vision Predictioin 정보
    ENDPOINT_URL = 'https://7aiteam05cv-prediction.cognitiveservices.azure.com'
    PREDICTION_KEY = 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC'
    PROJECT_ID = '3e17ba3a-0a1e-44a6-8c7d-237db4c93280'
    PUBLISHED_NAME = 'Iteration1'

    # Prediction 클라이언트 생성
    credentials = ApiKeyCredentials(in_headers={'Prediction-Key' : PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT_URL, credentials=credentials)

    # 전송할 이미지 (바이너리 형태)
    byte_data = pil_to_binary(image_path)
    image = PIL.Image.open(image_path)
    
    # 이미지 전송 및 예측
    results = predictor.detect_image(PROJECT_ID, PUBLISHED_NAME, byte_data)

    colors = {
        'cigarette' : (255, 0, 0),
        'plastic waste' : (0, 0, 255),
        'paper waste' : (0, 255, 0),
        'natural object' : (255, 0, 255),
        'other trash' : (0, 0, 0)
    }
    
    boxes = []
    for prediction in results.predictions :
        if prediction.probability > 0.5 :
            left = int(prediction.bounding_box.left * image.width)
            top = int(prediction.bounding_box.top * image.height)
            width = int(prediction.bounding_box.width * image.width)
            height = int(prediction.bounding_box.height * image.height)
            label = prediction.tag_name

            boxes.append({
                'xmin' : left,
                'ymin' : top,
                'xmax' : width,
                'ymax' : height,
                'label' : label,
                'color' : colors[label]
            })
    
    dict = {
        'image' : np.array(image),
        'boxes' : boxes
    }
    print(dict)
    
    return dict


def add_new_object(annotations) :
    # csv 파일 가져오기 (없으면 새로 생성)
    file_path = 'object_detection.csv'
    df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
    
    boxes = []
    if annotations['boxes'] :    
        for box in annotations['boxes'] :
            # 신규 데이터 저장
            boxes.append({
                'xmin' : box['xmin'],
                'ymin' : box['ymin'],
                'xmax' : box['xmax'],
                'ymax' : box['ymax'],
                'label' : box['label'],
                'color' : box['color']
            })   

        dict = {
            'image' : PIL.Image.fromarray(annotations['image']),
            'boxes' : boxes
        }
            
        new_row = pd.DataFrame(dict)
        df = pd.concat([df, new_row], ignore_index=True)
        result_msg = '새로운 개체 추가 완료'

    # 기존 csv 파일 덮어쓰기
    df.to_csv(file_path, index=False)

    return gr.update(value=result_msg, visible=True)


# 화면 UI
with gr.Blocks() as demo :
    gr.Markdown('## 🚧 격자형 빗물받이에 특화된 시범 서비스입니다.')

    with gr.Tabs() :
        # 분류 (clean/heavy)
        with gr.Tab('📸') :
            # 이미지 메타정보를 사용하기 위해서 type='filepath' 로 지정
            image_input = gr.Image(type='filepath', label='사진을 올려주세요.')
            validation = gr.Textbox(label='이미지 확인')
            prediction = gr.Textbox(label='오염 심각도 확인', visible=False)
            map = gr.HTML(visible=False)
            apply_btn = gr.Button('안전신문고로 신고하러 가기', visible=False)
            output = gr.HTML()

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, map, apply_btn]
            )

            # 안전신문고 신고
            apply_btn.click(
                fn=lambda: '<script>window.open("https://www.safetyreport.go.kr", "_blank");</script>',
                outputs=output
            )

        # 개체 감지
        with gr.Tab('🔎') :
            # 이미지 업로드
            image_input = gr.Image(type='filepath', label='사진을 올려주세요.')
            detect_btn = gr.Button('개체 감지')
        
            # 개체 감지 내역 노출용 이미지
            annotator = image_annotator(
                label_list=['cigarette', 'plastic waste', 'paper waste', 'natural object', 'other trash'],
                label_colors=[(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 0, 0)],
            )
            save_btn = gr.Button('새로운 개체 추가')
            result = gr.Textbox(label='알림', visible=False)
            
            # 개체 감지 결과 노출
            detect_btn.click(
                fn=detect_image,
                inputs=[image_input],
                outputs=[annotator]
            )

            # 새로운 개체 추가
            save_btn.click(
                fn=add_new_object,
                inputs=[annotator],
                outputs=[result]
            )

demo.launch()