import gradio as gr
import cv2
import torch
import io
import os
import requests
import numpy as np
import pandas as pd
import re
import folium
import PIL.Image
from gradio_modal import Modal
from PIL.ExifTags import TAGS

# Custom Vision Predictioin 정보 - 빗물받이 여부 모델
PREDICTION_KEY = 'BBvYKDdr5RDpSMjG34Z2XXw3hLxzlAQkktCPXwHTLleSagQPHGg0JQQJ99BEACYeBjFXJ3w3AAAIACOGH9bC'
ENDPOINT_URL = 'https://7aiteam05cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/58b52583-2cfb-4767-b9e0-8e83032f9d95/classify/iterations/Iteration3/image'

# API 호출 시, 사용할 헤더 셋팅
headers = {
    'Prediction-Key' : PREDICTION_KEY,
    # 바이너리 이미지 전송
    'Content-Type' : 'application/octec-stream'
}

# 이미지 처리
def process_image(image_path) :
    # 이미지가 삭제된 경우, 모든 셋팅 초기화
    if image_path is None :
        return '', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), (37.566535, 126.9779692), 0, 0

    # 이미지 빗물받이 여부 판단
    label, probability = predict_with_api(image_path)
    is_valid = label == 'service'
    validation_msg = f'✅유효한 사진입니다. (예측 : {(probability * 100) :.0f}%)' if is_valid else '🚫유효하지 않은 사진입니다.'

    # 빗물받이가 아닌 경우,
    if not is_valid :
        return validation_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), (37.566535, 126.9779692), 0, 0
    
    # 빗물받이인 경우,
    # 1. 심각도 예측    
    # pred_severity = predict_with_api(image_path)
    # print(pred_severity)
    # probability = pred_severity.get('probability')
    probability = 0

    # 2. 예측 결과를 바탕으로 깊이 고려하여 점수 환산
    # 깊이 고려 가중치
    depth = midas(image_path)
    avg_depth = np.mean(depth)
    # print(avg_depth)

    # 점수 계산
    total_score = 0

    # 30점 이하 : 깨끗
    # 30점 초과 ~ 70점 미만 : 주의 요망
    # 70점 이상 : 위험
    if total_score <= 30 :
        warning_msg = '🟢 깨끗'
    elif total_score >= 70 :
        warning_msg = '🔴 위험'
    else :
        warning_msg = '🟡 주의 요망'
    
    # 3. GPS 정보 추출
    gps = get_image_gps(image_path)
    # 서울 중심
    map = folium.Map(location=[37.566535, 126.9779692], zoom_start=11)
    folium.Marker(location=[gps[0], gps[1]], icon=folium.Icon(color='red', icon='star')).add_to(map)
    map_html = map._repr_html_()

    return validation_msg, gr.update(value=warning_msg, visible=True), gr.update(value=map_html, visible=True), gr.update(visible=True), gps, probability, total_score

# 업로드 이미지 상대적 깊이 추출
def midas(image_path) :
    # PIL 이미지를 OpenCV 이미지로 변환
    image = PIL.Image.open(image_path)
    pil_img = np.array(image)
    img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)

    # MiDaS 모델 가져오기
    model_type = 'DPT_Large'
    midas = torch.hub.load('intel-isl/MiDas', model_type)

    # GPU 사용 가능하다면 GPU 사용
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    midas.to(device)
    midas.eval()

    # 모델 tranform 객체 가져오기
    midas_transforms = torch.hub.load('intel-isl/MiDas', 'transforms')
    if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid' :
        transform = midas_transforms.dpt_transform
    else :
        transform = midas_transforms.small_transform

    # 모델 tranform
    input_batch = transform(img).to(device)

    with torch.no_grad() :
        # 예측
        prediction = midas(input_batch)
        # 크기 변경
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        # 결과
        result = prediction.cpu().numpy()
    return result

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
def predict_with_api(image_path) :
    image = PIL.Image.open(image_path)

    # PIL 이미지 객체 -> JPEG 형식의 바이너리 데이터로 변환
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_data = buf.getvalue()

    # API 호출
    response = requests.post(ENDPOINT_URL, headers=headers, data=byte_data)
    predictions = response.json()['predictions']

    # 확률이 가장 높은 예측 항목 선택
    top_prediction = max(predictions, key=lambda x : x['probability'])
    label = top_prediction['tagName']
    probability = top_prediction['probability']

    return label, probability

def submit_form(mobile, privacy, gps, pred_severity, total_score) :
    # print(mobile, privacy, gps, pred_severity, total_score)
    lat, lon = gps

    result_msg = ''
    error_msg = ''

    # 개인정보 이용내역 동의 시에만 전송
    if privacy is None :
        error_msg = '개인정보(휴대전화) 이용에 동의를 해야 접수가 가능합니다.'
        raise gr.Error(error_msg)
    
    # 개인정보 이용내역 동의 시에만 전송
    if mobile is None :
        error_msg = '휴대전화 정보를 입력해주세요.'
        raise gr.Error(error_msg)
    
    # 휴대전화 번호 유효성 검사 (추후 인증번호 수신 모듈 연결하는걸로 확장)
    pattern = r'\d{3}-\d{3,4}-\d{4}'
    if not re.match(pattern, mobile) :
        error_msg = '휴대전화 정보를 올바른 형식(예: 010-1234-5678)으로 입력해주세요.'
        raise gr.Error(error_msg)
    
    # csv 파일 가져오기 (없으면 새로 생성)
    if os.path.exists('reportDB.csv') :
        df = pd.read_csv('reportDB.csv')
    else :
        df = pd.DataFrame()

    # 신규 데이터 저장
    data = {
        'mobile' : mobile,
        'lat' : lat,
        'lon' : lon,
        'pred_severity' : pred_severity,
        'total_score' : total_score
    }
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)

    # 기존 csv 파일 덮어쓰기
    df.to_csv('reportDB.csv', index=False)

    result_msg = '접수가 완료되었습니다.'
    
    return result_msg, gr.update(visible=False), gr.update(value=''), gr.update(value=False)

with gr.Blocks() as demo :
    gr.Markdown('## 🚧 격자형 빗물받이에 특화된 시범 서비스입니다.')

    # global 변수
    # 위치 정보
    gps_state = gr.State()
    # 예측 확률
    pred_severity_state = gr.State()
    # 깊이 고려 환산 점수
    total_score_state = gr.State()

    with gr.Tabs() :
        # 사용자 이미지 업로드
        with gr.Tab('📸') :
            # 이미지 메타정보를 사용하기 위해서 type='filepath' 로 지정
            image_input = gr.Image(type='filepath', label='사진을 올려주세요.')
            validation = gr.Textbox(label='이미지 확인')
            prediction = gr.Textbox(label='오염 심각도 확인', visible=False)
            map = gr.HTML(visible=False)
            apply_btn = gr.Button('신고 접수', visible=False)

            # 이미지 업로드
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[validation, prediction, map, apply_btn, gps_state, pred_severity_state, total_score_state]
            )

            # 신고 접수 버튼 클릭 시, 노출되는 접수창
            with Modal(visible=False) as report_form :
                mobile_input = gr.Textbox(label='휴대전화')
                privacy_chk = gr.Checkbox(label='개인정보(휴대전화 번호) 이용 동의')
                gr.Markdown('#### 서비스 이용 시, 수집하는 개인정보(휴대전화 번호)는 신고접수 서비스를 하기 위함입니다. 사용자의 개인정보는 이용목적 달성 시, 지체없이 파기합니다.')
                submit_btn = gr.Button('제출')

            apply_btn.click(
                fn=lambda: Modal(visible=True),
                outputs=[report_form]
            )

            submit_result = gr.Textbox(visible=False, label='알림')

            submit_btn.click(
                fn=submit_form,
                inputs=[mobile_input, privacy_chk, gps_state, pred_severity_state, total_score_state],
                outputs=[submit_result, report_form, mobile_input, privacy_chk]
            )

        # 신고 접수현황 - 스쿨어택
        with gr.Tab('🏫') :
            # school_df = pd.read_csv('')
            school_df = pd.DataFrame({
                '학교명' : ['A 초등학교', 'B 초등학교', 'C 초등학교', 'D 초등학교', 'E 초등학교'],
                '신고건수' : [10, 5, 2, 23, 1]
            })

            # 데이터 가공
            school_df = school_df.sort_values(by='신고건수', ascending=False)
            school_df = school_df.reset_index(drop=True)

            # 1~3순위 표시
            medals = ['🥇', '🥈', '🥉']
            for i in range(3) :
                school_df.loc[i, '학교명'] = f'{medals[i]} {school_df.loc[i, "학교명"]}'
            
            gr.DataFrame(value=school_df)
        # 신고 접수현황 - 공무원 대상
        with gr.Tab('📋') :
            # report_df = pd.read_csv('')
            report_df = pd.DataFrame({})
            gr.DataFrame(value=report_df)

demo.launch()