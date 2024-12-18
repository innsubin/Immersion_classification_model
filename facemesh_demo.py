'''
facemesh 랜드마크를 이용하여 몰입도-감정 분류
작성자: 임수빈
작성일: 230804
'''
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
import time

def main():
    # 카메라 권한 접근
    capture = cv2.VideoCapture(0)

    # 카메라 가로 세로 길이 설정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # mediapipe 그리기 도구 로드
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # mediapipe 모델 로드
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose_detection = mp.solutions.pose

    # mediapipe 모델 속성값 설정
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_detection = mp_pose_detection.Pose(min_detection_confidence=0.5)

    # 몰입도 분류 모델 로드
    model = tf.lite.Interpreter("best.tflite")

    # 모델에 메모리 할당
    model.allocate_tensors()

    # 모델의 인풋, 아웃풋 속성 로드
    model_input = model.get_input_details()
    model_output = model.get_output_details()
    pred5 = -1

    ####################################################
    ####################################################
    ####################################################

    ## 두 랜드마크 포인트 사이의 거리 측정
    def calculate_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    ## 세 랜드마크 포인트 중 한 점이 나머지 두 점 사이에 위치할 때, 끼인각 측정
    def calculate_angle_between_points(point1, point2, point3):
        # 두 벡터를 이용하여 라디안 각도 계산
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        norm1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        norm2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        # 두 벡터의 크기가 0이면 각도를 구할 수 없음 (두 점이 같은 위치)
        if norm1 == 0 or norm2 == 0:
            return None
        cos_theta = dot_product / (norm1 * norm2)
        radian = math.acos(max(-1, min(1, cos_theta)))  # acos의 인수는 -1 ~ 1 범위여야 함
        angle = math.degrees(radian)
        # 각도가 0 ~ 180도 사이로 변환
        if vector1[0] * vector2[1] - vector1[1] * vector2[0] > 0:
            angle = 360 - angle
        return angle


    print("화면에서 무표정으로 대기하세요")
    time.sleep(1)
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_fm = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    ## facemesh의 468개 인덱스들의 x, y 좌표값
    landmark_x = [int(value.x * frame.shape[1]) for value in result_fm.multi_face_landmarks[0].landmark]
    landmark_y = [int(value.y * frame.shape[0]) for value in result_fm.multi_face_landmarks[0].landmark]


    ## 무표정, 정자세일 때 필요한 값들을 저장(기준이 되는 초기값 설정)

    ## 정면을 보고 있다고 판단했을 때, 여전히 정면인지 숙였는지 젖혔는지 - 각도 이용
    nose_tip = (landmark_x[5], landmark_y[5]) # 코 끝 
    nose_down_left = (landmark_x[75], landmark_y[75]) # 코 밑 가장 왼쪽
    nose_down_right = (landmark_x[455], landmark_y[455]) # 코 밑 가장 오른쪽 
    user_result = calculate_angle_between_points(nose_down_left, nose_tip, nose_down_right) # 각도 측정

    ## 눈을 감은 상태에서 하품을 하는지 졸음 판단 - 초기값 입 가로 크기 저장
    landmark_lip_up = (landmark_x[0], landmark_y[0]) # 윗입술 가장 중간
    landmark_lip_down = (landmark_x[17], landmark_y[17]) # 아랫입술 가장 중간
    user_lip = calculate_distance(landmark_lip_up, landmark_lip_down) # 입술 세로 거리 측정

    ## 정면 상태에서 차분함과 흥미로움 분류 - 초기값 입술 가로 길이 비율 저장 
    ## 흥미로움 안에서도 웃음과 놀라움 분류
    left_chin_lip = calculate_distance((landmark_x[215], landmark_y[215]), (landmark_x[61], landmark_y[61])) # 왼쪽 턱 4번째와 입술 왼쪽 가장 끝 사이의 거리
    left_lip_right_lip = calculate_distance((landmark_x[61], landmark_y[61]), (landmark_x[308], landmark_y[308])) # 입술 왼쪽 가장 끝과 입술 오른쪽 가장 끝 사이의 거리
    right_lip_chin = calculate_distance((landmark_x[308], landmark_y[308]), (landmark_x[435], landmark_y[435])) # 오른쪽 턱 4번째와 입술 오른쪽 가장 끝 사이의 거리
    total = left_chin_lip + left_lip_right_lip + right_lip_chin # 세 거리의 합
    user_ratio_center = left_lip_right_lip/total # 세 거리의 합에서 가운데 입술의 비율

    ## 집중하지 않음에서 지루함과 차분함 분류 - 인중 길이 
    nose_point = (landmark_x[94], landmark_y[94]) # 코 밑 중간
    lower_lip_center = (landmark_x[0], landmark_y[0]) # 윗입술 가장 중간
    user_ph = calculate_distance(nose_point, lower_lip_center) # 인중 세로 길이 측정

    # 카메라 실행
    while capture.isOpened():
        # 매 프레임 이미지 로드
        # ret: 정상 작동 여부
        # frame: 현재 프레임의 이미지
        ret, frame = capture.read()
        if not ret:
            print("cannot read camera")
            continue
        # 현재 프레임 이미지 좌우변환
        frame = cv2.flip(frame, 1)
            
        try:
            # mediapipe 모델 사용하기 위해 RGB 채널로 변경
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_fd = face_detection.process(frame)
            result_fm = face_mesh.process(frame)
            result_ps = pose_detection.process(frame)

            # opencv 시각화를 위해 BGR 채널로 변경
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # face detection 모델의 결과가 있으면 실행
            if result_fd.detections:
                # face detection의 결과들 반복문 실행
                # 카메라에 한명만 잡히도록 환경을 조절하기 때문에 실질적으론 한번만 실행될 것
                for detection in result_fd.detections:
                    # 바운딩박스 좌상단, 우하단 포인트 취득
                    top_left, bottom_right = mp_drawing.draw_detection(frame, detection)
                    # face detection 모델로 얻어지는 6개 랜드마크 포인트 역정규화
                    landmark_list = [(points.x, points.y) for points in detection.location_data.relative_keypoints]
                    landmark_list = list(map(lambda p:(int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])), landmark_list))
                    
                    # 오른쪽 눈, 왼쪽 눈, 입, 왼쪽 귀 포인트 저장
                    REye = landmark_list[0]
                    LEye = landmark_list[1]
                    mouth = landmark_list[3]
                    LEar = landmark_list[5]

                    # 긱 포인트들의 거리 계산
                    diff_y = abs(mouth[1] - (LEye[1] + REye[1]) / 2)
                    diff_y = (diff_y - 0.5) / (150 - 0.5)
                    diff_x = abs(LEye[0] - LEar[0])
                    diff_x = (diff_x - 0) / (175 - 0)
                    diff = np.array([diff_x, diff_y], dtype=np.float32).reshape(-1, 2)
                    
                    # 바운딩박스 포인트를 이용해 얼굴 부분 RoI
                    # 모델 인풋 사이즈에 맞게 리사이징 및 정규화
                    image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA).reshape(1, 224, 224, 3)
                    image = image.astype(np.float32)
                    image /= 255

                    # 모델 input detail 순서에 맞게 텐서 입력
                    model.set_tensor(model_input[0]["index"], diff)
                    model.set_tensor(model_input[1]["index"], image)
                    # 모델 실행
                    model.invoke()
                    # 모델 output detail 순서에 맞게 결과 저장
                    pred2 = model.get_tensor(model_output[0]["index"]).argmax()
                    pred5 = model.get_tensor(model_output[1]["index"]).argmax()

                    # 랜드마크 포인트 draw
                    # face mesh 모델 결과가 있을 때 실행
                    if not result_fm.multi_face_landmarks:
                        # face mesh 결과를 반복문 진행
                        # 위와 동일하게 환경이 조절되기 때문에 실질적으로는 한번만 실행

                        ## 랜드마크 인식이 가능하지 않을 때 측정 불가 문구를 화면에 띄움
                        cv2.putText(frame, "Measurement Unavailable", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:

                        for face_landmarks in result_fm.multi_face_landmarks:
                            # 478개의 포인트 역정규화
                            # [x, y, z]로 된 배열이 478개 있는데, x,y만 사용
                            # x와 y를 별개의 리스트에 저장
                            landmark_x = [int(value.x*frame.shape[1]) for value in face_landmarks.landmark]
                            landmark_y = [int(value.y*frame.shape[0]) for value in face_landmarks.landmark]
                            # x, y의 최대값-최소값을 이용해 바운딩박스 그림
                            cv2.rectangle(frame, 
                                          (min(landmark_x), min(landmark_y)), 
                                          (max(landmark_x), max(landmark_y)), 
                                          (255, 255, 0), 
                                          2)
                            
                            # 68개만 그려야됨
                            draw_point_68 = [0, 2, 5, 6, 13, 14, 17, 33, 37, 39, 
                                            40, 61, 63, 66, 70, 82, 91, 97, 98, 105, 
                                            107, 133, 136, 137, 144, 150, 152, 153, 158, 160,
                                            172, 176, 177, 178, 181, 195, 197, 215, 227, 263,
                                            267, 269, 270, 291, 293, 296, 300, 312, 321, 326,
                                            327, 334, 336, 365, 366, 373, 379, 380, 385, 387, 
                                            397, 398, 400, 401, 402, 405, 435, 447]

                            
                            # draw_point_68 = [291]
                            
                            for i in draw_point_68:
                                x, y = landmark_x[i], landmark_y[i]
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            ####################################################
            ####################################################
            ####################################################
            # print((landmark_x[291]))

            # #### 고개 돌림 판단
            # def look_side():
            #     nose_center = (landmark_x[94], landmark_y[94]) # 코 밑 중간
            #     chin_left = (landmark_x[177], landmark_y[177]) # 왼쪽턱 3번째
            #     chin_right = (landmark_x[401], landmark_y[401]) # 오른쪽턱 3번째
            #     chin_nose_left = calculate_distance(chin_left, nose_center) # 왼쪽턱과 코까지의 거리
            #     chin_nose_right = calculate_distance(chin_right, nose_center) # 오른쪽턱과 코까지의 거리
            #     chin_chin = chin_nose_left + chin_nose_right # 두 거리의 합
            #     left_ratio = chin_nose_left/chin_chin # 전체 중 왼쪽의 비율
            #     right_ratio = chin_nose_right/chin_chin # 전체 중 오른쪽의 비율
            #     return max(left_ratio, right_ratio) # 둘 중 더 큰 값을 리턴(정면일 때가 가장 최소, 0.5~1.0)

            # #### 정면 판단
            # def look_front():
            #     nose_tip = (landmark_x[5], landmark_y[5]) # 코 끝
            #     nose_down_left = (landmark_x[75], landmark_y[75]) # 코 밑 가장 왼쪽
            #     nose_down_right = (landmark_x[455], landmark_y[455]) # 코 밑 가장 오른쪽
            #     result = calculate_angle_between_points(nose_down_left, nose_tip, nose_down_right) # 코 끝 부분의 끼인각(측정값)
            #     updown_ratio = result/user_result # 초기값과 현재 측정값의 비율
            #     return updown_ratio

            # #### 눈을 떴는지 감았는지 판단(왼쪽눈 기준)
            # def eye_open():
            #     eye_left_index = [158, 153] # 왼쪽 눈 윗부분, 왼쪽 눈 아랫부분
            #     eye_left_points = [(landmark_x[i], landmark_y[i]) for i in eye_left_index] # 두 눈의 좌표
            #     eye_y = eye_left_points[1][1]-eye_left_points[0][1] # 두 눈의 y 값의 차이
            #     return eye_y 
            
            # #### 하품 판단
            # def yawn():
            #     landmark_lip_up = (landmark_x[0], landmark_y[0]) # 윗입술 가운데
            #     landmark_lip_down = (landmark_x[17], landmark_y[17]) # 아랫입술 가운데 
            #     yawn = calculate_distance(landmark_lip_up, landmark_lip_down) # 입술의 세로 길이
            #     result = yawn / user_lip # 초기값 입술 가로 길이와 하품 했을 때의 차이
            #     return result
            
            # #### 집중 - 차분함과 흥미로움(웃음과 놀람) 판단
            # def interest_or_not():
            #     left_chin_lip = calculate_distance((landmark_x[215], landmark_y[215]), (landmark_x[61], landmark_y[61])) # 턱 4번째와 입술 가장 왼쪽의 거리
            #     left_lip_right_lip = calculate_distance((landmark_x[61], landmark_y[61]), (landmark_x[308], landmark_y[308])) # 입술 양 끝의 거리
            #     right_lip_chin = calculate_distance((landmark_x[308], landmark_y[308]), (landmark_x[435], landmark_y[435])) # 입술 가장 오른쪽과 턱 4번째의 거리
            #     total = left_chin_lip + left_lip_right_lip + right_lip_chin # 세 거리의 총합
            #     ratio_center = left_lip_right_lip/total # 전체 중 입술의 가로 길이 비율
            #     result_ratio = user_ratio_center / ratio_center # 초기값과 현재 측정값과의 차이
            #     return result_ratio
            
            # #### 집중하지않음 - 차분함과 지루함 판단
            # #### 인중 짧아짐
            # def ph():
            #     nose_point = (landmark_x[94], landmark_y[94]) # 코 밑 가운데
            #     lower_lip_center = (landmark_x[0], landmark_y[0]) # 윗입술 가운데
            #     ph = calculate_distance(nose_point, lower_lip_center) # 인중 거리
            #     result = user_ph / ph # 초기값과 현재 측정값과의 차이(비)
            #     return result

            # ########### 분류 ###########
            # if look_side() < 0.67: # 정면응시
            #     if look_front() < 0.81: # 정면응시 - 상
            #         print("고개를 젖혀 위를 바라보고 있습니다.")
            #     elif look_front() < 1.25: # 정면응시 - 중간
            #         if eye_open() < 2.1: # 눈감음
            #             if yawn() > 1.7: # 눈감음 - 하품 o
            #                 print("눈을 감고 하품하고 있습니다. (졸음) ")
            #             else: # 눈감음 - 하품 x
            #                 print("눈을 감고 졸고 있습니다. (졸음) ")
            #         else: # 눈뜸
            #             if interest_or_not() >= 0.8 and interest_or_not() <= 1.15:
            #                 print("눈을 뜨고 정면을 바라보고 있습니다. (집중 - 차분함) ")
            #             else:
            #                 print("눈을 뜨고 정면을 바라보고 있습니다. 표정변화가 있습니다. (집중 - 흥미로움)")
            #     else: # 정면응시 - 하
            #         print("고개를 숙이고 있습니다")
            # else: # 좌 또는 우 응시
            #     if eye_open() > 1.1: # 눈을 떴다고 판단
            #         if ph() >= 1.5:
            #             print("집중하지않음 - 지루함 ")
            #         else:
            #             print("집중하지않음 - 차분함 ")

            ####################################################
            ####################################################
            ####################################################   

        except Exception as e:
            print(e)
   
        # 5개 class를 3개로 그룹화
        # 0:집중, 1:양호
        if pred5 == 0 or pred5 == 1: # 녹색
            # 화면 좌상단에 신호등 표시
            # 이미지, 좌표, 크기, 색상, 내부채우기
            cv2.circle(frame, (70, 20), 10, (0, 255, 0), -1)
        # 2: 졸음
        if pred5 == 2: # 노란색
            cv2.circle(frame, (70, 20), 10, (0, 255, 255), -1)
        # 3:집중하지 않음, 4:딴짓
        if pred5 == 3 or pred5 == 4: # 빨간색
            cv2.circle(frame, (70, 20), 10, (0, 0, 255), -1)

        # 프레임 출력
        cv2.imshow("Check Immersion", frame)
        
        # esc 눌러서 종료
        if cv2.waitKey(33) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
                   
if __name__ == "__main__":
    main()