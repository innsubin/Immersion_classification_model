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

            # 랜드마크 포인트 draw
            # face mesh 모델 결과가 있을 때 실행
            if result_fm.multi_face_landmarks:
                        # face mesh 결과를 반복문 진행
                        # 위와 동일하게 환경이 조절되기 때문에 실질적으로는 한번만 실행
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
                           
                            draw_point_68 = [
                            227, 137, 177, 215, 172, 136, 150, 176,     # chin_left
                            152,                                        # chin_center
                            400, 379, 365, 397, 435, 401, 366, 447,     # chin_right
                            70, 63, 105, 66, 107,                       # eyebrow_left
                            336, 296, 334, 293, 300,                    # eyebrow_right
                            6, 197, 195, 5,                             # nose
                            98, 97, 2, 326, 327,                        # nose_lower
                            33, 160, 158, 133, 153, 144,                # eye_left
                            398, 385, 387, 263, 373, 380,               # eye_right
                            61,	                                        # lip_left
                            40, 39, 37, 0, 267, 269, 270,               # lip_upper
                            291,                                        # lip_right
                            91, 181, 17, 405, 321,                      # lip_lower
                            82, 13, 312,                                # lip_upper_lower
                            178, 14, 402                                # lip_lower_upper
                            ]

                            draw_point_68 = [0, 2, 5, 6, 13, 14, 17, 33, 37, 39, 
                                            40, 61, 63, 66, 70, 82, 91, 97, 98, 105, 
                                            107, 133, 136, 137, 144, 150, 152, 153, 158, 160,
                                            172, 176, 177, 178, 181, 195, 197, 215, 227, 263,
                                            267, 269, 270, 291, 293, 296, 300, 312, 321, 326,
                                            327, 334, 336, 365, 366, 373, 379, 380, 385, 387, 
                                            397, 398, 400, 401, 402, 405, 435, 447]
                            
                            dict = {'chin_left' : [227, 137, 177, 215, 172, 136, 150, 176],
                                    'chin_center' : [152],
                                    'chin_right' : [400, 379, 365, 397, 435, 401, 366, 447],
                                    'eyebrow_left' : [70, 63, 105, 66, 107],
                                    'eyebrow_right' : [336, 296, 334, 293, 300],
                                    'nose' : [6, 197, 195, 5],
                                    'nose_lower' : [98, 97, 2, 326, 327],
                                    'eye_left' : [33, 160, 158, 133, 153, 144],
                                    'eye_right' : [398, 385, 387, 263, 373, 380],
                                    'lip_left' : [61],
                                    'lip_upper' : [40, 39, 37, 0, 267, 269, 270],
                                    'lip_right' : [291],
                                    'lip_lower' : [91, 181, 17, 405, 321],
                                    'lip_upper_lower' : [82, 13, 312],
                                    'lip_lower_upper' : [178, 14, 402]}
                            
                            dict = {0:'lip_upper', 2:'nose_lower', 5:'nose', 6:'nose', 13:'lip_upper_lower', 14:'lip_lower_upper', 17:'lip_lower', 33:'eye_left', 
                                    37:'lip_upper', 39:'lip_upper', 40:'lip_upper', 61:'lip_left', 63:'eyebrow_left', 66:'eyebrow_left', 70:'eyebrow_left', 
                                    82:'lip_upper_lower', 91:'lip_lower', 97:'nose_lower', 98:'nose_lower', 105:'eyebrow_left', 107:'eyebrow_left', 133:'eye_left', 
                                    136:'chin_left', 137:'chin_left', 144:'eye_left', 150:'chin_left', 152:'chin_center', 153:'eye_left', 158:'eye_left', 160:'eye_left', 
                                    172:'chin_left', 176:'chin_left', 177:'chin_left', 178:'lip_lower_upper', 181:'lip_lower', 195:'nose', 197:'nose', 215:'chin_left', 
                                    227:'chin_left', 263:'eye_right', 267:'lip_upper', 269:'lip_upper', 270:'lip_upper', 291:'lip_right', 293:'eyebrow_right', 
                                    296:'eyebrow_right', 300:'eyebrow_right', 312:'lip_upper_lower', 321:'lip_lower', 326:'nose_lower', 327:'nose_lower', 
                                    334:'eyebrow_right', 336:'eyebrow_right', 365:'chin_right', 366:'chin_right', 373:'eye_right', 379:'chin_right', 380:'eye_right', 
                                    385:'eye_right', 387:'eye_right', 397:'chin_right', 398:'eye_right', 400:'chin_right', 401:'chin_right', 402:'lip_lower_upper',
                                    405:'lip_lower', 435:'chin_right', 447:'chin_right'}

                            for i in draw_point_68:
                                x, y = landmark_x[i], landmark_y[i]
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        except Exception as e:
            print(e)

        # 프레임 출력
        cv2.imshow("Check Immersion", frame)
        
        # esc 눌러서 종료
        if cv2.waitKey(33) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
                   
if __name__ == "__main__":
    main()