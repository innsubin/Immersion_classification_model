# Immersion_classification_model

1. 68개 포인트를 추출하는 dlib 라이브러리 모델을 벤치마킹하여 mediapipe face-mesh에서 제공되는 478개 포인트(기본 제공 468 + 눈 주변 10) 중 위치가 유사한 68개 랜드마크 추출

2. 룰베이스 감정-몰입도 분류 모델
   
### 클래스별 가이드라인
+ 집중-흥미로움 : 얼굴과 신체가 카메라 정면을 바라보는 자세에서, 눈을 크게 뜨고 미소를 지으며 촬영
+ 집중-차분함  : 얼굴과 신체가 카메라 정면을 바라보는 자세에서, 무표정한 상태로 촬영
+ 비집중-차분함 : 얼굴 또는 신체가 카메라 정면을 바라보지 않는 자세에서, 무표정한 상태로 촬영
+ 비집중-지루함 : 얼굴 또는 신체가 카메라 정면을 바라보지 않는 자세에서, 인상을 찡그리며 촬영
+ 졸음 : 눈을 감은 상태, 하품
 
#

1. 정면판단
> 정면을 보고 있는지, 고개를 좌 또는 우로 돌리고 있는지(비집중 vs 집중, 졸음). 코끝 좌표와 수직인 선 상에 있는 좌우 턱선 좌표 이용하여 가운데 중심이 되는 좌표에서 각각 좌우로의 거리를 측정하고 이들의 합에서 좌우 각각의 비율을 구한 후, 큰 값을 추출
   ![image](https://github.com/user-attachments/assets/7385a713-837a-453c-8a21-0c3b7d21d7c6)

> * 사용자가 고개를 좌 또는 우로 너무 돌려 랜드마크가 인식되지 않을 때는 '측정 불가' 표시


2. 정면을 보고 있을 때, 고개를 위로 젖혔는지 아래로 숙였는지 정면인지
> 코 끝 점과 코 좌우 양 끝 쪽에 위치한 점들을 이용. 학습 시작 전, 사용자의 처음 정자세에서 코 끝에서 끼인 각의 크기를 측정하고 초기값으로 저장. 이후 달라지는 각과 초기값의 비율을 계산하여 고개를 숙이는지 젖히는지 판단
![image](https://github.com/user-attachments/assets/39fa0b7f-8715-439b-b8b0-3a23afd08fe1)

> 비율로 접근하였을 때, 0.81보다 작으면 고개를 젖히고 있고, 0.81~1.25면 정면 응시, 1.25보다 크면 고개를 숙이고 있다고 판단
> * 몸을 뒤로 기울인 상태에서도 같은 결과값이 나옴


3. 눈을 떴는지 감았는지(집중 vs 졸음)
> 왼쪽 눈의 윗부분과 아랫부분 좌표를 사용. y 좌표의 차이 정도에 따라 나뉨.
![image](https://github.com/user-attachments/assets/08643b98-346b-4cb7-922d-31722e06e646)

> 집중 상태의 무표정을 기준으로 y좌표의 거리가 7\~8, 눈을 감고 조는 상태에서 y좌표의 거리는 0\~2 값이 나옴. 2보다 작으면 눈을 감았고, 1보다 크면 눈을 떴다고 판단
> * 사용자에 따라 눈의 크기가 다르기 때문에 타이트하지 않게 계산


4. 하품유무
> 윗입술의 윗부분과 아랫입술의 아랫부분에 위치한 점 이용하여 두 점의 거리로 판단.
![image](https://github.com/user-attachments/assets/2042fe72-ae3a-4189-8036-57c47890c5d7)

> 하품하지 않는 무표정 상태에서 입술의 세로 길이는 14\~16, 고개를 들고 하품을 하는 상태에서는 50\~51, 턱을 내리고 하품하는 상태에서는 34\~38 이 나옴
> 하품을 하였을 때 가장 작은 값이 나오는 상황을 기준으로 그 비율보다 크면 하품을 한다고 판단하여 현재 입술 세로의 길이를 무표정 값으로 나눴을 때 1.7 값이 넘어가면 하품을 한다고 분류함
> 1.7보다 크면 하품(졸음), 작으면 하품안함

5. 집중 세부 분류(집중-차분함 vs 집중-흥미로움)
> 양쪽 턱 한 좌표에서부터 입술 양 끝 점과의 거리, 입술 가로 길이를 이용해 비율을 구함. 흥미로움에서 미소를 짓거나 입을 오므리는 상황을 생각하여 세 길이의 합에서 입술 가로 길이의 비율이 어떻게 달라지느냐에 따라서 분류함
![image](https://github.com/user-attachments/assets/9126ab67-dc6b-4a15-b6a4-6d48ffedf0a5)

> 무표정일 때는 0.29~0.30, 옅은 미소를 지었을 때는 0.35\~0.37, 놀란 표정을 하며 입술을 모을 때는 0.26\~27 비율이 나옴. 그래서 초기 무표정일 때를 기준으로 이보다 0.8보다 작으면 놀란 표정을 지었고, 1.16보다 크면 미소를 짓고 있다고 판단하였음

5. 비집중 세부 분류(비집중-차분함 vs 비집중-지루함)
> 코 밑 가운데 좌표와 윗입술 가운데 좌표 두 점의 거리를 구해 인중 거리를 구함. 가이드라인 상에서 지루함의 표정은 입술이 위로 올라가면서 인중이 짧아지기 때문에 인중의 길이 변화를 이용하여 판단


#### 보완점
눈을 뜨고 있는지 감았는지에 대한 판단을 왼쪽 눈만 이용하였는데, 양쪽 눈을 다 사용할 수 있는 방법을 찾아볼 예정.
또한, 눈의 윗부분과 아랫부분의 y 좌표 차이로 거리를 측정한 결과 차이값이 정수가 되어 성능이 크게 좋다고 말할 수 없는 상황. 포인트 좌표들의 거리를 계산하는 등 다른 방법을 찾아볼 필요가 있음.
