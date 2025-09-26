# 트레드밀 보행자 3D 자세 추정 파이프라인 기술 문서

## 01. 개요

> 본 문서는 Orbbec Femto Bolt 스테레오 카메라로 촬영된 트레드밀 보행 영상으로부터 3D 인체 자세를 추정하는 파이프라인에 대해 기술합니다. 해당 파이프라인은 2개의 동기화된 보행 영상을 입력으로 받아, 각 영상으로부터 2D 키포인트를 검출하고, 이를 *삼각측량(Triangulation)*하여 3D 공간상의 좌표를 계산합니다. 최종 결과물로는 3D 키포인트 좌표가 담긴 JSON 파일과 이를 시각화한 3D 스켈레톤 애니메이션 비디오가 생성됩니다.
이 결과물은 향후 보행 주기, 보폭, 관절 각도 등 다양한 보행 파라미터를 정량적으로 분석하는 데 기초 데이터로 활용될 수 있습니다. 

## 02. 시스템 환경 설정

본 파이프라인을 실행하기 위해 필요한 개발 환경을 구축하는 과정에 대해 설명합니다. Conda를 사용하여 `environment.yml` 파일에 명시된 의존성을 설치합니다.

```shell
$conda env create -f environment.yml
$conda activate mmpose_prac
```

- **주요 의존성:**
  - `python=3.9`
  - `pytorch=2.1.0`
  - `cuda-version=12.1`
  - `ffmpeg`
  - `mmcv`, `mmdet`, `mmengine`

### 모델 가중치 및 설정 파일

본 파이프라인의 2D 포즈 추정을 위해, COCO-WholeBody 데이터셋으로 pre-train된 **RTMDet** 객체 탐지 모델과, **DWPose-l** 포즈 추정 모델을 사용합니다. 아래 링크에서 필요한 가중치 파일을 다운로드하여 `checkpoints/` 디렉토리에 저장하십시오.

다운로드 링크는 https://drive.google.com/drive/folders/1wvVR5H_ys0FBo5Er6456_F9vdcVl3Cia?hl=ko 입니다.


### 하드웨어 설정
위 프로젝트를 실행하는 데 필요한 주요 준비물은 아래와 같습니다.

- 카메라 연결용 PC 2대, 포즈 추정 작업용 PC 1대
- 카메라 체커보드 (5 * 4 / 가로 140cm, 세로 110cm)
- Orbbec Femto Bolt RGB 카메라 2대

그리고 전반적인 프로젝트의 실시 과정을 보고 싶으시다면, https://docs.google.com/document/d/1XTRVwmFwfw6isAe-OnH_p8YGGVb0oNgv1HbypMcICXc/edit?tab=t.0 을 참조하시길 바랍니다.

## 03. 전체 워크플로우

해당 섹션에서는, 데이터 수집부터 3D 포즈 추정 결과를 얻기까지의 전체 과정을 설명합니다.



### a. 데이터 수집 및 준비하기 (스크립트 파일 실행 전)

1. 트레드밀이 놓일 공간에 5 * 4 체커보드를 배치하고, 2대의 카메라를 이용하여 체커보드 이미지를 촬영합니다. 
2. 트레드밀을 설치 후, 약 1분 간 보행하는 모습을 2대 이상의 카메라로 동시에 촬영합니다.
3. SSH 서버 등을 활용하여, 캘리브레이션 이미지와 보행 영상 파일을 작업용 PC로 전송합니다.

### b. 3D 포즈 추정 파이프라인 실행

전체 3D 포즈 추정 파이프라인은 `3d_pose_estimation.sh` 스크립트를 통해 실행할 수 있습니다.
```shell
$bash 3d_pose_estimation.sh
```
<details>
<summary>스크립트 상세 내용 보기 (3d_pose_estimation.sh)</summary>

```bash
#!/bin/bash

# --- 설정: 주요 변수 정의 ---
# 이 섹션의 변수들을 자신의 환경에 맞게 수정할 수 있습니다.

# 0. 비디오 전처리 관련 변수
ORIGINAL_VIDEO_0='./video_n_frames/sub_output00.mkv'
ORIGINAL_VIDEO_1='./video_n_frames/sub_output01.mkv'
TRIMMED_VIDEO_DIR='./video_n_frames/trimmed_videos'

# 1. 2D 포즈 추정 모델 관련 변수
DET_CONFIG='demo/mmdetection_cfg/rtmdet_l_8xb32-300e_coco.py'
DET_CHECKPOINT='checkpoints/rtmdet_l_8xb32-300e_coco_20220719_171800-24c67de1.pth'
POSE_CONFIG='configs/body_2d_keypoint/rtmo/coco-wholebody/rtmo-l_16xb16-600e_coco-wholebody-384x288.py'
POSE_CHECKPOINT='checkpoints/rtmo-l_16xb16-600e_coco-wholebody-384x288-64d897c6_20230925.pth'
RESULTS_2D_VIS_DIR='pipeline_results/2d_visualizations'

# 2. 3D 삼각측량 관련 변수
CALIB_DIR='calibration_results'
RESULTS_3D_COORD_DIR='pipeline_results/3d_coordinates'

# 3. 3D 애니메이션 관련 변수
RESULTS_3D_ANIM_DIR='pipeline_results/3d_animations'

# --- 파이프라인 실행 ---

# --- 0. 비디오 전처리: 두 비디오의 길이를 동일하게 맞춤 ---
echo ">> Step 0: Trimming videos to the same length..."
python3 cut_to_min_frames.py --video_dir ./video_n_frames

# 전처리된 비디오 경로 설정
TRIMMED_VIDEO_0="$TRIMMED_VIDEO_DIR/trimmed_$(basename $ORIGINAL_VIDEO_0)"
TRIMMED_VIDEO_1="$TRIMMED_VIDEO_DIR/trimmed_$(basename $ORIGINAL_VIDEO_1)"

# --- 1. 2D 포즈 추정: 각 비디오에서 2D 키포인트 검출 ---
echo ">> Step 1: Running 2D pose estimation for Video 0..."
python3 demo/topdown_demo_with_mmdet_filter.py \
    "$DET_CONFIG" "$DET_CHECKPOINT" "$POSE_CONFIG" "$POSE_CHECKPOINT" \
    --input "$TRIMMED_VIDEO_0" \
    --output-root "$RESULTS_2D_VIS_DIR" \
    --draw-bbox --kpt-thr 0.5 --bbox-thr 0.3 --save-predictions

echo ">> Step 1: Running 2D pose estimation for Video 1..."
python3 demo/topdown_demo_with_mmdet_filter.py \
    "$DET_CONFIG" "$DET_CHECKPOINT" "$POSE_CONFIG" "$POSE_CHECKPOINT" \
    --input "$TRIMMED_VIDEO_1" \
    --output-root "$RESULTS_2D_VIS_DIR" \
    --draw-bbox --kpt-thr 0.5 --bbox-thr 0.3 --save-predictions

# 2D 결과 JSON 파일 경로 설정
JSON_0="$RESULTS_2D_VIS_DIR/results_$(basename $TRIMMED_VIDEO_0 .mkv).json"
JSON_1="$RESULTS_2D_VIS_DIR/results_$(basename $TRIMMED_VIDEO_1 .mkv).json"

# --- 2. 3D 삼각측량: 2D 키포인트를 3D 좌표로 변환 ---
echo ">> Step 2: Running 3D triangulation..."
python3 3d_triangulation.py \
    --json-path0 "$JSON_0" \
    --json-path1 "$JSON_1" \
    --output-dir "$RESULTS_3D_COORD_DIR" \
    --calib-file-cam0 "$CALIB_DIR/calibration_result_calb0.npz" \
    --calib-file-cam1 "$CALIB_DIR/calibration_result_calb1.npz" \
    --calib-file-stereo "$CALIB_DIR/stereo_calibration_result.npz"

# 3D 결과 JSON 파일 경로 설정
JSON_3D="$RESULTS_3D_COORD_DIR/$(basename $TRIMMED_VIDEO_0 .mkv)_3d_results.json"

# --- 3. 3D 애니메이션 생성: 3D 좌표를 비디오로 시각화 ---
echo ">> Step 3: Generating 3D animation..."
python3 3d_floating_skeletons.py \
    --json-file "$JSON_3D" \
    --output-dir "$RESULTS_3D_ANIM_DIR"

echo ">> All steps completed successfully!"
```
</details>

#### 1. 비디오 전처리 (`cut_to_min_frames.py`)

두 입력 영상의 프레임 수를 확인하고, 길이가 더 짧은 영상에 맞춰 다른 영상을 잘라내고, 보행 파라미터 추출에 도움이 되지 않고 키포인트 검출에 비교적 어려운 영상의 앞/뒤 끝부분을 잘라내어, 두 영상의 길이를 동일하게 만들고, 프레임을 맞춰 줍니다. 이는 후속 3D 삼각측량 단계에서 프레임별로 2D 키포인트를 매칭하기 위해 필수적인 과정입니다.

본 프로젝트에선, `./video_n_frames` 디렉토리의 두 동기화 영상(`sub_output00.mkv`, `sub_output01.mkv`)을 입력으로 사용하면, 처리된 영상이 동일 디렉토리 내의 `./trimmed_videos/` 폴더에 저장됩니다.



#### 2. 카메라 캘리브레이션 (`inner_params.py`, `stereo_calibration.py`)

-   **내장 파라미터 (Intrinsics) 캘리브레이션:**
    -   **원리:** 각 카메라의 고유한 광학적 특성(초점 거리, 주점, 렌즈 왜곡 계수 등)을 찾는 과정입니다. 3D 공간의 점이 2D 이미지 평면에 어떻게 투영되는지를 정의합니다.
    -   **준비물:** 각 카메라 라벨에 걸맞는 디렉토리(`cam0`, `cam1`), 그리고 그 안의 체커보드 이미지 각각 30장
    -   **구현:** `inner_params.py`는 체커보드 이미지들에서 코너를 검출(`cv2.findChessboardCorners`)하고, 이를 이용해 `cv2.calibrateCamera` 함수로 카메라 행렬(matrix)과 왜곡 계수(distortion coefficients)를 계산하여 `.npz` 파일로 저장합니다. 

-   **스테레오 (외장 파라미터, Extrinsics) 캘리브레이션:**
![alt text](image-1.png)
    -   **원리:** 두 카메라 간의 상대적인 기하학적 관계, 즉 한 카메라 좌표계를 기준으로 다른 카메라가 얼마나 회전(Rotation)하고 이동(Translation)했는지를 계산합니다. 이 과정에서 월드 좌표계의 기준을 체커보드로 재설정하여(`cv2.solvePnP`), 체커보드 평면을 지면(z=0)으로 삼는 새로운 좌표계를 정의합니다.
    -   **준비물:** 'outward_calib' 디렉토리 내, 각각 체커보드 이미지(`new_calib0.jpg`, `new_calib1.jpg`)
    -   **구현:** `stereo_calibration.py`는 두 카메라의 내장 파라미터와 두 카메라로 동시에 촬영한 체커보드 이미지를 `cv2.stereoCalibrate` 함수에 입력하여, 두 카메라 간의 회전 행렬과 이동 벡터를 계산하고 `.npz` 파일로 저장합니다. 이 변환을 통해 아래 이미지와 같이 체커보드 평면이 새로운 월드 좌표계의 기준면(XY 평면)이 됩니다.

    -    **체커보드 기준 좌표계(카메라 0)**
    ![체커보드 기준 좌표계 (카메라 0)](cam0_with_axes.jpg)
    -    **체커보드 기준 좌표계(카메라 1)**
    ![체커보드 기준 좌표계 (카메라 1)](cam1_with_axes.jpg)



#### 3. 2D 포즈 추정 (`topdown_demo_with_mmdet_filter.py`)

Top-Down 방식으로 2D 포즈를 추정합니다.
1.  **객체 탐지 (Detection):** DWPose-l 모델을 사용하여 영상의 각 프레임에서 사람의 위치를 바운딩 박스(Bounding Box)로 먼저 찾아냅니다.
2.  **포즈 추정 (Pose Estimation):** 검출된 각 바운딩 박스 영역 내에서 DWPose-l 모델을 적용하여 14개의 신체 주요 키포인트(관절)의 2D 좌표를 추정합니다. 이 때, 사람을 탐지하는 바운딩 박스의 신뢰도(confidence score)는 `0.3` 이상(`--bbox-thr 0.3`), 각 키포인트의 신뢰도는 `0.5` 이상(`--kpt-thr 0.5`)인 경우에만 유효한 것으로 간주합니다.

그리고 본 파이프라인에서 검출할 키포인트는 다음과 같습니다.
```
-   `5`: left_shoulder
-   `6`: right_shoulder
-   `11`: left_hip
-   `12`: right_hip
-   `13`: left_knee
-   `14`: right_knee
-   `15`: left_ankle
-   `16`: right_ankle
-   `17`: left_big_toe
-   `18`: left_small_toe
-   `19`: left_heel
-   `20`: right_big_toe
-   `21`: right_small_toe
-   `22`: right_heel
```

결과는 `pipeline_results/2d_visualizations/` 디렉토리에 프레임별 키포인트 좌표가 담긴 JSON 파일과, 이를 시각화한 비디오로 저장됩니다.

#### 4. 3D 삼각측량 (`3d_triangulation.py`)
![카메라 투영 행렬](image-2.png)
-   **원리:** 두 개 이상의 시점에서 얻은 2D 관측값(키포인트)과 카메라의 내/외장 파라미터를 이용하여 3D 공간상의 실제 위치를 복원하는 기하학적 기법입니다. 각 카메라로부터 3D 포인트를 향하는 광선(ray)을 긋고, 이 광선들이 가장 가깝게 교차하는 지점을 3D 포인트로 추정합니다. 쉽게 말해, 삼각측량은 위 행렬식으로 표현되는 **투영(Projection)의 역연산** 과정입니다. 투영이 3D 공간의 점(X,Y,Z)을 2D 이미지 좌표(u,v)로 변환하는 것이라면, 삼각측량은 2대 이상의 카메라에서 얻은 2D 좌표(u,v)와 카메라 파라미터를 이용해 원래의 3D 공간 좌표(X,Y,Z)를 복원하는 기법입니다.
-   **구현:** `3d_triangulation.py`는 특정 프레임에서 **두 카메라 뷰에 공통으로 검출된 키포인트**만을 대상으로 3D 좌표를 계산합니다. 이 공통 키포인트들의 2D 좌표와 캘리브레이션으로 얻은 투영 행렬(Projection Matrix)을 `cv2.triangulatePoints` 함수에 전달하여 3D 좌표(x, y, z)를 계산하고, 결과는 `pipeline_results/3d_coordinates/`에 JSON 파일로 저장됩니다.
-    **재투영 오차(Reprojection Error)의 측정:**
        -   **원리:** 삼각측량으로 계산된 3D 포인트가 얼마나 정확한지를 평가하는 핵심 지표입니다. 계산된 3D 포인트를 다시 각 카메라의 2D 이미지 평면으로 투영(re-project)했을 때, 이 재투영된 2D 포인트와 원래의 2D 키포인트 검출 값 사이의 픽셀 거리(Euclidean distance)를 의미합니다.
        -   **의의:** 이 오차가 작을수록 3D 복원이 2D 관측 결과와 잘 부합함을 의미하며, 카메라 캘리브레이션과 3D 복원 결과의 신뢰도를 정량적으로 평가할 수 있습니다. 오차가 특정 임계값보다 큰 3D 포인트는 부정확한 것으로 간주하고 후처리 과정에서 제거하거나 보간(interpolation)하는 기준으로 삼을 수 있습니다.
-   **후처리: 누락된 키포인트 보간 (Interpolation):**
    -   **문제점:** 특정 프레임에서 키포인트가 한쪽 카메라에 의해 가려지거나(Occlusion), 2D 포즈 추정에 실패하면 해당 키포인트의 3D 좌표를 계산할 수 없어 데이터가 누락됩니다.
    -   **해결:** `3d_triangulation.py` 스크립트는 `interpolate_missing_keypoints` 함수를 통해 이러한 누락 데이터를 후처리합니다. 짧은 구간(기본값: 5프레임 이하)의 데이터 누락이 발생했을 경우, 누락 직전과 직후의 3D 좌표를 이용한 **선형 보간(Linear Interpolation)**을 통해 중간 좌표들을 채워 넣습니다.
    -   **의의:** 이 과정을 통해 불완전한 3D 스켈레톤 데이터를 보완하여, 더 부드럽고 연속적인 3D 애니메이션을 생성하고 향후 보행 분석의 정확도를 높일 수 있습니다.

#### 5. 3D 애니메이션 생성 (`3d_floating_skeletons.py`)
[3D 애니메이션 비디오 예시 보기 (click to watch)](pipeline_results/3d_animations/trimmed_sub_output_3d_floating_animation.mp4)

`3d_triangulation.py`로 생성된 3D 키포인트 JSON 파일을 읽어옵니다. Matplotlib의 `FuncAnimation`을 사용하여 각 프레임의 3D 스켈레톤을 그리고, 이를 `FFMpegWriter`를 통해 원본 영상의 fps에 맞춘 `.mp4` 비디오 파일로 인코딩하여 저장합니다. 이를 통해 시간에 따른 자세 변화를 직관적으로 확인할 수 있습니다.




위 스크립트의 실행이 끝나면, 출력되는 결과 디렉토리 `pipeline_results`의 내부는 아래와 같습니다.

```
pipeline_results/
├── 2d_visualizations/
│   ├── results_trimmed_sub_output00.json
│   ├── results_trimmed_sub_output01.json
│   ├── trimmed_sub_output00.mkv
│   └── trimmed_sub_output01.mkv
├── 3d_animations/
│   └── trimmed_sub_output_3d_floating_animation.mp4
└── 3d_coordinates/
    └── trimmed_sub_output_3d_results.json
```

-   **`2d_visualizations/`**: 각 입력 영상에 대한 2D 포즈 추정 결과가 저장되는 디렉토리입니다.
    -   `results_trimmed_sub_output*.json`: 각 영상의 프레임별 2D 키포인트 좌표 데이터입니다.
    -   `trimmed_sub_output*.mkv`: 2D 포즈 추정 결과를 시각화한 비디오입니다.
-   **`3d_coordinates/`**: 3D 삼각측량을 통해 계산된 최종 3D 키포인트 좌표가 저장되는 디렉토리입니다.
    -   `trimmed_sub_output_3d_results.json`: 전체 프레임에 대한 3D 키포인트 좌표 데이터입니다.
-   **`3d_animations/`**: 추정된 3D 포즈를 시각화한 결과물이 저장되는 디렉토리입니다.
    -   `trimmed_sub_output_3d_floating_animation.mp4`: 3D 스켈레톤 애니메이션 비디오입니다.


## 04. 한계점 및 향후 연구 방향

본 프로젝트는 8주라는 제한된 시간 내에 3D 인체 자세 추정 파이프라인의 전체 워크플로우를 성공적으로 구축했습니다. 하지만 시간적 제약으로 인해 다음과 같은 추가적인 실험을 진행하지 못한 점이 아쉬움으로 남으며, 이는 향후 연구 방향으로 이어질 수 있습니다.

### 1. 향상된 키포인트 보간(Interpolation) 기법 적용

-   **현황 및 한계:** 현재 파이프라인은 짧은 구간의 누락된 3D 키포인트를 선형 보간(Linear Interpolation)으로 처리합니다. 이 방법은 간단하고 빠르지만, 복잡한 움직임이나 긴 구간의 데이터 누락이 발생했을 때 실제 움직임과 차이가 발생할 수 있습니다.
-   **향후 연구 방향:** 사람의 움직임(영상 프레임)에 따른 키포인트들은 시간적 연속성을 갖는 Sequential Data입니다. 따라서 **Transformer**나 **LSTM**과 같은 시계열 데이터 처리에 특화된 딥러닝 모델을 학습하여 누락된 키포인트를 예측하고 채워 넣는다면, 훨씬 더 자연스럽고 정확한 움직임을 복원할 수 있을 것입니다.

### 2. 다양한 2D 포즈 추정 모델 성능 비교

-   **현황 및 한계:** 본 프로젝트에서는 우수한 성능으로 널리 알려진 **RTMDet**과 **DWPose-l** 모델을 2D 포즈 추정에 사용했습니다. 하지만 이 모델이 현재의 트레드밀 보행 환경에 최적이라고 단정하기는 어렵습니다.
-   **향후 연구 방향:** 더 많은 시간이 주어졌다면, **YOLO 계열**의 다른 객체 탐지 모델이나, **RTMPose**, **HRNet** 등 다양한 2D 포즈 추정 모델들을 동일한 데이터셋으로 테스트하고 성능을 비교 분석하는 작업을 수행할 수 있었을 것입니다. 이를 통해 특정 환경(예: 낮은 조도, 빠른 움직임)에서 더 강건한 모델을 선택하여 파이프라인 전체의 정확도를 향상시킬 수 있습니다.
