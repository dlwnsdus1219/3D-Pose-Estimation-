#!/bin/bash

# --- 0. 비디오 전처리: 두 비디오의 길이를 동일하게 맞춤 ---
echo ">> Step 0: Trimming videos to the same length..."

# 원본 비디오 파일 경로
ORIGINAL_VIDEO_0='./video_n_frames/sub_output00.mkv'
ORIGINAL_VIDEO_1='./video_n_frames/sub_output01.mkv'
TRIMMED_VIDEO_DIR='./video_n_frames/trimmed_videos'

# 원본 비디오 앞/뒤에서 잘라낼 프레임의 개수 (영상 직접 확인해 보고 적절히 조정할 수 있습니다)
TRIM_FRAMES_START=200
TRIM_FRAMES_END=200

# 내장 파라미터 캘리브레이션 이미지 디렉토리 
CALIB_IMG_DIR_0='calb0'
CALIB_IMG_DIR_1='calb1'

# 외장 파라미터 캘리브레이션 이미지 디렉토리 (체커보드 이미지 재촬영 후 반드시 경로 및 이름 수정 바람!!)
STEREO_CALIB_IMG_0='./outward_calib/new_calib0.jpg'
STEREO_CALIB_IMG_1='./outward_calib/new_calib1.jpg'



# 스테레오 캘리브레이션 관련 설정
CHECKERBOARD_DIMS="5 4"  # 체커보드 내부 코너 개수 (가로, 세로)
SQUARE_SIZE_MM=300       # 체커보드 사각형의 실제 크기

# 결과물 출력 디렉토리 설정
PIPELINE_OUTPUT_DIR='pipeline_results'
RESULTS_2D_VIS_DIR="$PIPELINE_OUTPUT_DIR/2d_visualizations"
RESULTS_3D_COORD_DIR="$PIPELINE_OUTPUT_DIR/3d_coordinates"
RESULTS_3D_ANIM_DIR="$PIPELINE_OUTPUT_DIR/3d_animations"

# 파라미터 파일 경로
INTRINSIC_CALIB_0='calibration_result_calb0.npz'
INTRINSIC_CALIB_1='calibration_result_calb1.npz'
STEREO_CALIB_RESULT='stereo_calibration_result.npz'

# --- 0. 비디오 전처리: 두 비디오의 길이를 동일하게 맞춤 ---
echo ">> Step 0: Trimming videos to the same length..."

# 이전 결과 디렉토리 정리하고 새로 생성하기
rm -rf "$PIPELINE_OUTPUT_DIR"
mkdir -p "$RESULTS_2D_VIS_DIR" "$RESULTS_3D_COORD_DIR" "$RESULTS_3D_ANIM_DIR"

# 비디오를 자르는 Python 스크립트 실행
python3 ./video_n_frames/cut_to_min_frames.py \
    "$ORIGINAL_VIDEO_0" \
    "$ORIGINAL_VIDEO_1" \
    "$TRIMMED_VIDEO_DIR" \
    --trim "$TRIM_FRAMES_START" "$TRIM_FRAMES_END"

# 잘린 비디오 파일 경로를 변수에 할당
VIDEO_0="$TRIMMED_VIDEO_DIR/trimmed_$(basename $ORIGINAL_VIDEO_0)"
VIDEO_1="$TRIMMED_VIDEO_DIR/trimmed_$(basename $ORIGINAL_VIDEO_1)"

echo ">> Videos trimmed and saved in $TRIMMED_VIDEO_DIR"

# FRAMES_DIR='my_frames'
# RESULTS_2D_CAM0='my_results_2d/cam0'
# RESULTS_2D_CAM1='my_results_2d/cam1'
# RESULTS_3D_DIR='my_results_3d'

# # 이전 결과 디렉토리 정리하고 새로 생성
# rm -rf my_results_2d $RESULTS_3D_DIR
# mkdir -p $RESULTS_2D_CAM0 $RESULTS_2D_CAM1 $RESULTS_3D_DIR

# 모델 및 설정 파일 경로
DET_CONFIG="/home/kist/GIST/mmpose-main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CHECKPOINT="/home/kist/GIST/mmpose-main/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG="/home/kist/GIST/mmpose-main/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
POSE_CHECKPOINT="/home/kist/GIST/mmpose-main/checkpoints/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth"


echo "===== 3D Pose Pipeline Start (Video Input Method) ====="

# --- 1. 카메라 캘리브레이션 ---
echo ">> Step 1: Camera Calibration..."

# 1-1. 내부 파라미터 캘리브레이션
echo "   - Running intrinsic calibration..."
python3 inner_params.py "$CALIB_IMG_DIR_0" "$CALIB_IMG_DIR1"

# 1-2. 스테레오 캘리브레이션
echo "   - Running stereo calibration..."
python3 stereo_calibration.py \
    --image0 "$STEREO_CALIB_IMG_0" \
    --image1 "$STEREO_CALIB_IMG_1" \
    --calib0 "$INTRINSIC_CALIB_0" \
    --calib1 "$INTRINSIC_CALIB_1" \
    --checkerboard $CHECKERBOARD_DIMS \
    --square_size $SQUARE_SIZE_MM

if [ ! -f "$STEREO_CALIB_RESULT" ]; then
    echo "!!! Stereo calibration failed. Exiting."
    exit 1
fi
echo ">> Camera Calibration Completed."

# --- 2. 2D Pose Estimation ---
## 2D 키포인트 JSON 파일 경로 설정
# JSON_0="$TEMP_2D_JSON_DIR/results_$(basename -s .mkv $VIDEO_0).json"
# JSON_1="$TEMP_2D_JSON_DIR/results_$(basename -s .mkv $VIDEO_1).json"
# JSON_0="$RESULTS_2D_CAM0/results_$(basename -s .mkv $VIDEO_0).json"
# JSON_1="$RESULTS_2D_CAM1/results_$(basename -s .mkv $VIDEO_1).json"

# 2D 결과 파일이 이미 있는가 확인
if [ -f "$JSON_0" ] && [ -f "$JSON_1" ]; then
    echo ">> Step 2: 기존 2D 결과 파일($JSON_0, $JSON_1)을 찾았습니다. 2D 포즈 추정 단계를 건너뜁니다."
else
    echo ">> Step 2: Running 2D pose estimation for Video 0..."
    python3 demo/topdown_demo_with_mmdet_filter.py \
        "$DET_CONFIG" \
        "$DET_CHECKPOINT" \
        "$POSE_CONFIG" \
        "$POSE_CHECKPOINT" \
        --input "$VIDEO_0" \
        --output-root "$RESULTS_2D_VIS_DIR" \
        --draw-bbox \
        --kpt-thr 0.5 \
        --bbox-thr 0.3 \
        --save-predictions

    echo ">> Step 2: Running 2D pose estimation for Video 1..."
    python3 demo/topdown_demo_with_mmdet_filter.py \
        "$DET_CONFIG" \
        "$DET_CHECKPOINT" \
        "$POSE_CONFIG" \
        "$POSE_CHECKPOINT" \
        --input "$VIDEO_1" \
        --output-root "$RESULTS_2D_VIS_DIR" \
        --draw-bbox \
        --kpt-thr 0.5 \
        --bbox-thr 0.3 \
        --save-predictions
fi

echo ">> 2D Pose Estimation Completed for Both Videos."

# --- 3. 3D Triangulation & Floating ---
## 2D 키포인트 JSON 파일 경로 설정
JSON_0="$RESULTS_2D_VIS_DIR/results_$(basename -s .mkv $VIDEO_0).json"
JSON_1="$RESULTS_2D_VIS_DIR/results_$(basename -s .mkv $VIDEO_1).json"
# JSON_0="$RESULTS_2D_CAM0/results_$(basename -s .mkv $VIDEO_0).json"
# JSON_1="$RESULTS_2D_CAM1/results_$(basename -s .mkv $VIDEO_1).json"

## 3D 결과 파일에 사용할 공통 기본 이름 추출 
BASE_NAME_0=$(basename -s .mkv "$VIDEO_0")
COMMON_BASE_NAME=${BASE_NAME_0%00}    # 끝에 붙은 숫자(00, 01 등등) 제거 


echo ">> Step 3: 3D Triangulation 및 플로팅을 실행합니다..."
echo "   - Input JSON 0: $JSON_0"
echo "   - Input JSON 1: $JSON_1"
echo "   - Using calibration files: $INTRINSIC_CALIB_0, $INTRINSIC_CALIB_1, $STEREO_CALIB_RESULT"

python3 3d_triangulation.py \
    "$JSON_0" \
    "$JSON_1" \
    "$RESULTS_3D_COORD_DIR" \
    "$COMMON_BASE_NAME" \
    "$INTRINSIC_CALIB_0" \
    "$INTRINSIC_CALIB_1" \
    "$STEREO_CALIB_RESULT"

echo ">> 3D coordinate file saved to '$RESULTS_3D_COORD_DIR'"

# --- 4. 3D Skeleton Animation ---      
echo ">> Step 4: Generating 3D Skeleton Animation..."

# [수정] 3D 좌표 JSON 파일 경로 정의
JSON_3D_PATH="$RESULTS_3D_COORD_DIR/${COMMON_BASE_NAME}_3d_results.json"

# 애니메이션 결과 파일 경로
ANIMATION_OUTPUT_PATH="$RESULTS_3D_ANIM_DIR/${COMMON_BASE_NAME}_3d_floating_animation.mp4"

# 원본 비디오의 FPS를 가져와서 애니메이션 FPS로 사용 (ffprobe 사용)
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$VIDEO_0" | awk -F'/' '{print $1/$2}')

echo "   - Input 3D JSON: $JSON_3D_PATH"
echo "   - Output Animation: $ANIMATION_OUTPUT_PATH"
echo "   - FPS: $FPS"

python3 3d_floating_skeletons.py \
    --input "$JSON_3D_PATH" \
    --output "$ANIMATION_OUTPUT_PATH" \
    --fps "$FPS"

echo ">> 3D animation saved to '$ANIMATION_OUTPUT_PATH'"

echo "===== Pipeline Finished ====="
