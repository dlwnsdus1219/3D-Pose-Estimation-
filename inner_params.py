import cv2
import numpy as np
import glob
import os
import sys
import argparse

# --- 설정 ---
# 체커보드의 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (8, 6)
# 체커보드 사각형의 실제 크기 (mm 단위). 이 값을 정확하게 측정해야 합니다.
SQUARE_SIZE = 21
# # 캘리브레이션할 이미지들이 있는 디렉토리 경로
# ## 'calb1' 등으로 변경하여 다른 카메라를 캘리브레이션할 수 있습니다.
# image_dir = 'calb1' 
# # ---

def calib_inner_params(image_dir):
    """
    지정된 디렉토리의 이미지를 사용하여 카메라 캘리브레이션 수행
    """
    print(f"--- '{image_dir}' 디렉토리에 대한 캘리브레이션 시작 ---")

    # 3D 공간의 체커보드 코너 좌표 생성 (z=0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not images:
        print(f"오류: '{image_dir}'에서 이미지를 찾을 수 없습니다.")
        sys.exit()

    print(f"총 {len(images)}개의 이미지로 캘리브레이션을 시도합니다.")

    # 코너 검출 시각화를 위한 디렉토리
    debug_dir = f"debug_{os.path.basename(image_dir)}"
    os.makedirs(debug_dir, exist_ok=True)

    gray = None # 이미지 크기를 가져오기 위해 마지막 gray 이미지를 저장할 변수

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            print(f"{os.path.basename(fname)}: 코너 검출 성공")
            objpoints.append(objp)

            # 코너 좌표를 더 정확하게 찾기
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)

            # (디버깅용) 코너가 그려진 이미지 저장
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_sub, ret)
            debug_img_path = os.path.join(debug_dir, f"corners_{os.path.basename(fname)}")
            cv2.imwrite(debug_img_path, img)
        else:
            print(f"{os.path.basename(fname)}: 코너 검출 실패")

    if len(objpoints) < 5:
        print(f"\n오류: 캘리브레이션을 수행하기에 유효한 이미지가 너무 적습니다. (현재 {len(objpoints)}개, 최소 5개 이상 필요)")
        sys.exit()

    print(f"\n총 {len(objpoints)}개의 유효한 이미지로 캘리브레이션을 수행합니다...")

    # 카메라 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("캘리브레이션 실패.")
        return

    print("\n--- 캘리브레이션 결과 ---")
    print(f"재투영 오차 (Reprojection Error): {ret}\n")
    print("내부 파라미터 (Camera Matrix):\n", mtx)
    print("\n왜곡 계수 (Distortion Coefficients):\n", dist.ravel())

    # 결과 저장
    output_filename = f'calibration_result_{os.path.basename(image_dir)}.npz'
    np.savez(output_filename, mtx=mtx, dist=dist, error=ret)
    print(f"\n캘리브레이션 결과를 '{output_filename}' 파일로 저장했습니다.")

    # 재투영 오차 상세 계산
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"전체 평균 재투영 오차: {mean_error/len(objpoints)}")

def main():
    """
    지정된 모든 디렉토리에 대해 카메라 캘리브레이션을 실행합니다.
    """
    # 캘리브레이션을 수행할 디렉토리 목록
    calibration_dirs = ['calb0', 'calb1']

    for calib_dir in calibration_dirs:
        if not os.path.isdir(calib_dir):
            print(f"경고: '{calib_dir}' 디렉토리를 찾을 수 없습니다. 건너뜁니다.")
            continue
        calib_inner_params(calib_dir)

    print(f"\n{'='*50}")
    print("모든 캘리브레이션 작업이 완료되었습니다.")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform camera intrinsic calibration for one or more directories.')
    parser.add_argument('dirs', nargs='+', help='List of directories containing calibration images (e.g., calb0 calb1).')
    args = parser.parse_args()
    main(args)

# if __name__ == '__main__':
#     main()