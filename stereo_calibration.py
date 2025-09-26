import numpy as np
import cv2
import glob
import os
import argparse

def load_calibration_params(filepath):
    """ .npz 파일에서 카메라 매트릭스와 왜곡 계수를 로드합니다. """
    if not os.path.exists(filepath):
        print(f"!!! 에러: 캘리브레이션 파일 '{filepath}'을(를) 찾을 수 없습니다.")
        return None, None
    try:
        data = np.load(filepath)
        return data['mtx'], data['dist']
    except Exception as e:
        print(f"!!! 에러: '{filepath}' 파일 로드 중 오류 발생: {e}")
        return None, None
    
def main(args):
    # --- 1. 캘리브레이션 설정 (인자로부터 받기) ---
    CHECKERBOARD = tuple(args.checkerboard)
    SQUARE_SIZE_MM = args.square_size
    IMAGE_PATH_0 = args.image0
    IMAGE_PATH_1 = args.image1

    print("--- 입력된 설정 ---")
    print(f"  - 체커보드 코너: {CHECKERBOARD}")
    print(f"  - 사각형 크기: {SQUARE_SIZE_MM}mm")
    print(f"  - 카메라 0 이미지: {IMAGE_PATH_0}")
    print(f"  - 카메라 1 이미지: {IMAGE_PATH_1}")
    print(f"  - 카메라 0 캘리브레이션 파일: {args.calib0}")
    print(f"  - 카메라 1 캘리브레이션 파일: {args.calib1}")
    print("--------------------")

    # 내부 파라미터 로드
    cam0_matrix, dist_coeffs0 = load_calibration_params(args.calib0)
    cam1_matrix, dist_coeffs1 = load_calibration_params(args.calib1)

    if cam0_matrix is None or cam1_matrix is None:
        print("\n!!! 내부 파라미터 로드 실패. 스크립트를 종료합니다.")
        return

    print("\n>> 내부 파라미터 로드 성공.")
    print("  - Cam 0 Matrix:\n", cam0_matrix)
    print("  - Cam 1 Matrix:\n", cam1_matrix)

    # --- 2. 체커보드 코너 검출 ---
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM
    objpoints = [objp]

    print("\n--- 체커보드 격자점의 3D 월드 좌표 (단위: mm) ---")
    for i, point in enumerate(objp):
        print(f"  코너 {i}: (x={point[0]:.1f}, y={point[1]:.1f}, z={point[2]:.1f})")
    print("--------------------------------------------------\n")

    imgpoints0 = []
    imgpoints1 = []

    if not (os.path.exists(IMAGE_PATH_0) and os.path.exists(IMAGE_PATH_1)):
        print(f"!!! 에러: 이미지 파일 '{IMAGE_PATH_0}' 또는 '{IMAGE_PATH_1}'을 찾을 수 없습니다.")
        return
    
    img0 = cv2.imread(IMAGE_PATH_0)
    img1 = cv2.imread(IMAGE_PATH_1)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    ret0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)

    if ret0 and ret1:
        imgpoints0.append(corners0)
        imgpoints1.append(corners1)
        print(">> 성공: 두 이미지에서 모두 코너를 찾았습니다.")
        
        print(">> 검출된 코너를 시각화하여 'cam0_corners.jpg'와 'cam1_corners.jpg'로 저장합니다...")
        img0_vis = img0.copy()
        cv2.drawChessboardCorners(img0_vis, CHECKERBOARD, corners0, ret0)
        for i, corner in enumerate(corners0):
            x, y = map(int, corner[0])
            cv2.putText(img0_vis, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite("cam0_corners.jpg", img0_vis)

        img1_vis = img1.copy()
        cv2.drawChessboardCorners(img1_vis, CHECKERBOARD, corners1, ret1)
        for i, corner in enumerate(corners1):
            x, y = map(int, corner[0])
            cv2.putText(img1_vis, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite("cam1_corners.jpg", img1_vis)
    else:
        print("!!! 실패: 한 쪽 또는 양쪽 이미지에서 코너를 찾지 못했습니다.")
        print(f"  - 카메라 0 ({os.path.basename(IMAGE_PATH_0)}) 코너 검출: {'성공' if ret0 else '실패'}")
        print(f"  - 카메라 1 ({os.path.basename(IMAGE_PATH_1)}) 코너 검출: {'성공' if ret1 else '실패'}")
        print(f"  - 현재 설정된 코너 개수(가로, 세로): {CHECKERBOARD}")
        return

    # --- 3. 스테레오 캘리브레이션 실행 ---
    if len(imgpoints0) > 0 and len(imgpoints1) > 0:
        print("\n>> 스테레오 캘리브레이션을 실행합니다...")
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints0, imgpoints1,
            cam0_matrix, dist_coeffs0,
            cam1_matrix, dist_coeffs1,
            gray0.shape[::-1],
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        if ret:
            print("\n" + "="*50)
            print("     스테레오 캘리브레이션 성공!")
            print("="*50)
            print(f"\n[결과] 재투영 에러: {ret:.4f}")
            print("\n[결과] 회전 행렬 (R) - Cam0 -> Cam1:\n", R)
            print("\n[결과] 변환 벡터 (T) - Cam0 -> Cam1 (단위: mm):\n", T)
            print("\n" + "="*50)
            print("아래 내용을 3d_triangulation.py 등에 복사하세요. (카메라0에서 카메라1로의 변환)")
            print("="*50)
            print("R_cam1_from_cam0 = np.array(" + np.array2string(R, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
            print("T_cam1_from_cam0 = np.array(" + np.array2string(T, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
            print("="*50)

            # # 결과 파일 저장
            # output_filename = 'stereo_calibration_result.npz'
            # np.savez(output_filename, R=R, T=T, error=ret)
            # print(f"\n스테레오 캘리브레이션 결과를 '{output_filename}' 파일로 저장했습니다.")

            # --- 4. 각 카메라의 월드 좌표계 기준 Pose 계산 및 시각화 ---
            print("\n>> solvePnP를 사용하여 각 카메라의 월드 좌표계를 계산하고 시각화합니다...")
            axis_length = 3 * SQUARE_SIZE_MM
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)

            R0, tvec0, R1, tvec1 = None, None, None, None

            # --- 카메라 0 ---
            ret_pnp0, rvec0, tvec0 = cv2.solvePnP(objp, corners0, cam0_matrix, dist_coeffs0)
            if ret_pnp0:
                R0, _ = cv2.Rodrigues(rvec0)
                print("\n--- 카메라 0의 월드->카메라 변환 정보 ---")
                print("R0 = np.array(" + np.array2string(R0, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
                print("T0 = np.array(" + np.array2string(tvec0, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
                print("------------------------------------")

                img_points_2d_0, _ = cv2.projectPoints(axis_points, rvec0, tvec0, cam0_matrix, dist_coeffs0)
                origin = tuple(img_points_2d_0[0].ravel().astype(int))
                img0_axes = img0.copy()
                img0_axes = cv2.line(img0_axes, origin, tuple(img_points_2d_0[1].ravel().astype(int)), (0, 0, 255), 10) # X: Red
                img0_axes = cv2.line(img0_axes, origin, tuple(img_points_2d_0[2].ravel().astype(int)), (0, 255, 0), 10) # Y: Green
                img0_axes = cv2.line(img0_axes, origin, tuple(img_points_2d_0[3].ravel().astype(int)), (255, 0, 0), 10) # Z: Blue
                cv2.imwrite("cam0_with_axes.jpg", img0_axes)
                print("  - 성공: 'cam0_with_axes.jpg' 파일로 월드 좌표계가 그려진 이미지를 저장했습니다.")
            else:
                print("  - 실패: 카메라 0에 대한 solvePnP 계산에 실패했습니다.")

            # --- 카메라 1 ---
            ret_pnp1, rvec1, tvec1 = cv2.solvePnP(objp, corners1, cam1_matrix, dist_coeffs1)
            if ret_pnp1:
                R1, _ = cv2.Rodrigues(rvec1)
                print("\n--- 카메라 1의 월드->카메라 변환 정보 ---")
                print("R1 = np.array(" + np.array2string(R1, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
                print("T1 = np.array(" + np.array2string(tvec1, separator=', ').replace('\n', '').replace(' ', '') + ", dtype=np.float32)")
                print("------------------------------------")

                img_points_2d_1, _ = cv2.projectPoints(axis_points, rvec1, tvec1, cam1_matrix, dist_coeffs1)
                origin = tuple(img_points_2d_1[0].ravel().astype(int))
                img1_axes = img1.copy()
                img1_axes = cv2.line(img1_axes, origin, tuple(img_points_2d_1[1].ravel().astype(int)), (0, 0, 255), 10) # X: Red
                img1_axes = cv2.line(img1_axes, origin, tuple(img_points_2d_1[2].ravel().astype(int)), (0, 255, 0), 10) # Y: Green
                img1_axes = cv2.line(img1_axes, origin, tuple(img_points_2d_1[3].ravel().astype(int)), (255, 0, 0), 10) # Z: Blue
                cv2.imwrite("cam1_with_axes.jpg", img1_axes)
                print("  - 성공: 'cam1_with_axes.jpg' 파일로 월드 좌표계가 그려진 이미지를 저장했습니다.")
            else:
                print("  - 실패: 카메라 1에 대한 solvePnP 계산에 실패했습니다.")

            # 결과 파일 저장하기 (R0, T0, R1, T1 추가하기)
            if R0 is not None and R1 is not None:
                output_filename = 'stereo_calibration_result.npz'
                np.savez(output_filename, R_rel=R, T_rel=T, R0=R0, T0=tvec0, R1=R1, T1=tvec1, error=ret)
                print(f"\n스테레오 캘리브레이션 결과를 '{output_filename}' 파일로 저장했습니다.")
            else:
                print("\n!!! solvePnP 실패로 인해 일부 외부 파라미터가 저장되지 않았습니다.")

        else:
            print("\n!!! 에러: 스테레오 캘리브레이션에 실패했습니다.")
    else:
        print("\n>> 스테레오 캘리브레이션을 시작할 수 없습니다. (코너 검출 실패)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration using pre-calculated intrinsic parameters.')
    parser.add_argument('--image0', type=str, required=True, help='Path to the calibration image for camera 0.')
    parser.add_argument('--image1', type=str, required=True, help='Path to the calibration image for camera 1.')
    parser.add_argument('--calib0', type=str, required=True, help='Path to the .npz calibration file for camera 0 (e.g., calibration_result_calb0.npz).')
    parser.add_argument('--calib1', type=str, required=True, help='Path to the .npz calibration file for camera 1 (e.g., calibration_result_calb1.npz).')
    parser.add_argument('--checkerboard', type=int, nargs=2, default=[5, 4], help='Checkerboard inner corners (width height). Default: 8 6')
    parser.add_argument('--square_size', type=float, default=300, help='Size of a checkerboard square in mm. Default: 21.0')
    
    args = parser.parse_args()
    main(args)