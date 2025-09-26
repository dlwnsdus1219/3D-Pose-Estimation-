"""2대의 카메라에서 촬영된 2D 키포인트 데이터를 활용하여, 3D 공간의 좌표를 계산하고 이를 json 파일로 저장한다."""
import numpy as np
import cv2
import json
import os
import sys

def load_params(calib_file_cam0, calib_file_cam1, calib_file_stereo):
    """캘리브레이션 파일들로부터 모든 파라미터를 로드한다(내장, 외장 파라미터)"""
    try:
        ## 0번 카메라 내장 파라미터
        data_cam0 = np.load(calib_file_cam0)
        cam0_matrix = data_cam0['mtx']
        dist0_coeff = data_cam0['dist']

        ## 1번 카메라 내장 파라미터
        data_cam1 = np.load(calib_file_cam1)
        cam1_matrix = data_cam1['mtx']
        dist1_coeff = data_cam1['dist']

        ## 외장 파라미터 (두 카메라 간의 상대적 위치 및 방향)
        data_stereo = np.load(calib_file_stereo)
        R0 = data_stereo['R0']
        T0 = data_stereo['T0']
        R1 = data_stereo['R1']
        T1 = data_stereo['T1']

        print(">> 모든 파라미터를 성공적으로 로드했습니다.")
        return cam0_matrix, dist0_coeff, R0, T0, cam1_matrix, dist1_coeff, R1, T1
    except Exception as e:
        print(f"!!! 에러: 캘리브레이션 파일 로드 중 오류 발생: {e}")
        return [None] * 8
    
def interpolate_missing_keypoints(data, max_gap=5):
    """
    누락된 3D 키포인트를 선형 보간합니다.
    testing_interpolation.py의 로직을 참고하여 구현합니다.
    """
    if not data:
        return []

    print(f"\n--- 후처리: 키포인트 보간 시작 (최대 간격: {max_gap} 프레임) ---")
    
    # 보간을 위해 데이터를 프레임 ID 기반 딕셔너리로 변환
    frames_dict = {item['frame_id']: item['keypoints_3d'] for item in data}
    sorted_frame_ids = sorted(frames_dict.keys())
    
    # 처리할 키포인트 ID 목록 (기존 required_indices와 동일)
    kpt_ids_to_process = {str(i) for i in [5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]}

    # 각 키포인트의 시계열 데이터 생성
    kpt_timeseries = {kpt_id: {} for kpt_id in kpt_ids_to_process}
    for frame_id in sorted_frame_ids:
        for kpt_id in kpt_ids_to_process:
            coords = frames_dict[frame_id].get(kpt_id)
            if coords and len(coords) == 3:
                kpt_timeseries[kpt_id][frame_id] = np.array(coords)

    interpolated_count = 0
    # 각 키포인트 별로 순회하며 보간 수행
    for kpt_id, series in kpt_timeseries.items():
        present_frames = sorted(series.keys())
        for i in range(len(present_frames) - 1):
            p1_frame, p2_frame = present_frames[i], present_frames[i+1]
            gap = p2_frame - p1_frame
            
            if 1 < gap <= max_gap:
                p1_coords = series[p1_frame]
                p2_coords = series[p2_frame]
                
                for j in range(1, gap):
                    target_frame = p1_frame + j
                    t_ratio = (target_frame - p1_frame) / (p2_frame - p1_frame)
                    interpolated_coords = p1_coords + t_ratio * (p2_coords - p1_coords)
                    
                    if target_frame in frames_dict:
                        frames_dict[target_frame][kpt_id] = interpolated_coords.tolist()
                        interpolated_count += 1

    print(f"보간 완료. 총 {interpolated_count}개의 키포인트 좌표가 새로 채워졌습니다.")
    
    # [수정] 최종 반환 시, 각 프레임의 keypoints_3d 딕셔너리를 키(ID) 기준으로 정렬
    final_sorted_data = []
    for frame_id in sorted_frame_ids:
        sorted_keypoints = {
            k: frames_dict[frame_id][k] 
            for k in sorted(frames_dict[frame_id].keys(), key=int)
        }
        final_sorted_data.append({
            "frame_id": frame_id,
            "keypoints_3d": sorted_keypoints
        })
        
    return final_sorted_data

def main(json_path0, json_path1, output_dir, output_basename, 
         calib_file_cam0, calib_file_cam1, calib_file_stereo):
    """3D triangulation 메인 함수"""
    ## --- 1. 카메라 파라미터 정의 ---
    cam0_matrix, dist0_coeff, R0, T0, cam1_matrix, dist1_coeff, R1, T1 = load_params(
        calib_file_cam0, calib_file_cam1, calib_file_stereo
    )
    if cam0_matrix is None:
        sys.exit("파라미터 로드 실패로 프로그램을 종료합니다!!")

    # ## 카메라 0 (우측 상단) -> 구형 파라미터
    # cam0_matrix = np.array([
    #     [2.21807122e+03, 0.00000000e+00, 1.88683451e+03],
    #     [0.00000000e+00, 2.23187083e+03, 1.08737398e+03],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    # ], dtype=np.float32)
    # dist0_coeff = np.array(
    #     [0.05490184, -0.0996034, -0.00178527, -0.00259743,  0.10502617],
    #     dtype=np.float32
    # )

    # # 기준 카메라이므로 월드 좌표계 원점에 위치한다고 가정!!
    # R0 = np.eye(3, dtype=np.float32)
    # T0 = np.zeros((3, 1), dtype=np.float32)
    # [수정] 기준을 '카메라0'에서 '체커보드(월드)'로 변경
    # stereo_calibration.py의 실행 결과로 얻은 R0, T0 값을 붙여넣습니다.
    # R0 = np.array([[-0.96590967, 0.21972189, 0.13689705],
    #                [0.0438194, -0.38241048, 0.92295292],
    #                [0.25514383, 0.89748789, 0.3597459]], 
    #                dtype=np.float32)
    # T0 = np.array([[721.08203627], [821.01984215], [1828.54456148]], 
    #               dtype=np.float32)

    # ## 카메라 1 (좌측 상단) -> 구형 파라미터
    # cam1_matrix = np.array([
    #     [2.24446709e+03, 0.00000000e+00, 1.91777953e+03],
    #     [0.00000000e+00, 2.24601269e+03, 1.09467240e+03],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    # ], dtype=np.float32)
    # dist1_coeff = np.array(
    #     [0.09616735, -0.22221305, -0.00235942, 0.00131059, 0.20666361], 
    #     dtype=np.float32
    # )

    # # # 카메라 0에 대한 상대적 외부 파라미터
    # R1 = np.array([[-0.97214963, -0.22941589, -0.04789],
    #                [0.04963734, -0.40126227, 0.91461725],
    #                [-0.22904418, 0.88676769, 0.40147456]], dtype=np.float32)
    # T1 = np.array([[624.03966169], [768.66098198], [2189.85642747]], dtype=np.float32)

    # --- 2. 투영 행렬(Projection Matrix) 계산 ---
    # P = K * [R|t]
    proj_matrix0 = cam0_matrix @ np.hstack((R0, T0))
    proj_matrix1 = cam1_matrix @ np.hstack((R1, T1))

    print("--- Projection Matrix for Cam 0 ---")
    print(proj_matrix0)
    print("\n--- Projection Matrix for Cam 1 ---")
    print(proj_matrix1)
    print("\n카메라 파라미터 및 투영 행렬이 성공적으로 준비되었습니다.")


    # --- 3. 2D 키포인트 데이터 로드하기(json 파일) ---
    with open(json_path0, 'r') as f:
        data0 = json.load(f)
    with open(json_path1, 'r') as f:
        data1 = json.load(f)
    print(f">> JSON 파일 로드 완료: {json_path0}, {json_path1}")

    # 프레임 ID를 키로 하는 딕셔너리로 데이터를 효율적으로 재구성
    frames_data0 = {item['frame_id']: item for item in data0}
    frames_data1 = {item['frame_id']: item for item in data1}

    # --- 4. 3D 좌표 계산 및 저장 ---
    results_3d = []
    total_error0, total_error1 = [], []
    
    # 두 카메라에서 공통으로 검출된 프레임 ID 찾기 (한 쪽 카메라에서만 검출된 키포인트는 3D triangulation 자체가 불가능하므로 스킵)
    common_frame_ids = sorted(list(set(frames_data0.keys()) & set(frames_data1.keys())))
    print(f"총 {len(common_frame_ids)}개의 공통 프레임에 대해 3D 좌표를 계산합니다.")

    for frame_id in common_frame_ids:
        frame0 = frames_data0[frame_id]
        frame1 = frames_data1[frame_id]

        # 'persons' 리스트가 비어있거나 없는 경우 건너뛰기
        if not frame0.get('persons') or not frame1.get('persons'):
            print(f"경고: 프레임 {frame_id}에 'persons' 데이터가 없어 건너뜁니다.")
            continue
        
        # 첫 번째 사람의 키포인트 데이터를 사용
        kpts0_list = frame0['persons'][0].get('keypoints', [])
        kpts1_list = frame1['persons'][0].get('keypoints', [])

        if not kpts0_list or not kpts1_list:
            print(f"경고: 프레임 {frame_id}에 키포인트 데이터가 없어 건너뜁니다.")
            continue

        # 키포인트 idx를 키로 하는 딕셔너리 생성
        kpts0_dict = {kp['idx']: (kp['x'], kp['y']) for kp in kpts0_list if 'x' in kp and 'y' in kp}
        kpts1_dict = {kp['idx']: (kp['x'], kp['y']) for kp in kpts1_list if 'x' in kp and 'y' in kp}

        # 모든 키포인트가 존재하는지 확인할 기준 목록 정의
        # 사용자가 요청한 키포인트 목록
        required_indices = {5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

        # 두 카메라 모두에서 모든 필수 키포인트가 검출되었는지 확인
        cam0_keys = set(kpts0_dict.keys())
        cam1_keys = set(kpts1_dict.keys())
        
        # # 공통 키포인트 다 있는 경우만 ㄱㄱ
        # if not (required_indices.issubset(cam0_keys) and required_indices.issubset(cam1_keys)):
        #     print(f"경고: 프레임 {frame_id}에서 모든 필수 키포인트가 검출되지 않아 건너뜁니다.")
        #     continue

        # 두 카메라에서 공통으로 검출된 키포인트 인덱스 찾기
        common_indices = sorted(list(set(kpts0_dict.keys()) & set(kpts1_dict.keys())))
        
        if len(common_indices) < 5:
            print(f"경고: 프레임 {frame_id}의 공통 키포인트가 부족하여 건너뜁니다 ({len(common_indices)}개 발견).")
            continue

        # 공통 키포인트의 2D 좌표를 순서에 맞게 추출
        points2d_0 = np.array([kpts0_dict[idx] for idx in common_indices], dtype=np.float32)
        points2d_1 = np.array([kpts1_dict[idx] for idx in common_indices], dtype=np.float32)

        # 렌즈 왜곡 보정
        undistorted_pt0 = cv2.undistortPoints(points2d_0, cam0_matrix, dist0_coeff, P=cam0_matrix)
        undistorted_pt1 = cv2.undistortPoints(points2d_1, cam1_matrix, dist1_coeff, P=cam1_matrix)

        # 3D 좌표 계산
        points_4d_hom = cv2.triangulatePoints(proj_matrix0, proj_matrix1, undistorted_pt0, undistorted_pt1)
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
        
        # 결과 저장
        keypoints_3d_dict = {str(idx): pt.tolist() for idx, pt in zip(common_indices, points_3d)}
        if keypoints_3d_dict:
            results_3d.append({"frame_id": frame_id, "keypoints_3d": keypoints_3d_dict})
        
        # (선택) 재투영 오차 계산
        reprojected_points0, _ = cv2.projectPoints(points_3d, R0, T0, cam0_matrix, dist0_coeff)
        error0 = np.linalg.norm(points2d_0 - reprojected_points0.reshape(-1, 2), axis=1)
        total_error0.extend(error0)

        reprojected_points1, _ = cv2.projectPoints(points_3d, R1, T1, cam1_matrix, dist1_coeff)
        error1 = np.linalg.norm(points2d_1 - reprojected_points1.reshape(-1, 2), axis=1)
        total_error1.extend(error1)

    interpolated_res = interpolate_missing_keypoints(results_3d, max_gap=5)

    # --- 5. 결과 파일 저장 ---
    output_path = os.path.join(output_dir, f"{output_basename}_3d_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        # json.dump(results_3d, f, indent=4)
        json.dump(interpolated_res, f, indent=4)
        
    print(f"\n성공: 3D 좌표가 '{output_path}' 파일로 저장되었습니다.")
    if total_error0 and total_error1:
        print(f"전체 재투영 오차(Cam 0): 평균={np.mean(total_error0):.2f}px, 최대={np.max(total_error0):.2f}px")
        print(f"전체 재투영 오차(Cam 1): 평균={np.mean(total_error1):.2f}px, 최대={np.max(total_error1):.2f}px")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("사용법: python 3d_triangulation.py [cam0_json] [cam1_json] [output_dir] [output_basename] [calib_file_cam0] [calib_file_cam1] [calib_file_stereo]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
