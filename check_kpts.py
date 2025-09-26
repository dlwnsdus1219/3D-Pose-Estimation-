import json

# 우리가 찾고자 하는 키포인트 ID와 이름 목록
expected_keypoints = {
    "5": "left_shoulder",
    "6": "right_shoulder",
    "11": "left_hip",
    "12": "right_hip",
    "13": "left_knee",
    "14": "right_knee",
    "15": "left_ankle",
    "16": "right_ankle",
    "17": "left_big_toe",
    "18": "left_small_toe",
    "19": "left_heel",
    "20": "right_big_toe",
    "21": "right_small_toe",
    "22": "right_heel",
}

# 키포인트 ID 목록을 set으로 변환하여 비교 효율을 높입니다
expected_ids = set(expected_keypoints.keys())

# JSON 파일 경로
file_path = './pipeline_results/3d_coordinates/trimmed_sub_output_3d_results.json'

missing_keypoints_found = False

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 모든 프레임을 순회
    for frame in data:
        frame_id = frame.get("frame_id")
        keypoints_3d = frame.get("keypoints_3d", {})

        # 현재 프레임에 있는 키포인트 ID를 set으로 변환
        present_ids = set(keypoints_3d.keys())

        # 전체 키포인트 ID에서 현재 프레임에 없는 ID를 찾기
        missing_ids = expected_ids - present_ids

        # 누락된 키포인트가 있을 경우 출력
        if missing_ids:
            missing_keypoints_found = True
            missing_names = [expected_keypoints[kp_id] for kp_id in sorted(list(missing_ids), key=int)]
            print(f"프레임 {frame_id} 에서 누락된 키포인트: {', '.join(missing_names)} (ID: {', '.join(sorted(list(missing_ids), key=int))})")

    if not missing_keypoints_found:
        print("모든 프레임에서 모든 키포인트가 정상적으로 감지되었습니다. 👍")

except FileNotFoundError:
    print(f"에러: 파일 경로를 찾을 수 없습니다. '{file_path}' 경로를 확인해주세요.")
except json.JSONDecodeError:
    print(f"에러: '{file_path}' 파일의 JSON 형식이 올바르지 않습니다.")