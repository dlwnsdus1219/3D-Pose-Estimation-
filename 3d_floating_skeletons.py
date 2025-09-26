import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm

# --- 1. 스켈레톤 정보 정의 ---
# 3D Triangulation 결과 JSON의 키포인트 ID(문자열)를 기준으로 정의
SKELETON_CONNECTIONS = [
    ('5', '6'), ('11', '12'), ('5', '11'), ('6', '12'),  # 몸통
    ('11', '13'), ('13', '15'),  # 왼다리
    ('12', '14'), ('14', '16'),  # 오른다리
    ('15', '19'), ('15', '17'), ('15', '18'), ('17', '18'),  # 왼발
    ('16', '22'), ('16', '20'), ('16', '21'), ('20', '21'),  # 오른발
]

def main(args):
    # --- 2. 3D JSON 파일 로드 및 전처리 ---
    print(f"데이터 로드 중: {args.input}")
    try:
        with open(args.input, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{args.input}'을(를) 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{args.input}' 파일이 비어 있거나 유효한 JSON 형식이 아닙니다.")
        return

    if not all_data:
        print("오류: JSON 파일에 데이터가 없습니다.")
        return

    # 프레임 ID를 키로, 3D 좌표를 값으로 하는 딕셔너리 생성
    animation_frames = {item['frame_id']: item['keypoints_3d'] for item in all_data}
    
    # 애니메이션을 위해 프레임 ID를 순서대로 정렬한다
    sorted_frame_ids = sorted(animation_frames.keys())

    # --- 3. 애니메이션 3D 플롯 초기 설정 ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 축 범위와 레이블, 시점 고정
    ax.set_xlabel('X 좌표 (mm)')
    ax.set_ylabel('Y 좌표 (mm)')
    ax.set_zlabel('Z 좌표 (높이, mm)')
    ax.set_xlim([1500, -1500])
    ax.set_ylim([-1500, 1500])
    ax.set_zlim([0, 2000])
    ax.view_init(elev=30, azim=-45)
    ax.set_title("3D Skeleton Animation")

    # 첫 프레임 데이터로 초기 스켈레톤을 그림
    first_frame_id = sorted_frame_ids[0]
    initial_kpts = animation_frames[first_frame_id]
    
    # 좌표계 변환: (X, Y, Z) -> (X, Z, -Y)
    # 월드 좌표계(Y가 아래)를 시각화 좌표계(Z가 위)로 변경
    points = np.array([coords for coords in initial_kpts.values() if coords])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # 점(scatter)과 선(plot) 객체를 한 번만 생성
    kpt_scatters = ax.scatter(x, y, -z, c='red', marker='o')
    lines = [ax.plot([], [], [], 'gray')[0] for _ in SKELETON_CONNECTIONS]


    # --- 4. 애니메이션 업데이트 함수 정의 ---
    def update(frame_id):
        keypoints_3d = animation_frames[frame_id]
        
        # 모든 키포인트 좌표를 한 번에 추출하고 변환
        valid_indices = [idx for idx in keypoints_3d if keypoints_3d[idx]]
        if not valid_indices:
            return kpt_scatters, *lines

        points = np.array([keypoints_3d[idx] for idx in valid_indices])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # 점 데이터 업데이트 (set_offsets는 2D용, 3D는 _offsets3d 사용)
        kpt_scatters._offsets3d = (x, y, -z)

        # 선 데이터 업데이트
        for i, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
            if start_idx in keypoints_3d and end_idx in keypoints_3d and keypoints_3d[start_idx] and keypoints_3d[end_idx]:
                start_pt = np.array(keypoints_3d[start_idx])
                end_pt = np.array(keypoints_3d[end_idx])
                
                # 점과 동일한 좌표계 변환을 선에도 적용
                # X축 데이터 -> 원본 x
                line_x = [start_pt[0], end_pt[0]]
                # Y축 데이터 -> 원본 z (깊이)
                line_y = [start_pt[1], end_pt[1]]
                # Z축 데이터 -> 원본 -y (높이)
                line_z = [-start_pt[2], -end_pt[2]]
                
                lines[i].set_data(line_x, line_y)
                lines[i].set_3d_properties(line_z)
            else:
                # 해당 프레임에 뼈대가 없으면 보이지 않게 처리
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
        
        progress_bar.update(1)
        return kpt_scatters, *lines

    # --- 5. 애니메이션 생성 및 저장 ---
    num_frames = len(sorted_frame_ids)
    print(f"애니메이션 생성 중... (총 {num_frames} 프레임)")
    
    progress_bar = tqdm(total=num_frames)
    
    ## 애니메이션 생성
    ani = FuncAnimation(
        fig, 
        update, 
        frames=sorted_frame_ids, 
        blit=False, # 3D에서는 blit=False 사용
        interval=1000 / args.fps  # 10fps 기준 (1000ms / 10)
    )

    # MP4 파일로 저장
    try:
        writer = FFMpegWriter(fps=args.fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(args.output, writer=writer)
        print(f"\n성공: 애니메이션이 '{args.output}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: MP4 저장에 실패했습니다. 'ffmpeg'이 시스템에 설치되어 있는지 확인하세요.")
        print(f"에러 메시지: {e}")
    
    progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D 스켈레톤 좌표를 읽어 애니메이션 비디오를 생성합니다.')
    parser.add_argument('--input', type=str, required=True, help='입력 3D 좌표 JSON 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 MP4 비디오 파일 경로')
    # --fps 인자 추가 (기본값을 25로 설정)
    parser.add_argument('--fps', type=int, default=25, help='출력 비디오의 초당 프레임 수(FPS)')
    
    parsed_args = parser.parse_args()
    main(parsed_args)

