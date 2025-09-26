# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from tqdm import tqdm

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def iou(boxA, boxB):
    """IoU 계산을 위한 함수"""
    # 두 박스의 좌표 추출
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 교차 영역의 넓이 계산
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # 각 박스의 넓이 계산
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU 계산
    iou_value = interArea / float(boxAArea + boxBArea - interArea)

    return iou_value


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            # draw_keypoint_labels=True,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    # parser.add_argument(
    #     '--json-file',
    #     type=str,
    #     default='',
    #     help='Path to store the prediction results in JSON format.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    # Simplified the path creation to a single line to avoid any syntax issues.
    if args.save_predictions:
        assert args.output_root != ''
        basename = os.path.basename(args.input)
        filename_without_ext = os.path.splitext(basename)[0]
        args.pred_save_path = os.path.join(args.output_root, f'results_{filename_without_ext}.json')

    # if args.save_predictions:
    #     assert args.output_root != ''
    #     args.pred_save_path = (
    #         f'{args.output_root}/results_'
    #         f'{os.path.splitext(os.path.basename(args.input))[0]}.json')
    #     # args.pred_save_path = f'{args.output_root}/results_' \
    #     #     f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator

    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    
    # # <<<--- 시각화 필터링 코드 추가 --- START --->>>

    # # 1. 필터링할 키포인트 인덱스 정의 (JSON 저장 로직과 동일하게)
    # target_indices = [5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    
    # # 2. 원본 메타 정보 가져오기
    # original_meta = pose_estimator.dataset_meta
    
    # # 3. 필터링된 새 메타 정보 생성
    # new_meta = original_meta.copy()
    
    # # 키포인트 이름/ID 맵 필터링
    # sorted_target_indices = sorted(target_indices)
    # new_meta['keypoint_id2name'] = {
    #     i: original_meta['keypoint_id2name'][idx]
    #     for i, idx in enumerate(sorted_target_indices)
    # }
    # new_meta['keypoint_name2id'] = {
    #     name: new_id
    #     for new_id, name in new_meta['keypoint_id2name'].items()
    # }
    
    # # 스켈레톤(뼈대) 연결 정보 필터링
    # original_links = original_meta.get('skeleton_links', [])
    # target_names = set(new_meta['keypoint_name2id'].keys())
    
    # new_links = []
    # for p1_idx, p2_idx in original_links:
    #     p1_name = original_meta['keypoint_id2name'][p1_idx]
    #     p2_name = original_meta['keypoint_id2name'][p2_idx]
        
    #     # 두 점이 모두 우리가 원하는 키포인트에 속할 경우에만 링크 추가
    #     if p1_name in target_names and p2_name in target_names:
    #         new_links.append(
    #             (new_meta['keypoint_name2id'][p1_name], new_meta['keypoint_name2id'][p2_name])
    #         )
    
    # new_meta['skeleton_links'] = new_links
    # new_meta['num_keypoints'] = len(sorted_target_indices)

    # # 4. 필터링된 메타 정보를 visualizer에 설정
    # visualizer.set_dataset_meta(
    #     new_meta, skeleton_style=args.skeleton_style)
    
    # ## 뼈대와 키포인트 개수 변경 이슈로, 색상 팔레트 재설정(visualizer가 new_meta에 맞춰 색상을 새로 만듭니다)
    # visualizer.kpt_color = None
    # visualizer.link_color = None
    
    # # 5. 키포인트에 텍스트 라벨을 표시하도록
    # visualizer.draw_keypoint_labels = True

    # # # <<<--- 시각화 필터링 코드 추가 --- END --->>>

    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input.lower().endswith(video_exts):
        input_type = 'video'
    else:
        mt = mimetypes.guess_type(args.input)[0]
        input_type = mt.split('/')[0] if mt is not None else 'unknown'
    # if args.input == 'webcam':
    #     input_type = 'webcam'
    # else:
    #     input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        # 비디오의 총 프레임 수 갖고오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        frame_idx = 0
        all_frames_results = []  # 모든 프레임 결과를 누적할 리스트

        # 필터링할 키포인트 인덱스 정의
        target_indices = [5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        # 프로그레스 바 생성
        progress_bar = None
        if input_type == 'video':
            progress_bar = tqdm(total=total_frames, desc=f"Processing Video '{os.path.basename(args.input)}'")

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break
            
            frame_idx += 1

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)

            # if args.save_predictions:
            #     # save prediction results
            #     pred_instances_list.append(
            #         dict(
            #             frame_id=frame_idx,
            #             instances=split_instances(pred_instances)))

            if args.save_predictions and pred_instances is not None:
                # 프레임 별로 결과 생성 로직
                frame_persons_data = []

                # 각 사람에 대해 반복
                for person_id, instance in enumerate(split_instances(pred_instances)):
                    person_data = {
                        'person': person_id,
                        'keypoints': []
                    }

                    keypoints = np.array(instance['keypoints'])
                    scores = np.array(instance['keypoint_scores'])

                    # 원하는 키포인트만 필터링
                    for original_idx in target_indices:
                        # 원본 인덱스에 해당하는 좌표와 점수 가져오기
                        x, y = keypoints[original_idx]
                        score = scores[original_idx]
                        
                        if score >= args.kpt_thr:
                            person_data["keypoints"].append({
                                "idx": original_idx, # 원본 인덱스 번호 저장
                                "x": float(x),
                                "y": float(y),
                                "score": float(score)
                            })

                    if person_data['keypoints']:
                        frame_persons_data.append(person_data)


                # 생성 결과를 전체 리스트에 추가 ㄱㄱ
                if frame_persons_data:
                    all_frames_results.append({
                        'frame_id': frame_idx, 
                        'persons': frame_persons_data
                    })
                    
            # 프로그레스 바 업데이트
            if progress_bar:
                progress_bar.update(1)

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)
        
        # 프로그레스 바 닫기
        if progress_bar:
            progress_bar.close()

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:

        # ## --- 필터링 코드 추가 --- ##
        # target_indices = {5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

        # print(f"Filtering keypoints to keep only indices: {sorted(list(target_indices))}")

        # # 각 프레임의 각 인스턴스(사람)에 대해 키포인트 필터링
        # for frame_info in pred_instances_list:
        #     for instance in frame_info.get('instances', []):
        #         # keypoints와 keypoint_scores를 NumPy 배열로 변환
        #         keypoints = np.array(instance['keypoints'])
        #         scores = np.array(instance['keypoint_scores'])

        #         # target_indices에 해당하는 키포인트와 점수만 선택
        #         filtered_keypoints = keypoints[target_indices]
        #         filtered_scores = scores[target_indices]

        #         # 필터링된 결과로 다시 할당
        #         instance['keypoints'] = filtered_keypoints.tolist()
        #         instance['keypoint_scores'] = filtered_scores.tolist()
        
        # # meta_info도 필터링된 정보에 맞게 업데이트 (선택 사항이지만 권장)
        # original_meta = pose_estimator.dataset_meta
        # filtered_meta = original_meta.copy() # 얕은 복사
        
        # # 키포인트 이름과 ID 맵 필터링 (더 안전한 방식으로 수정)
        # original_id2name = original_meta.get('keypoint_id2name', {})
        # # target_indices가 정렬되어 있다는 가정 하에 새 ID 부여
        # id2name = {new_idx: original_id2name.get(old_idx, f'unknown_{old_idx}') 
        #            for new_idx, old_idx in enumerate(sorted(target_indices))}
        # name2id = {name: new_id for new_id, name in id2name.items()}

        # filtered_meta['num_keypoints'] = len(target_indices)
        # filtered_meta['keypoint_id2name'] = id2name
        # filtered_meta['keypoint_name2id'] = name2id

        with open(args.pred_save_path, 'w') as f:
            json.dump(all_frames_results, f, indent=2)
        
        # with open(args.pred_save_path, 'w') as f:
        #     json.dump(
        #         dict(
        #             meta_info=filtered_meta,
        #             instance_info=pred_instances_list),
        #         f,
        #         indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()