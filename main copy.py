from utils import read_video, save_video#, save_video2
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from draw_line import Drawline
from minimap import Minimap
from ultralytics import YOLO


def main():

    video_frames = read_video('/Users/chan/tennis/football_keypoint_pkl/input_videos/white_yellow_input.mp4')
    # #class를 불러옵니다. Minimap이 아니고 minimap으로 한건 별 의미 없어요.
    # minimap = Minimap("models/best_key_point.pt")
    tracker = Tracker('/Users/chan/tennis/football_keypoint_pkl/models/yolov8x_player.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                         stub_path= '/Users/chan/tennis/football_keypoint_pkl/stubs/_track_stubs_deep.pkl')
    # keypoints = minimap.get_object_keypoints(video_frames,
    #                                     read_from_stub=True,
    #                                     stub_path= 'stubs/_keypoints_stubs.pkl')
    #tracks 딕셔너리에 position밸류값을 더 해주는 함수에요.
    tracker.add_position_to_tracks(tracks)
    
    # #h에 변환행렬 리스트 mat을 저장합니다.
    # h = minimap.get_h(video_frames)

    # #이 함수가 tracks딕셔너리에 after_t밸류값을 추가하는 함수에요.
    # minimap.add_transfromed_position(tracks, h)
    #tracks딕셔너리의 오브젝트 중 players만, 그중에서도 21프레임 까지만 출력하겠다는 의미입니다.
    # #print(tracks['players'][:21])
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    # dt = minimap.draw_minimap(video_frames,tracks)
    # print(len(dt))
    # save_video(dt, 'output_videos/5.avi')
    #tracks 딕셔너리는 tracks - object(players,refrees,ball) - frame_num순서대로 구성되어 있고, frame_num안에 객체들의 정보가 담겨있어요. bbox,position,after_trans같은 것들이요.
    #frame_num은 따로 숫자로 표기되어있지 않습니다. print(tracks['players'][:21]) 이 출력값에도 보이지 않을거에요
    
    #이제 해야할일은 출력값을 보시면 알겠지만 after_trans값을 이용해서 minimap위에 점을 프레임별로 찍어내고, 그 프레임을 return값으로 받아낸 후 save_video를 통해서
    #영상으로 받아내면 됩니다.
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, '/Users/chan/tennis/football_keypoint_pkl/output_videos/original3.mp4')
    #키포인트 피클이 없어서 키포인트를 감지하는 속도가 느려요. 혹시 작업하시면서 키포인트 피클을 만들수 있으시면 그것부터 하시면 작업시간을 단축할 수 있을 겁니다.
    #아! 참고로 이 키포인트 가중치는 저희 마지막 학습이아닙니다! 그래서 11초짜리 영상에만 적용해 주세요

if __name__ == '__main__':
    main()