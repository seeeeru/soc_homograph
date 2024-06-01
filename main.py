from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from minimap import Minimap
from draw_line import Drawline
from player_ball_assigner import PlayerBallAssigner

import os


def main():

    video_frames = read_video('input_videos/white_yellow_input.mp4')

    minimap = Minimap("models/best_key_point.pt")
    tracker = Tracker('models/yolov8x_player.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path= 'stubs/_track_stubs_deep.pkl')
    keypoints = minimap.get_object_keypoints(video_frames,
                                        read_from_stub=True,
                                        stub_path= 'stubs/_keypoints_stubs.pkl')

    tracker.add_position_to_tracks(tracks)
    h = minimap.get_h(video_frames,keypoints)
    minimap.add_transfromed_position(tracks, h)
    
    draw_line = Drawline()

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



    # 공 점유율 구하기
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in enumerate(tracks['players']):
         ball_bbox= tracks["ball"][frame_num][1]["bbox"]
         assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

         if assigned_player != -1:
              tracks['players'][frame_num][assigned_player]['has_ball'] = True
              team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

         else:
              team_ball_control.append(team_ball_control[-1])
    team_ball_control=np.array(team_ball_control)

    
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    
    target_ids = [6,15]
    specific_postions = draw_line.get_positions_for_specific_track_ids(tracks, target_ids)
    transformed_positions = draw_line.transform_positions(specific_postions,tracks)
    
    draw_line.draw_lines_between_tracks(output_video_frames, specific_postions,transformed_positions, target_ids)

    minimap_frames = minimap.draw_minimap(video_frames,tracks)
    conbine_f = minimap.combine_frames(output_video_frames,minimap_frames)

    save_video(output_video_frames, 'output_videos/original2.mp4')
    save_video(minimap_frames, 'output_videos/minimap2.mp4')
    save_video(conbine_f,'output_videos/combine2.mp4')
    os.system(f'ffmpeg -i output_videos/combine2.mp4 -vcodec libx264 output_videos/combine2_change.mp4')
if __name__ == '__main__':
    main()