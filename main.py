import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mot_utilities import det_utilities
from mot_utilities import window_analysis
from sort import Sort

# -- customizing trackers - BoT-SORT, ByteTrack, SORT, DeepSORT

# -- customized ByteTrack

class ByteTrack_CropCounter:
    def __init__(self):
        # Initialize constants
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians
        self.y1 = [1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695, 660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 
                   310, 275, 240, 205, 170, 135, 100, 65, 30, 0]
        self.y2 = [1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430, 1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 
                   1745, 1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060, 2095, 2130, 2160]

    def calculate_angles(self, xc, yc):
        xcs = np.array(xc)
        ycs = np.array(yc)
        xd = xcs - self.width / 2
        yd = self.height / 2 - ycs
        theta_h = xd * self.happ
        horiz_angle = np.degrees(theta_h)
        return horiz_angle.tolist()
  
    def calculate_fov_for_line(self, fov, dimension, start_pixel, end_pixel, is_horizontal=True):
        degrees_per_pixel = fov / dimension
        line_length_pixels = abs(end_pixel - start_pixel)
        line_fov = line_length_pixels * degrees_per_pixel
        return line_fov

    def calculate_vertical_fov(self):
        vertical_fov = []
        for y1_val, y2_val in zip(self.y1, self.y2):
            fov = self.calculate_fov_for_line(self.vertical_fov, self.height, y1_val, y2_val, is_horizontal=False)
            vertical_fov.append(fov)
        return vertical_fov

    def detect_results(self, set, model, video_source):
        weight = f"C:/Users/ashis/Desktop/THESIS/DT_flow/resources/weights/set{set}/{model}.pt"
        model = YOLO(weight)
        results = model.predict(source=video_source, conf=.4, verbose=False, save=False)
        return results

    def ret_gt_cropcount(self, df, y1, y2):
        trim_df = df[(df['CenterY'] > y1) & (df['CenterY'] < y2)]
        crop_count = trim_df['id'].nunique()
        return crop_count

    def get_accy(self, gt_count, pred_count):
        error_rate = abs(pred_count - gt_count) / gt_count
        if error_rate > 1:
            return 0
        else:
            accuracy = (1 - error_rate) * 100
            return round(accuracy, 2)

    def process_gt_counts(self, row_data, y1_data, y2_data):
        results = {}
        for row_idx in range(1, len(row_data) + 1):
            row_key = f'gt_row{row_idx}'
            gt_counts = []
            for i in range(len(y1_data)):
                gt_count = self.ret_gt_cropcount(row_data[row_idx - 1], y1_data[i], y2_data[i])
                gt_counts.append(gt_count)
            results[row_key] = gt_counts
        return results

    def calculate_accuracies(self, gt_data, count_data):
        results = {}
        for row_idx in range(1, len(gt_data) + 1):
            accuracies = []
            gt_row = gt_data[f'gt_row{row_idx}']
            count_row = count_data[f'row{row_idx}']
            for i in range(len(self.y1)):
                accuracy = self.get_accy(gt_row[i], count_row[i])
                accuracies.append(accuracy)
            results[f'accuracy{row_idx}'] = accuracies
        return results

    def rows_setA(self, wc_gt):
        row1 = wc_gt[(wc_gt['CenterX'] >= 350) & (wc_gt['CenterX'] <= 725)]
        row2 = wc_gt[(wc_gt['CenterX'] >= 705) & (wc_gt['CenterX'] <= 1100)]
        row3 = wc_gt[(wc_gt['CenterX'] >= 1100) & (wc_gt['CenterX'] <= 1600)]
        row4 = wc_gt[(wc_gt['CenterX'] >= 1600) & (wc_gt['CenterX'] <= 2150)]
        row5 = wc_gt[(wc_gt['CenterX'] >= 2150) & (wc_gt['CenterX'] <= 2600)]
        row6 = wc_gt[(wc_gt['CenterX'] >= 2700) & (wc_gt['CenterX'] <= 3050)]
        row7 = wc_gt[
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3390) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3500) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]
        row8 = wc_gt[
            ((wc_gt['CenterX'] >= 3350) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3490) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]

        return row1, row2, row3, row4, row5, row6, row7, row8

    def rows_setB(self, wc_gt):
        row1 = wc_gt[(wc_gt['CenterX'] >= 350) & (wc_gt['CenterX'] <= 725)]
        row2 = wc_gt[(wc_gt['CenterX'] >= 725) & (wc_gt['CenterX'] <= 1100)]
        row3 = wc_gt[(wc_gt['CenterX'] >= 1100) & (wc_gt['CenterX'] <= 1600)]
        row4 = wc_gt[(wc_gt['CenterX'] >= 1600) & (wc_gt['CenterX'] <= 2150)]
        row5 = wc_gt[(wc_gt['CenterX'] >= 2150) & (wc_gt['CenterX'] <= 2600)]
        row6 = wc_gt[(wc_gt['CenterX'] >= 2700) & (wc_gt['CenterX'] <= 3050)]
        row7 = wc_gt[
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3390) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3500) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]
        row8 = wc_gt[
            ((wc_gt['CenterX'] >= 3370) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3490) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]

        return row1, row2, row3, row4, row5, row6, row7, row8

    def gen_gt_results(self, model, type):
        gt_counts_results = {}
        df = pd.read_csv(f'C:/Users/ashis/Desktop/THESIS/DT_flow/resources/dataframes{type}/{model}_gt.csv')
        
        if type == 'A':
            rows = self.rows_setA(df)
        else:
            rows = self.rows_setB(df)
            
        gt_counts_results = self.process_gt_counts(rows, self.y1, self.y2)
        return gt_counts_results

    def get_avg_accuracy(self, accuracy_results1, accuracy_results2):
        average_results = {}
        for key in accuracy_results1:
            list_a = accuracy_results1[key]
            list_b = accuracy_results2[key]
            average_list = [(a + b) / 2 for a, b in zip(list_a, list_b)]
            average_results[key] = average_list
        return average_results

    def ret_frame_dets(self, tensor, i):
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df_appended = pd.concat([df1, df2], axis=1)
        final_df = df_appended[df_appended.iloc[:, 1] >= 660]
        return final_df

    def ret_trimmed_df(self, df_appended, xmin, xmax, y1, y2):
        trim_df = df_appended[((((df_appended.iloc[:, 0] + df_appended.iloc[:, 2]) / 2 >= xmin) &
                               ((df_appended.iloc[:, 0] + df_appended.iloc[:, 2]) / 2 <= xmax)) &
                              (((df_appended.iloc[:, 1] + df_appended.iloc[:, 3]) / 2 >= y1) &
                               ((df_appended.iloc[:, 1] + df_appended.iloc[:, 3]) / 2 <= y2)))]
        xyxy = trim_df.iloc[:, :-1].values
        confidence = trim_df.iloc[:, -1].values
        return (xyxy, confidence)

    def get_count(self, window_listup):
        tracker = sv.ByteTrack(minimum_matching_threshold=.8, track_activation_threshold=.5, lost_track_buffer=24)
        id = []
        count = []
        track_ids = []
        for i in range(29):
            results = window_listup[i]
            boxes, confidences = results
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=np.zeros(len(confidences), dtype=int)
            )
            detections = tracker.update_with_detections(detections)
            ids = detections.tracker_id.tolist()
            track_ids.append(ids)
        for i in range(len(track_ids)):
            for j in range(len(track_ids[i])):
                id.append(track_ids[i][j])
        count = len(set(id))
        return count

    def count_crops_rows_1to6(self, y1_bounds, y2_bounds, xmin_list, xmax_list, model):
        results = {}
        for i in range(len(xmin_list)):
            count_in_row = []
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin_list[i], xmax_list[i], ymin, ymax)
                    results_window.append(frame_detections)
                crops_count = self.get_count(results_window)
                count_in_row.append(crops_count)
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
        return results

    def count_crops_rows_78(self, y1_bounds, y2_bounds, xmin, xmax, limit, model):
        count_in_row = []
        if limit > 20:
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3400) & (df.iloc[:, 3] <= 2160) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3500) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
        else:
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3420) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3493) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
        return count_in_row

    def count_crops_for_model(self, model_name, results, xmin_limits, xmax_limits, y1, y2):
        row16 = self.count_crops_rows_1to6(y1, y2, xmin_limits[:6], xmax_limits[:6], results)
        row7 = self.count_crops_rows_78(y1, y2, xmin_limits[6], xmax_limits[6], 25 if model_name == "1" else 21, results)
        row8 = self.count_crops_rows_78(y1, y2, xmin_limits[7], xmax_limits[7], 15, results)
        row16['row7'] = row7
        row16['row8'] = row8
        return row16

    def plot_accuracy9(accuracy_results_list, model_names, vertical_fov):
        
        threshold = 95  # Set the threshold for accuracy
        fig, axs = plt.subplots(1, 8, figsize=(20, 5))  # Create 1 row and 8 columns of subplots

        # Define specific colors for each model
        colors = ['orange', 'blue', 'cyan', 'magenta', 'green']

        if len(model_names) > len(colors):
            raise ValueError("Not enough unique colors defined for the number of models.")

        for i in range(8):
            for accuracy_results, model_name, color in zip(accuracy_results_list, model_names, colors):
                accuracy_key = f'accuracy{i+1}'
                data_x = vertical_fov
                data_y = accuracy_results[accuracy_key]

                # Plot line segments individually
                for j in range(len(data_x) - 1):
                    segment_color = 'red' if data_y[j] >= threshold else color
                    axs[i].plot([data_x[j], data_x[j+1]], [data_y[j], data_y[j+1]], marker='o', markersize=.7, color=segment_color, linewidth=0.3, label=model_name if j == 0 else "")

                # Draw a horizontal line at the threshold value
                axs[i].axhline(y=threshold, color='red', linestyle='--', linewidth=.5)

            axs[i].set_title(f'Row{i+1} ({horz_va1[i]}°, {horz_va2[i]}°)', fontsize=10)  # Set the font size here
            axs[i].set_ylim(0, 105)
            axs[i].grid()

        axs[3].set_xlabel('Vertical FOV (degrees°)')      
        axs[0].set_ylabel('Accuracy')  

        # Create a legend for the models and place it above the subplots
        handles = [plt.Line2D([0], [0], color=color, lw=1) for color in colors[:len(model_names)]]
        labels = model_names

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(model_names))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust the top to make room for the legend
        plt.show()

# -- customized sort tracker
 
class SORT_CropCounter:
    
    def __init__(self):
        # Initialize constants
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians
        self.y1 = [1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695, 660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 
                   310, 275, 240, 205, 170, 135, 100, 65, 30, 0]
        self.y2 = [1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430, 1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 
                   1745, 1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060, 2095, 2130, 2160]

    def return_detections(i, tensor, xmin, xmax, y1, y2):
        
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df_appended = pd.concat([df1, df2], axis=1)
        df = df_appended[df_appended.iloc[:, 1] >= 660] 
        
        trim_df = df[
            ((df.iloc[:, 0] + df.iloc[:, 2]) / 2 >= xmin) & 
            ((df.iloc[:, 0] + df.iloc[:, 2]) / 2 <= xmax) & 
            ((df.iloc[:, 1] + df.iloc[:, 3]) / 2 >= y1) & 
            ((df.iloc[:, 1] + df.iloc[:, 3]) / 2 <= y2)
        ]  # -- 0-x1, 1-y1, 2-x2, 3-y2
        xyxyc = trim_df.values
        return xyxyc

    def return_detections_row7(i, tensor, xmin, xmax, y1, y2):
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df_appended = pd.concat([df1, df2], axis=1)
        df = df_appended[df_appended.iloc[:, 1] >= 660]
        trim_df = df[((df.iloc[:,2] >= 3050) & (df.iloc[:,2] <= 3300) & (df.iloc[:,3] <= 2160) & (df.iloc[:,3] > 1500)) | ((df.iloc[:,2] >= 3050) & (df.iloc[:,2] <= 3500) & (df.iloc[:,3] <= 1500) & (df.iloc[:,3] >= 660))]
        xyxyc = trim_df.values
        return xyxyc

    def process_sort(det_results, xmin, xmax, y1_lim, y2_lim):

        tracker = Sort(max_age=25, min_hits=5, iou_threshold=.01)  # -- min_hits=5 found through tuning
        ids = []
        for i in range(29):   # -- len(det_results) - paste this inplace of 29 for bigger videos
            
            detections = np.empty((0, 5))
            xyxyc = return_detections(i, det_results, xmin, xmax, y1_lim, y2_lim)
            detections = np.vstack((detections, xyxyc))     # -- updating the detections to the tracker

            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:            
                _, _, _, _, obj_id = map(int, obj)
                ids.append(obj_id)
                
        crop_count = len(set(ids))
        return crop_count

    def process_sort_row7(det_results, xmin, xmax, y1_lim, y2_lim):

        tracker = Sort(max_age=25, min_hits=5, iou_threshold=.01)  # -- min_hits=5 found through tuning

        ids = []
        
        for i in range(29):   # -- len(det_results) - paste this inplace of 29 for bigger videos
            
            detections = np.empty((0, 5))
            xyxyc = return_detections_row7(i, det_results, xmin, xmax, y1_lim, y2_lim)
            detections = np.vstack((detections, xyxyc))     # -- updating the detections to the tracker
            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:            
                _, _, _, _, obj_id = map(int, obj)
                ids.append(obj_id)
                
        crop_count = len(set(ids))
        return crop_count
    
    def count_crops_rows_1to6(y1_bounds, y2_bounds, xmin_list, xmax_list, model):

        results = {}
        for i in range(len(xmin_list)):
            
            count_in_row = []   
            
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                crops_count = process_sort(model, xmin_list[i], xmax_list[i], ymin, ymax)
                count_in_row.append(crops_count)
            
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
            
        return results
    
    def count_crops_rows_78(y1_bounds, y2_bounds, xmin_list, xmax_list, limit, model):

        # -- here we will write a, if - else cond, where 20 is the threshold for the limit 
        # -- for row 7 - limit > 20 as the limit is 25
        # -- for row 8 - limit < 20 as the limit is 15
        if limit > 20:
            
            # -- logic for row 7
            
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
                
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = process_sort_row7(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count2)
                
            return count_in_row
            
        else:
        
        # -- logic for row 8
        
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
                
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count2)
                
            return count_in_row
        