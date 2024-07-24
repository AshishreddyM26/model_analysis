import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

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

    def calculate_vertical_fov(self, y1, y2):
        vertical_fov = []
        for y1_val, y2_val in zip(y1, y2):
            fov = self.calculate_fov_for_line(self.vertical_fov, self.height, y1_val, y2_val, is_horizontal=False)
            vertical_fov.append(fov)
        return vertical_fov

    def detect_results(self, set, model, video_source):
        weight = f"C:/Users/ashis/Desktop/THESIS/DT_flow/resources/weights/set{set}/{model}.pt"
        model = YOLO(weight)
        results = model.predict(source=video_source, project='files/roi_detection/custom_botsort_detections', conf=.4, verbose=False,
                                save=True, save_frames=True, line_width=2)
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
            for i in range(len(y1)):
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
        """
        Divide the data into rows based on specified conditions for set A.

        Parameters:
            wc_gt (DataFrame): DataFrame containing the data to be divided.

        Returns:
            list: List of DataFrames, each representing a row in set A.
        """
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
        """
        Divide the data into rows based on specified conditions for set B.

        Parameters:
            wc_gt (DataFrame): DataFrame containing the data to be divided.

        Returns:
            list: List of DataFrames, each representing a row in set B.
        """
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
            row1, row2, row3, row4, row5, row6, row7, row8 = rows_setA(df)
        else:
            row1, row2, row3, row4, row5, row6, row7, row8 = rows_setB(df)
            
        gt_df = [row1, row2, row3, row4, row5, row6, row7, row8]
        gt_counts_results = self.process_gt_counts(gt_df, y1, y2)
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

    def count_crops_rows_78(self, y1_bounds, y2_bounds, xmin_list, xmax_list, limit, model):
        if limit > 20:
            count_in_row = []
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin_list, xmax_list, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3400) & (df.iloc[:, 3] <= 2160) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3500) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin_list, xmax_list, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
            return count_in_row
        else:
            count_in_row = []
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin_list, xmax_list, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(29):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3420) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3493) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin_list, xmax_list, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
            return count_in_row
