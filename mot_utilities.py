# -- Install Dependencies


from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
# from ultralytics import YOLOv10
from ultralytics import YOLO
import motmetrics as mm
from scipy import stats
from sort import Sort
import seaborn as sns
import pandas as pd
import numpy as np
import math
import cv2
import os


# -- Utility class function


class det_utilities:
    
    def __init__(self):
        pass

    def process_tensors(self, tensor):
        """
        Process the tensor results from the YOLO model.

        Parameters:
            tensor: Tensor results from the YOLO model.

        Returns:
            list: Processed list of frame annotations.
        """
        processed_list = []
        for i in range(29):  
            frame_list = []
            frame = i + 1  # Frame number
            
            for j in range(len(tensor[i].boxes.xywh)):
                # frame_id = int(tensor[i].boxes.id[j])  # ID of each bbox in frame 'i'

                x_center, y_center, w, h = map(float, tensor[i].boxes.xywh[j][:])  # x, y, w, h
                conf = float(tensor[i].boxes.conf[j])  # Confidence
                x_min, y_min = x_center - (w / 2), y_center - (h / 2)
                if y_center >= 660:
                    frame_list.append((frame, 0, np.round(x_min, 4), np.round(y_min, 4),                   # -- include frame_id - id of the bbox if necessary, usually helps in tracking
                                       np.round(w, 4), np.round(h, 4), np.round(conf, 3), -1, -1, -1))
            
            processed_list.append(frame_list)
        return processed_list
    
    def annotations_to_text(self, tensor_list, output_path):
        """
        Convert tensor annotations to text and save to the output path.

        Parameters:
            tensor_list (list): List of processed tensor annotations.
            output_path (str): Path to save the text annotations.

        Returns:
            None
        """
        
        with open(output_path, 'w') as file:
            for frame_list in tensor_list:
                for annotation in frame_list:
                    file.write(','.join(map(str, annotation)) + '\n')


    def detect_and_process(self, weight_file, data_source, output_dir, project, file_name):
        """
        Detect objects in the data source using the YOLO model and process the detection results.

        Parameters:
            weight_file (str): Path to the YOLO weight file.
            data_source (str): Path to the data source (e.g., video file, image folder).
            output_path (str): Path to save the processed annotations.
            project (str): Project name or identifier.

        Returns:
            None
        """
        model = YOLO(weight_file)
        det_results = model.predict(
            source=data_source, iou=0.5, line_width=2, save_frames=True, save=True,
            show_labels=True, show_conf=True, conf=0.25, save_txt=True, show=False, project=project
        )

        tensor_list = self.process_tensors(det_results)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, file_name)
        self.annotations_to_text(tensor_list, output_file)
    
    # def detect_and_process10(self, weight_file, data_source, output_dir, project, file_name):
    #     """
    #     Detect objects in the data source using the YOLO model and process the detection results.

    #     Parameters:
    #         weight_file (str): Path to the YOLO weight file.
    #         data_source (str): Path to the data source (e.g., video file, image folder).
    #         output_path (str): Path to save the processed annotations.
    #         project (str): Project name or identifier.

    #     Returns:
    #         None
    #     """
    #     model = YOLOv10(weight_file)
    #     det_results = model.predict(
    #         source=data_source, iou=0.5, line_width=2, save_frames=True, save=True,
    #         show_labels=True, show_conf=True, conf=0.25, save_txt=True, show=False, project=project
    #     )

    #     tensor_list = self.process_tensors(det_results)
        
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
        
    #     output_file = os.path.join(output_dir, file_name)
    #     self.annotations_to_text(tensor_list, output_file)
    
    def process_bounding_boxes(self, ground_truth_file, prediction_file):
        """
        Load and process bounding boxes from ground truth and prediction files.

        Parameters:
            ground_truth_file (str): Path to the ground truth file.
            prediction_file (str): Path to the prediction file.

        Returns:
            tuple: DataFrames for ground truth and predictions with additional columns.
        """
        gt_labels = pd.read_csv(ground_truth_file, sep=',', header=None, 
                                names=['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])
        pred_labels = pd.read_csv(prediction_file, sep=',', header=None, 
                                  names=['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])

        for df in [gt_labels, pred_labels]:
            df['x2'] = df['x1'] + df['width']
            df['y2'] = df['y1'] + df['height']
            df['CenterX'] = (df['x1'] + df['x2']) / 2
            df['CenterY'] = (df['y1'] + df['y2']) / 2

        return gt_labels, pred_labels

    def process_and_match_boxes(self, gt_df, pred_df):
        """
        Process and match bounding boxes using Intersection over Union (IoU).

        Parameters:
            gt_df (DataFrame): Ground truth DataFrame.
            pred_df (DataFrame): Prediction DataFrame.

        Returns:
            DataFrame: Ground truth DataFrame with matched IoU scores.
        """
        def calculate_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        gt_boxes = gt_df[['x1', 'y1', 'x2', 'y2']].values
        pred_boxes = pred_df[['x1', 'y1', 'x2', 'y2']].values

        original_iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                original_iou_matrix[i, j] = calculate_iou(gt_boxes[i], pred_boxes[j])

        max_boxes = max(len(gt_boxes), len(pred_boxes))
        extended_iou_matrix = np.full((max_boxes, max_boxes), -1e-5)
        extended_iou_matrix[:len(gt_boxes), :len(pred_boxes)] = original_iou_matrix

        gt_indices, pred_indices = linear_sum_assignment(-extended_iou_matrix)

        valid_indices = (gt_indices < len(gt_boxes)) & (pred_indices < len(pred_boxes))
        valid_gt_indices = gt_indices[valid_indices]
        valid_pred_indices = pred_indices[valid_indices]

        matched_iou_scores = original_iou_matrix[valid_gt_indices, valid_pred_indices]

        matched_data = pd.DataFrame({
            'IoU': matched_iou_scores
        }, index=valid_gt_indices)

        gt_df = gt_df.merge(matched_data, left_index=True, right_index=True, how='left')
        return gt_df

    def fill_nan_values(self, df_list):
        """
        Fill NaN values in a list of DataFrames.

        Parameters:
            df_list (list): List of DataFrames to process.

        Returns:
            None
        """
        for i, df in enumerate(df_list):
            if df.isna().any().any() == False:
                print(f'No NaN {i}')
            else:
                df.fillna(0.0, inplace=True)
                print(f'NaN values are converted in {i}')

    def calculate_viewing_angles(self, df, x_center, y_center, happ, vapp):
        """
        Calculate viewing angles for bounding boxes.

        Parameters:
            df (DataFrame): DataFrame containing bounding box data.
            x_center (float): X-coordinate of the image center.
            y_center (float): Y-coordinate of the image center.
            happ (float): Horizontal angular pixel size.
            vapp (float): Vertical angular pixel size.

        Returns:
            pd.Series: Series with calculated viewing angles.
        """
        xd = df['CenterX'] - x_center
        yd = y_center - df['CenterY']

        theta_h = xd * happ
        theta_v = yd * vapp

        combined_oblique_radians = np.arctan(np.sqrt(np.tan(theta_h)**2 + np.tan(theta_v)**2))

        return pd.Series({
            'oblique_angle': np.degrees(combined_oblique_radians),
            'horiz_angle': np.degrees(theta_h),
            'vertical_angle': np.degrees(theta_v)
        })

    def apply_angles_to_dataframe(self, df, hfov_degrees, vfov_degrees, width, height):
        """
        Apply viewing angles to a DataFrame based on camera parameters.

        Parameters:
            df (DataFrame): DataFrame to modify.
            hfov_degrees (float): Horizontal field of view in degrees.
            vfov_degrees (float): Vertical field of view in degrees.
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            DataFrame: Modified DataFrame with viewing angles.
        """
        happ = math.radians(hfov_degrees) / width
        vapp = math.radians(vfov_degrees) / height
        x_center = width / 2
        y_center = height / 2

        df[['oblique_va', 'hor_va', 'vert_va']] = df.apply(
            self.calculate_viewing_angles, axis=1, args=(x_center, y_center, happ, vapp))
        return df

    def processing_dfs(self, ground_truth_files, prediction_files, output_paths):
        """
        Process ground truth and prediction files and save results.

        Parameters:
            ground_truth_files (list): List of paths to ground truth files.
            prediction_files (list): List of paths to prediction files.
            output_paths (list): List of output paths for saving results.

        Returns:
            None
        """
        for gt_file, pred_file, output_prefix in zip(ground_truth_files, prediction_files, output_paths):
            gt_df, pred_df = self.process_bounding_boxes(gt_file, pred_file)
            matched_gt_df = self.process_and_match_boxes(gt_df, pred_df)
            
            camera1_settings = {'hfov_degrees': 118, 'vfov_degrees': 69.2, 'width': 3840, 'height': 2160}
            camera2_settings = {'hfov_degrees': 80, 'vfov_degrees': 50, 'width': 1920, 'height': 1080}
            
            self.fill_nan_values([matched_gt_df, pred_df])
            matched_gt_df = self.apply_angles_to_dataframe(matched_gt_df, **camera1_settings) # -- for whitecity 
            
            matched_gt_df.to_csv(f'{output_prefix}_gt.csv', index=False)
            pred_df.to_csv(f'{output_prefix}_pred.csv', index=False)
            print('Done...')

    # -- Dividing the dataframe into rows depending upon its data collected (from setA or setB) 
    
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
    
    
    def plot_mean_iou_heatmap(self, dfs):
        
        """
        Plots a heatmap of the mean IoU values.

        Parameters:
        dfs (list of DataFrames): List of DataFrames containing 'Y' and 'IoU' columns.

        Returns:
        None
        """
        # Initialize the mean IoU matrix
        mean_iou_matrix = np.zeros((3, len(dfs)))
        height = 2160 / 3

        # Calculate mean IoU for each section and each DataFrame
        for col_index, df in enumerate(dfs):
            mean_iou_matrix[0, col_index] = df[df['CenterY'] < height]['IoU'].mean()  # top section
            mean_iou_matrix[1, col_index] = df[(df['CenterY'] > height) & (df['CenterY'] <= height*2)]['IoU'].mean()  # middle section
            mean_iou_matrix[2, col_index] = df[df['CenterY'] > height*2]['IoU'].mean()  # bottom section

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.heatmap(mean_iou_matrix, annot=True, fmt=".2f", cmap='jet',
                    xticklabels=[f'Row {i+1}' for i in range(len(dfs))],
                    yticklabels=['Top', 'Middle', 'Bottom'], vmin=0, vmax=1, cbar_kws={'label': 'IoU'},
                    ax=ax)
        ax.set_title('Mean IoU')
        
        plt.tight_layout()
        plt.show()
        
        
    def plot_iou_scores(self, df, title):
        """
        Plots bounding box centers colored by IoU scores.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing columns 'CenterX', 'CenterY', and 'IoU'.
        title (str): Title of the plot.
        """
        
        plt.figure(figsize=(16, 6))
        plt.scatter(df['CenterX'], df['CenterY'], alpha=0.7, c=df['IoU'], cmap='seismic', vmin=0, vmax=1, s=10)
        plt.colorbar(label='IoU Scores')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim((0, 3840))
        plt.ylim((2160, 0))
        plt.title(title)
        plt.show()


# -- add the significance test codes here -- then we are done for the step 1

    def significance_test(self, dfs):
        """
        Divide dataframes into segments and calculate correlation between the number of bounding boxes and mean IoU for each segment.

        Parameters:
            dfs (list): List of DataFrames to be segmented.

        Returns:
            tuple: DataFrame with mean IoU for each segment, correlation coefficient, and p-value.
        """
        segment_means_list = []
        
        for df in dfs:
            segments = {'Top': df[(df['CenterY'] >= 0) & (df['CenterY'] < 720)],
                        'Middle': df[(df['CenterY'] >= 720) & (df['CenterY'] < 1440)],
                        'Bottom': df[(df['CenterY'] >= 1440) & (df['CenterY'] < 2160)]}

            segment_means = {'Segment': [], 'Number_of_BBoxes': [], 'Mean_IoU': []}
            
            for segment, segment_df in segments.items():
                if not segment_df.empty:
                    segment_means['Segment'].append(segment)
                    segment_means['Number_of_BBoxes'].append(len(segment_df))
                    segment_means['Mean_IoU'].append(segment_df['IoU'].mean())

            segment_means_list.append(pd.DataFrame(segment_means))
        
        combined_results = pd.concat(segment_means_list)
        
        # Perform correlation analysis
        correlation, p_value = stats.pearsonr(combined_results['Number_of_BBoxes'], combined_results['Mean_IoU'])
        return combined_results, correlation, p_value
    
    
# -- Tracking Utilites

    def process_tracker_tensors(self, tensor):
        """
        Process the tensor results from the YOLO model.

        Parameters:
            tensor: Tensor results from the YOLO model.

        Returns:
            list: Processed list of frame annotations.
        """
        processed_list = []
        for i in range(29):  
            frame_list = []
            frame = i + 1  # Frame number
            
            for j in range(len(tensor[i].boxes.xywh)):
                frame_id = int(tensor[i].boxes.id[j])  # ID of each bbox in frame 'i'

                x_center, y_center, w, h = map(float, tensor[i].boxes.xywh[j][:])  # x, y, w, h
                conf = float(tensor[i].boxes.conf[j])  # Confidence
                x_min, y_min = x_center - (w / 2), y_center - (h / 2)
                if y_center >= 660:
                    frame_list.append((frame, frame_id, np.round(x_min, 4), np.round(y_min, 4),
                                       np.round(w, 4), np.round(h, 4), np.round(conf, 3), -1, -1, -1))
            
            processed_list.append(frame_list)
        return processed_list

    # -- obtaining the 'pred' text file for further processing

    def track_and_process(self, weight_file, data_source, output_dir, project, file_name, tracker):
        """
        Detect objects in the data source using the YOLO model and process the detection results.

        Parameters:
            weight_file (str): Path to the YOLO weight file.
            data_source (str): Path to the data source (e.g., video file, image folder).
            output_dir (str): Output directory to save the processed annotations, frames (folder), and video.
            project (str): Project name or identifier.
            filename (str): resultant name for prediction's file 
            tracker (str): defined the tracker to be used (BytrTrack, BoT-SORT)    
            
        Returns:
            None
        """
        model = YOLO(weight_file)
        det_results = model.track(
            source=data_source, iou=0.5, line_width=2, save_frames=True, save=True,
            show_labels=True, show_conf=True, conf=0.25, save_txt=True, show=False, project=project, tracker = tracker
        )

        tensor_list = self.process_tracker_tensors(det_results)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, file_name)
        self.annotations_to_text(tensor_list, output_file)
        
    # def hota_metrics(gt: np.ndarray, t: np.ndarray) -> dict:
    #     """
    #     Calculate HOTA metrics using TrackEval.

    #     Parameters:
    #     gt (numpy.ndarray): Ground truth detections, expected shape [N, 6] with columns [frame, id, x, y, width, height]
    #     t (numpy.ndarray): Tracker detections, expected shape [N, 6] with columns [frame, id, x, y, width, height]

    #     Returns:
    #     dict: HOTA metrics.
    #     """

    #     # Prepare data in the format required by TrackEval
    #     gt_dict = {i: gt[gt[:, 0] == i, 1:].tolist() for i in np.unique(gt[:, 0])}
    #     t_dict = {i: t[t[:, 0] == i, 1:].tolist() for i in np.unique(t[:, 0])}

    #     gt_data = {'gt': gt_dict}
    #     t_data = {'t': t_dict}

    #     # Initialize TrackEval
    #     evaluator = trackeval.Evaluator()
    #     dataset_list = [trackeval.datasets.GroundTruthDetections(gt_data), trackeval.datasets.TrackerDetections(t_data)]
    #     metrics_list = [trackeval.metrics.HOTA()]

    #     # Evaluate
    #     results = evaluator.evaluate(dataset_list, metrics_list)

    #     # Extract HOTA metrics
    #     hota_metrics = results['HOTA']['HOTA']
    #     return hota_metrics
    

class MOTMetrics:
    
    @staticmethod
    def convert_rownumpy(gt_file, pred_file):
        """
        Convert ground truth and prediction text files to NumPy arrays.

        Parameters:
        gt_file (str): Path to the ground truth text file.
        pred_file (str): Path to the prediction text file.

        Returns:
        tuple: Two NumPy arrays, one for ground truth and one for predictions.
        """
        gt = np.loadtxt(gt_file, delimiter=',')
        pred = np.loadtxt(pred_file, delimiter=',')
        return gt, pred
    
    @staticmethod
    def motMetrics_Calculator(gt, t):
        """
        Calculate Multiple Object Tracking (MOT) metrics.

        Parameters:
        gt (numpy.ndarray): Ground truth detections, expected shape [N, 6] with columns [frame, id, x, y, width, height]
        t (numpy.ndarray): Tracker detections, expected shape [N, 6] with columns [frame, id, x, y, width, height]

        Returns:
        pandas.DataFrame: Summary of MOT metrics.
        """
        # Ensure inputs are numpy arrays
        gt = np.asarray(gt)
        t = np.asarray(t)

        # Create an accumulator that will be updated during each frame
        acc = mm.MOTAccumulator(auto_id=True)

        # Max frame number maybe different for gt and t files
        max_frame = int(max(gt[:, 0].max(), t[:, 0].max()))
        for frame in range(max_frame):
            frame += 1  # detection and frame numbers begin at 1

            # select id, x, y, width, height for current frame
            gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
            t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

            C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)  # format: gt, t

            # Call update once per frame.
            acc.update(
                gt_dets[:, 0].astype('int').tolist(),
                t_dets[:, 0].astype('int').tolist(),
                C
            )

        mh = mm.metrics.create()

        summary = mh.compute(
            acc, 
            metrics=[
                'num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 
                'num_objects', 'mostly_tracked', 'partially_tracked', 
                'mostly_lost', 'num_false_positives', 'num_misses', 
                'num_switches', 'num_fragmentations', 'mota', 'motp'
            ], 
            name='acc'
        )

        strsummary = mm.io.render_summary(
            summary,
            namemap={
                'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll',
                'precision': 'Prcn', 'num_objects': 'GT', 'mostly_tracked': 'MT',
                'partially_tracked': 'PT', 'mostly_lost': 'ML', 
                'num_false_positives': 'FP', 'num_misses': 'FN', 
                'num_switches': 'IDsw', 'num_fragmentations': 'FM', 
                'mota': 'MOTA', 'motp': 'MOTP'
            }
        )

        return summary
    
    # -- Counting Acuuracy

    def get_CountingAccuracy(self, df):
        gt_count = df['mostly_tracked'].acc + df['partially_tracked'].acc + df['mostly_lost'].acc
        pred_count = df['mostly_tracked'].acc + df['partially_tracked'].acc
        
        error_rate = abs(pred_count - gt_count) / gt_count
        if error_rate > 1:
            return 0  # Returns 0% accuracy if error rate exceeds 100%
        else:
            accuracy = (1 - error_rate) * 100
            return round(accuracy, 2)
        
    # def get_Accuracy(self, gt_df, pred_df):
    #     gt_count = gt_df['id'].nunique()
    #     pred_count = pred_df['id'].nunique()
    #     error_rate = abs(pred_count - gt_count) / gt_count
    #     if error_rate > 1:
    #         return 0  # Returns 0% accuracy if error rate exceeds 100%
    #     else:
    #         accuracy = (1 - error_rate) * 100
    #         return round(accuracy, 2)

class SortTracker:
    def __init__(self, weights, source, result):
        self.source = source
        self.result = result
        self.model = YOLO(weights)
        self.model.fuse()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    def process_sort(self):
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), "Error: Video source not opened."

        if not os.path.exists(self.result):
            os.makedirs(self.result)

        result_path = os.path.join(self.result, 'tracked_results.avi')
        text_file_path = os.path.join(self.result, 'sort.txt')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_width, vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        with open(text_file_path, 'w') as file:
            frame_number = 1

            while True:
                ret, img = cap.read()
                if not ret:
                    break

                detections = np.empty((0, 5))
                results = self.model(img, stream=True)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        cls = int(box.cls[0])
                        conf = math.ceil(box.conf[0] * 100) / 100

                        if conf > 0.4:
                            detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

                tracked_objects = self.tracker.update(detections)

                for obj in tracked_objects:
                    x1, y1, x2, y2, obj_id = obj
                    x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
                    w, h = x2 - x1, y2 - y1
                    conf = -1

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Save to text file
                    file.write(f'{frame_number},{obj_id},{x1},{y1},{w},{h},{conf:.2f},-1,-1,-1\n')

                out.write(img)
                cv2.imshow('Tracked Video', img)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_number += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


class DeepSortTracker:
    def __init__(self, weights, source, result):
        self.source = source
        self.result = result
        self.model = YOLO(weights)
        self.model.fuse()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.tracker = DeepSort(max_age=20, n_init=3, nms_max_overlap=1.0)

    def process_deepsort(self):
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), "Error: Video source not opened."

        if not os.path.exists(self.result):
            os.makedirs(self.result)

        result_path = os.path.join(self.result, 'tracked_results.avi')
        text_file_path = os.path.join(self.result, 'tracking_results.txt')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_width, vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        with open(text_file_path, 'w') as file:
            frame_number = 1

            while True:
                ret, img = cap.read()
                if not ret:
                    break

                detections = []
                results = self.model(img, stream=True)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        cls = int(box.cls[0])
                        conf = math.ceil(box.conf[0] * 100) / 100

                        if conf > 0.4:
                            detections.append(([x1, y1, x2 - x1, y2 - y1], conf))  # Correcting the format

                tracked_objects = self.tracker.update_tracks(detections, frame=img)

                for track in tracked_objects:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()
                    obj_id = track.track_id
                    x1, y1, x2, y2 = [int(i) for i in bbox]
                    w, h = x2 - x1, y2 - y1

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Save to text file
                    file.write(f'{frame_number},{obj_id},{x1},{y1},{w},{h},{conf:.2f},-1,-1,-1\n')

                out.write(img)
                cv2.imshow('Video', img)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_number += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


# -- Window Analysis

# -- add a function that converts numpy array to dataframe *


class window_analysis:
    
    def __init__(self):
        # Initialize constants
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians

    @staticmethod
    def text_to_dataframe(file_path):
        
        dataframe = pd.read_csv(file_path, sep=',', header=None, names=['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])
        for df in [dataframe]:
            df['x2'] = df['x1'] + df['width']
            df['y2'] = df['y1'] + df['height']
            df['CenterX'] = (df['x1'] + df['x2']) / 2
            df['CenterY'] = (df['y1'] + df['y2']) / 2
            
        return df
        
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

    def calculate_angles(self, xc, yc):
        xcs = np.array(xc)
        ycs = np.array(yc)
        xd = xcs - self.width / 2  # Difference in x-coordinates
        yd = self.height / 2 - ycs  # Difference in y-coordinates
        theta_h = xd * self.happ  # Horizontal angle in radians
        horiz_angle = np.degrees(theta_h)  # Convert to degrees
        return horiz_angle.tolist()

    def get_Accuracy(self, gt_df, pred_df):
        gt_count = gt_df['id'].nunique()
        pred_count = pred_df['id'].nunique()
        error_rate = abs(pred_count - gt_count) / gt_count
        if error_rate > 1:
            return 0  # Returns 0% accuracy if error rate exceeds 100%
        else:
            accuracy = (1 - error_rate) * 100
            return round(accuracy, 2)

    def calc_accuracy(self, y1, y2, df_gt, df_pred):
        reg_gt = df_gt[((df_gt['CenterY'] > y1) & (df_gt['CenterY'] < y2))]
        reg_pred = df_pred[((df_pred['CenterY'] > y1) & (df_pred['CenterY'] < y2))]
        accuracy = self.get_Accuracy(reg_gt, reg_pred)
        return accuracy

    def calculate_accuracies(self, y1, y2, gts, preds):
        '''
        gts, preds : list of dataframes
        y1, y2 : list of lower and upper 'y' limits of windows
        
        '''
        accuracies = []
        for gt, pred in zip(gts, preds):
            accuracy = [self.calc_accuracy(y1[i], y2[i], gt, pred) for i in range(len(y1))] # lengths of y1 and y2 are equal
            accuracies.append(accuracy)
        return accuracies

    def create_dataframe(self, accuracies, vertical_fov, horizontal_va):
        columns = {f'Count_accuracy{i+1}': acc for i, acc in enumerate(accuracies)}
        columns.update({
            'vertical_fov': vertical_fov,
            'hva1': horizontal_va[0],
            'hva2': horizontal_va[1],
            'hva3': horizontal_va[2],
            'hva4': horizontal_va[3],
            'hva5': horizontal_va[4],
            'hva6': horizontal_va[5],
            'hva7': horizontal_va[6],
            'hva8': horizontal_va[7]
        })
        return pd.DataFrame(columns)

    def plot_8rows(self, wc_df):
        fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(24, 4))
        x_limits = [(0, 70), (0, 70), (0, 70), (0, 70), (0, 70), (0, 70), (0, 70), (0, 70)]
        y_limits = [(0, 110), (0, 110), (0, 110), (0, 110), (0, 110), (0, 110), (0, 110), (0, 110)]
        for i in range(8):
            axes[i].plot(wc_df['vertical_fov'], wc_df[f'Count_accuracy{i+1}'], marker='s', linestyle='-', color='b', markersize=3)
            axes[i].set_title(f'Row {i+1} ({wc_df[f'hva{i+1}'].iloc[0]:.2f}°)')
            axes[i].grid(True)
            if x_limits:
                axes[i].set_xlim(x_limits[i])
            if y_limits:
                axes[i].set_ylim(y_limits[i])
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        fig.text(0.5, -0.03, 'Vertical Field of View (VFOV)', ha='center', va='center')
        fig.text(0, 0.5, 'Counting Accuracy', ha='center', va='center', rotation='vertical')
        plt.tight_layout()
        plt.show()

