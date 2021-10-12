import numpy as np
from .face_utilities import Face_utilities
from .signal_processing import Signal_processing
import cv2


class CalcHR():
        def __init__(self,dir):
                self.fu = Face_utilities()
                self.sp = Signal_processing()
                self.cap = cv2.VideoCapture(dir)
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.BUFFER_SIZE = 100
                self.times = []
                self.fft_of_interest = []
                self.freqs_of_interest = []
                self.data_buffer = []  # state
                self.indexFrame = 0  # state
                self.lbpm = []
                

        def getHR(self):
                while True:
                        ret, frame = self.cap.read()
                        if frame is None: 
                                print("End of video") 
                                break
                        ret_process = self.fu.no_age_gender_face_process(frame, "68")
                        if ret_process is None:
                                continue
                        rects, face, shape, aligned_face, aligned_shape = ret_process
                        # for signal_processing
                        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
                        green_val = self.sp.extract_color(ROIs)
                        self.data_buffer.append(green_val)
                        self.times.append((1.0/self.video_fps)*self.indexFrame)
                        L = len(self.data_buffer)
                        if L > self.BUFFER_SIZE:
                                self.data_buffer = self.data_buffer[-self.BUFFER_SIZE:]
                                self.times = self.times[-self.BUFFER_SIZE:]
                                L = self.BUFFER_SIZE
                        if L == 100:
                                fps = float(L) / (self.times[-1] - self.times[0])
                                detrended_data = self.sp.signal_detrending(self.data_buffer)
                                interpolated_data = self.sp.interpolation(detrended_data, self.times)
                                normalized_data = self.sp.normalization(interpolated_data)
                                fft_of_interest, freqs_of_interest = self.sp.fft(normalized_data, fps)
                                max_arg = np.argmax(fft_of_interest)
                                bpm = freqs_of_interest[max_arg]
                                self.lbpm.append(bpm)
                                print(f"{bpm}   ***************\n")
                        self.indexFrame = self.indexFrame + 1
                return np.average(np.array(self.lbpm))
