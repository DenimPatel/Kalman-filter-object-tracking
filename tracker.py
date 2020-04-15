import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque

class TrackingObject(object):
	"""
	TrackingObject: maintains and updates all the information 
	for one tracked object
	"""
	def __init__(self, detection, trackId):
		super(TrackingObject, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.trace = deque(maxlen=20)
		self.prediction = detection.reshape(1,2)
		self.trackId = trackId
		self.skipped_frames = 0

	def predict_and_correct(self, measurement):
		self.prediction = np.array(self.KF.predict()).reshape(1,2)
		self.KF.correct(np.matrix(measurement).reshape(2,1))


class Tracker(object):
	"""
	Tracker: Contains all infromation 
	"""
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.totalObjects = 0
		self.objects_to_track = []

	def update(self, detections):
		"""
		updates the tracker based on new detections
		performs:
			additions of new objects
			Data assiciation: assign measurement to correct object
			deletion of old objects
			predtion and updation of each object
		"""
		# if no objects to track, add all in the tracker list
		if len(self.objects_to_track) == 0:
			for i in range(detections.shape[0]):
				track = TrackingObject(detections[i], self.totalObjects)
				self.totalObjects += 1
				self.objects_to_track.append(track)

		dist = []

		# iterate over all objects and find cost from an object to each measurement
		for i in range(len(self.objects_to_track)):
			diff = np.linalg.norm(self.objects_to_track[i].prediction - detections.reshape(-1,2), axis=1)
			dist.append(diff)

		
		dist = np.array(dist)*0.1
		row, col = linear_sum_assignment(dist) # row, col contains object and detection pair
		assignment = [-1]*len(self.objects_to_track)

		for i in range(len(row)):
			assignment[row[i]] = col[i]

		un_assigned_tracks = []
		for i in range(len(assignment)):
			if (assignment[i] != -1 and dist[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					un_assigned_tracks.append(i)
			else:
				self.objects_to_track[i].skipped_frames += 1

		del_tracks = []
		for i in range(len(self.objects_to_track)):
			if self.objects_to_track[i].skipped_frames > self.max_frame_skipped :
				del_tracks.append(i)
				del self.objects_to_track[i]
				del assignment[i]



		# if len(del_tracks) > 0:
		# 	for i in range(len(del_tracks)):
		# 		del self.objects_to_track[i]
		# 		del assignment[i]

		# add new detection to the system
		for i in range(len(detections)):
			if i not in assignment:
				track = TrackingObject(detections[i], self.trackId)
				self.totalObjects += 1
				self.objects_to_track.append(track)

		# update tracks 
		for i in range(len(assignment)):
			if (assignment[i] != -1):
				self.objects_to_track[i].skipped_frames = 0
				self.objects_to_track[i].predict_and_correct(detections[assignment[i]])
			self.objects_to_track[i].trace.append(self.objects_to_track[i].prediction)