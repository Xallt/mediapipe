// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#ifndef RUN_GRAPH_GPU_H
#define RUN_GRAPH_GPU_H

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>

enum HandLandmark {
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP = 13,
    RING_FINGER_PIP = 14,
    RING_FINGER_DIP = 15,
    RING_FINGER_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20
};

typedef std::pair<HandLandmark, HandLandmark> LandmarkPair;

const std::vector<LandmarkPair> landmarkConnections = {
    {HandLandmark::WRIST, HandLandmark::THUMB_CMC},
    {HandLandmark::THUMB_CMC, HandLandmark::THUMB_MCP},
    {HandLandmark::THUMB_MCP, HandLandmark::THUMB_IP},
    {HandLandmark::THUMB_IP, HandLandmark::THUMB_TIP},
    {HandLandmark::WRIST, HandLandmark::INDEX_FINGER_MCP},
    {HandLandmark::INDEX_FINGER_MCP, HandLandmark::INDEX_FINGER_PIP},
    {HandLandmark::INDEX_FINGER_PIP, HandLandmark::INDEX_FINGER_DIP},
    {HandLandmark::INDEX_FINGER_DIP, HandLandmark::INDEX_FINGER_TIP},
    {HandLandmark::INDEX_FINGER_MCP, HandLandmark::MIDDLE_FINGER_MCP},
    {HandLandmark::MIDDLE_FINGER_MCP, HandLandmark::MIDDLE_FINGER_PIP},
    {HandLandmark::MIDDLE_FINGER_PIP, HandLandmark::MIDDLE_FINGER_DIP},
    {HandLandmark::MIDDLE_FINGER_DIP, HandLandmark::MIDDLE_FINGER_TIP},
    {HandLandmark::MIDDLE_FINGER_MCP, HandLandmark::RING_FINGER_MCP},
    {HandLandmark::RING_FINGER_MCP, HandLandmark::RING_FINGER_PIP},
    {HandLandmark::RING_FINGER_PIP, HandLandmark::RING_FINGER_DIP},
    {HandLandmark::RING_FINGER_DIP, HandLandmark::RING_FINGER_TIP},
    {HandLandmark::RING_FINGER_MCP, HandLandmark::PINKY_MCP},
    {HandLandmark::WRIST, HandLandmark::PINKY_MCP},
    {HandLandmark::PINKY_MCP, HandLandmark::PINKY_PIP},
    {HandLandmark::PINKY_PIP, HandLandmark::PINKY_DIP},
    {HandLandmark::PINKY_DIP, HandLandmark::PINKY_TIP}
};

struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> presence;
    std::vector<float> visibility;
};

class SimpleMPPGraphRunner {
   public:
    SimpleMPPGraphRunner();
    ~SimpleMPPGraphRunner();
    bool RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path);
    bool InitMPPGraph(std::string calculator_graph_config_file);
    bool ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, bool &landmark_presence);

   private:
    void* runnerVoid;
};

#endif  // RUN_GRAPH_GPU_H