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
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include <opencv2/opencv.hpp>

class SimpleMPPGraphRunner {
public:
    SimpleMPPGraphRunner();
    absl::Status RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path);
};
class SimpleVideoReader {
public:
   SimpleVideoReader();
   ~SimpleVideoReader();
   absl::Status init(std::string url, bool flipFrame = false);
   absl::Status getFrame(cv::Mat &frame);
   double getFPS();
   void setFPS(double fps);
   void setResolution(int width, int height);
private:
    cv::VideoCapture capture;
    bool flipFrame;
};

#endif // RUN_GRAPH_GPU_H