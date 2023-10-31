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
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/resource_util.h"

#include "run_graph_gpu.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

absl::Status CreateGraphFromFile(std::string calculator_graph_config_file, mediapipe::CalculatorGraph &graph) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      calculator_graph_config_file,
      &calculator_graph_config_contents));
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  MP_RETURN_IF_ERROR(graph.Initialize(config));
  
  return absl::OkStatus();
}

// SimpleVideoWrapper class for video input from webcam / video file
SimpleVideoReader::SimpleVideoReader() {}

SimpleVideoReader::~SimpleVideoReader() {
    if (capture.isOpened()) capture.release();
}

absl::Status SimpleVideoReader::init(std::string url, bool flipFrame) {
    if (url == "0")
        capture.open(0);
    else
        capture.open(url);
    
    if (!capture.isOpened()) {
        return absl::InternalError(
            absl::StrCat("Failed to open video capture for URL: ", url));
    }

    this->flipFrame = flipFrame;
    return absl::OkStatus();
}

absl::Status SimpleVideoReader::getFrame(cv::Mat &frame) {
    bool ret = capture.read(frame);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
    if (!ret) {
        return absl::InternalError("Failed to read frame from capture.");
    }
    
    if (flipFrame) cv::flip(frame, frame, 1);
    return absl::OkStatus();
}

double SimpleVideoReader::getFPS() {
    return capture.get(cv::CAP_PROP_FPS);
}

void SimpleVideoReader::setFPS(double fps) {
    capture.set(cv::CAP_PROP_FPS, fps);
}

void SimpleVideoReader::setResolution(int width, int height) {
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

class MPPGraphRunner {
public:
	absl::Status RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path) {

        ABSL_LOG(INFO) << "Initialize the calculator graph.";

        MP_RETURN_IF_ERROR(CreateGraphFromFile(calculator_graph_config_file, graph));

        ABSL_LOG(INFO) << "Initialize the GPU.";
        MP_ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
        MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));

        gpu_helper.InitializeForTest(graph.GetGpuResources().get());

        ABSL_LOG(INFO) << "Initialize the camera or load the video.";
        if (input_video_path.empty())
            input_video_path = "0";
        SimpleVideoReader capture;
        MP_RETURN_IF_ERROR(capture.init(input_video_path, input_video_path == "0"));

        cv::VideoWriter writer;
        const bool save_video = !output_video_path.empty();
        if (!save_video) {
            cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.setResolution(640, 480);
    capture.setFPS(30);
#endif
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame;
    MP_RETURN_IF_ERROR(capture.getFrame(camera_frame));

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, this]() -> absl::Status {
            // Convert ImageFrame to GpuBuffer.
            auto texture = this->gpu_helper.CreateSourceTexture(*input_frame.get());
            auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
            glFlush();
            texture.Release();
            // Send GPU image packet into the graph.
            // MP_RETURN_IF_ERROR(
            auto status = this->graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(gpu_frame.release())
                                  .At(mediapipe::Timestamp(frame_timestamp_us)));
            ABSL_LOG(INFO) << status;
            // );
            return absl::OkStatus();
        })
      );

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) {
      ABSL_LOG(INFO) << "Break on empty packet.";
      break;
    }
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext(
            [&packet, &output_frame, this]() -> absl::Status {
                auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
                auto texture = this->gpu_helper.CreateSourceTexture(gpu_frame);
                output_frame = absl::make_unique<mediapipe::ImageFrame>(
                    mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                    gpu_frame.width(), gpu_frame.height(),
                    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                this->gpu_helper.BindFramebuffer(texture);
                const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
                    gpu_frame.format(), 0, gpu_helper.GetGlVersion());
                glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                            info.gl_type, output_frame->MutablePixelData());
                glFlush();
                texture.Release();
                return absl::OkStatus();
            }
        )
    );

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    if (output_frame_mat.channels() == 4)
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
    else
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Prepare video writer.";
        writer.open(output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.getFPS(), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  ABSL_LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
	}
private:
	mediapipe::CalculatorGraph graph;
	mediapipe::GlCalculatorHelper gpu_helper;
};

SimpleMPPGraphRunner::SimpleMPPGraphRunner() {}
absl::Status SimpleMPPGraphRunner::RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path) {
	MPPGraphRunner runner;
	return runner.RunMPPGraph(calculator_graph_config_file, input_video_path, output_video_path);
}